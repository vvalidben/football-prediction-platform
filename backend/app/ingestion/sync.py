"""Idempotent synchronisation of provider data into the database.

For each operation the flow is: fetch (provider-neutral records) → normalise →
validate → resolve each record to an internal row through the
`<entity>_external_ids` mapping tables → insert or update → record data-quality
issues → log the run.

Guarantees:
- **Idempotent.** A record already mapped for this provider is updated in place, never
  inserted again, so running a sync twice creates no duplicates.
- **Isolated failures.** Each record is applied in its own savepoint. A record that
  cannot be stored is counted as failed, the rest still commit, and the run is
  `partial`.
- **Nothing disappears silently.** Every rejected or questionable record produces a
  `data_quality_issues` row. Rejected records keep their full content in the issue.
  Nothing is ever deleted: records a provider stops listing stay stored, and issues
  are resolved rather than removed once a re-check no longer finds them.
- **Serialised per provider.** A transaction-scoped advisory lock stops two syncs of
  the same provider from racing to insert the same new entity.
- **Missing data does not erase data.** Descriptive fields (logo, venue, referee, …)
  and statistics a provider leaves empty keep their stored value. State fields
  (status, kickoff, scores) always follow the provider.
- **Chronology is preserved.** Every match keeps its exact kickoff as a UTC
  timestamp; kickoffs without a timezone or far outside reality are rejected.
"""

import dataclasses
import logging
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import exists, func, select, text, update
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.orm import InstrumentedAttribute, Session, sessionmaker

from app.ingestion import normalize
from app.ingestion.runs import IngestionRunLogger, Outcome, RunStats
from app.models import (
    Competition,
    CompetitionExternalId,
    DataQualityIssue,
    DataSource,
    Match,
    MatchExternalId,
    Season,
    SeasonExternalId,
    Team,
    TeamExternalId,
    TeamMatchStatistics,
)
from app.models.enums import (
    IngestionStatus,
    IssueEntity,
    IssueSeverity,
    IssueType,
    MatchStatus,
)
from app.providers.base import FetchResult, FootballDataProvider
from app.providers.errors import ProviderError
from app.providers.schemas import (
    ExternalId,
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
    ScorePair,
)
from app.quality import rules
from app.quality.issues import IssueDraft, IssueStore, ingestion_detector

logger = logging.getLogger(__name__)

STATISTICS_DETECTOR = ingestion_detector("team_statistics")

#: First key of the two-key advisory lock; the second is the data source id.
ADVISORY_LOCK_NAMESPACE = 0x1F00_7BA1

SeasonSelection = Literal["current", "all"] | Collection[ExternalId]
STATISTIC_FIELDS = tuple(
    name
    for name in ProviderTeamMatchStatistics.model_fields
    if name not in {"match_external_id", "team_external_id", "extra"}
)


class RecordError(Exception):
    """A single record cannot be applied; the run continues with the others."""

    def __init__(self, message: str, issue: IssueDraft | None = None) -> None:
        super().__init__(message)
        self.issue = issue


class UnresolvedReferenceError(RecordError):
    """A record refers to an entity that has not been synchronised yet."""


class RecordConflictError(RecordError):
    """A record contradicts data already stored and needs manual review."""


def _set_fields(entity: object, values: dict[str, Any], *, keep_when_none: Iterable[str]) -> bool:
    """Assign changed values; return whether anything changed.

    Fields in `keep_when_none` are left alone when the new value is None, so a
    provider omitting optional data never wipes what is already stored.
    """
    keep = set(keep_when_none)
    changed = False
    for name, value in values.items():
        if value is None and name in keep:
            continue
        if getattr(entity, name) != value:
            setattr(entity, name, value)
            changed = True
    return changed


def _find_by_external_id[E](
    session: Session,
    entity: type[E],
    mapping: type[Any],
    mapping_fk: InstrumentedAttribute[int],
    source_id: int,
    external_id: str,
) -> E | None:
    return session.scalars(
        select(entity)
        .join(mapping, mapping_fk == entity.id)  # type: ignore[attr-defined]
        .where(mapping.data_source_id == source_id, mapping.external_id == external_id)
    ).one_or_none()


def _record_issue(
    issue_type: IssueType,
    severity: IssueSeverity,
    entity: IssueEntity,
    external_id: str | None,
    message: str,
    **details: Any,
) -> IssueDraft:
    return IssueDraft(
        issue_type=issue_type,
        severity=severity,
        entity_type=entity,
        external_id=external_id,
        key=issue_type.value,
        message=message,
        details=details,
    )


@dataclass
class _Operation[R]:
    """How one sync operation fetches, normalises, applies and audits records."""

    job: str
    entity: IssueEntity
    parameters: dict[str, Any]
    fetch: Callable[[], FetchResult[R]]
    normalise: Callable[[R], R]
    apply: Callable[[Session, int, R], Outcome]
    #: Extra issues derived from the whole fetch (e.g. matches without statistics),
    #: plus the (entity, external id) pairs those checks covered.
    after_fetch: Callable[[FetchResult[R]], tuple[list[IssueDraft], set[tuple[IssueEntity, str]]]]
    identify: Callable[[R], str] = lambda record: str(record.external_id)  # type: ignore[attr-defined]


@dataclass
class ProviderSync:
    """Synchronises one provider's competitions, seasons, teams, matches and
    team statistics."""

    provider: FootballDataProvider
    session_factory: sessionmaker[Session]
    run_logger: IngestionRunLogger | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self._runs = self.run_logger or IngestionRunLogger(self.session_factory)
        self.issues = IssueStore(self.session_factory, clock=self.clock)
        self._source_id: int | None = None
        #: Per season external ID: the provider's claim that statistics exist.
        self.season_statistics: dict[str, bool | None] = {}
        self._warnings: list[IssueDraft] = []

    # -- public operations ---------------------------------------------------------

    def sync_competitions(self, only: Collection[ExternalId] | None = None) -> RunStats:
        """Synchronise competitions, optionally restricted to the given external IDs."""
        wanted = set(only) if only is not None else None

        def fetch() -> FetchResult[ProviderCompetition]:
            result = self.provider.get_competitions()
            if wanted is None:
                return result
            return FetchResult(
                [c for c in result.records if c.external_id in wanted],
                [r for r in result.rejected if r.external_id in wanted],
            )

        return self._run(
            _Operation(
                job="competitions",
                entity=IssueEntity.COMPETITION,
                parameters={"competitions": sorted(wanted)} if wanted is not None else {},
                fetch=fetch,
                normalise=normalize.competition,
                apply=self._apply_competition,
                after_fetch=_no_extra_issues,
            )
        )

    def sync_seasons(self, competition_id: ExternalId) -> RunStats:
        def fetch() -> FetchResult[ProviderSeason]:
            result = self.provider.get_seasons(competition_id)
            for season in result.records:
                self.season_statistics[season.external_id] = season.statistics_available
            return result

        return self._run(
            _Operation(
                job="seasons",
                entity=IssueEntity.SEASON,
                parameters={"competition": competition_id},
                fetch=fetch,
                normalise=lambda record: self._normalise_season(record, competition_id),
                apply=self._apply_season,
                after_fetch=_no_extra_issues,
            )
        )

    def sync_teams(self, competition_id: ExternalId, season_id: ExternalId) -> RunStats:
        return self._run(
            _Operation(
                job="teams",
                entity=IssueEntity.TEAM,
                parameters={"competition": competition_id, "season": season_id},
                fetch=lambda: self.provider.get_teams(competition_id, season_id),
                normalise=normalize.team,
                apply=self._apply_team,
                after_fetch=_no_extra_issues,
            )
        )

    def sync_matches(self, competition_id: ExternalId, season_id: ExternalId) -> RunStats:
        return self._run(
            _Operation(
                job="matches",
                entity=IssueEntity.MATCH,
                parameters={"competition": competition_id, "season": season_id},
                fetch=lambda: self.provider.get_matches(competition_id, season_id),
                normalise=lambda record: self._normalise_match(record, competition_id, season_id),
                apply=self._apply_match,
                after_fetch=_no_extra_issues,
            )
        )

    def sync_team_statistics(
        self,
        competition_id: ExternalId,
        season_id: ExternalId,
        *,
        retry_missing: bool = False,
    ) -> RunStats:
        """Fetch statistics for this season's finished matches that have none yet.

        Matches the provider returned no statistics for are recorded as
        `missing_statistics` issues and skipped by later runs (saving quota) unless
        `retry_missing` is set.
        """
        wanted = self._matches_needing_statistics(season_id, retry_missing=retry_missing)

        def fetch() -> FetchResult[ProviderTeamMatchStatistics]:
            if not wanted:
                return FetchResult()
            return self.provider.get_team_statistics(wanted)

        def missing(
            result: FetchResult[ProviderTeamMatchStatistics],
        ) -> tuple[list[IssueDraft], set[tuple[IssueEntity, str]]]:
            covered = {record.match_external_id for record in result.records}
            covered |= {
                (r.external_id or "").split(":")[0] for r in result.rejected if r.external_id
            }
            drafts = [
                _record_issue(
                    IssueType.MISSING_STATISTICS,
                    IssueSeverity.INFO,
                    IssueEntity.MATCH,
                    match_id,
                    "provider returned no statistics for this finished match",
                    season=season_id,
                )
                for match_id in wanted
                if match_id not in covered
            ]
            return drafts, {(IssueEntity.MATCH, match_id) for match_id in wanted}

        return self._run(
            _Operation(
                job="team_statistics",
                entity=IssueEntity.TEAM_MATCH_STATISTICS,
                parameters={
                    "competition": competition_id,
                    "season": season_id,
                    "matches": len(wanted),
                },
                fetch=fetch,
                normalise=normalize.team_statistics,
                apply=self._apply_team_statistics,
                after_fetch=missing,
            )
        )

    def sync_competition(
        self, competition_id: ExternalId, seasons: SeasonSelection = "current"
    ) -> list[RunStats]:
        """Refresh one competition: itself, its seasons, then teams and matches of
        the selected seasons (statistics are imported by `HistoricalImport`).

        A provider failure on one season is logged (as a failed run) and the next
        season is still attempted. Failures fetching the competition or its season
        list stop the refresh, since nothing after them can succeed.
        """
        runs = [self.sync_competitions(only=[competition_id])]
        runs.append(self.sync_seasons(competition_id))
        for season_id in self.select_seasons(competition_id, seasons):
            for operation in (self.sync_teams, self.sync_matches):
                try:
                    runs.append(operation(competition_id, season_id))
                except ProviderError as exc:
                    logger.error(
                        "Skipping rest of season %s of %s: %s", season_id, competition_id, exc
                    )
                    break
        return runs

    def select_seasons(
        self,
        competition_id: str,
        selection: SeasonSelection,
        *,
        first_year: int | None = None,
        last_year: int | None = None,
    ) -> list[str]:
        """Season external IDs for a competition, oldest first, from the database.

        `first_year`/`last_year` filter on the season's starting year (from its
        start date, or the first year in its name).
        """
        if selection not in ("current", "all"):
            return list(selection)
        source_id = self.data_source_id()
        query = (
            select(SeasonExternalId.external_id, Season.name, Season.start_date, Season.is_current)
            .join(Season, SeasonExternalId.season_id == Season.id)
            .join(
                CompetitionExternalId,
                (CompetitionExternalId.competition_id == Season.competition_id)
                & (CompetitionExternalId.data_source_id == source_id),
            )
            .where(
                SeasonExternalId.data_source_id == source_id,
                CompetitionExternalId.external_id == competition_id,
            )
        )
        with self.session_factory() as session:
            rows = session.execute(query).all()
        chosen = []
        for external_id, name, start_date, is_current in rows:
            year = start_date.year if start_date else _leading_year(name)
            if selection == "current" and not is_current:
                continue
            if first_year is not None and (year is None or year < first_year):
                continue
            if last_year is not None and (year is None or year > last_year):
                continue
            chosen.append((year or 0, external_id))
        return [external_id for _, external_id in sorted(chosen)]

    def record_season_without_statistics(self, season_id: ExternalId) -> None:
        """Note that the provider claims no statistics for a season (not an error)."""
        self.issues.record(
            [
                _record_issue(
                    IssueType.MISSING_STATISTICS,
                    IssueSeverity.INFO,
                    IssueEntity.SEASON,
                    season_id,
                    "provider reports no match statistics for this season",
                )
            ],
            detector=STATISTICS_DETECTOR,
            data_source_id=self.data_source_id(),
        )
        logger.info("Season %s: provider offers no statistics; skipped", season_id)

    def clear_season_statistics_gap(self, season_id: ExternalId) -> None:
        self.issues.resolve_absent(
            detector=STATISTICS_DETECTOR,
            present=(),
            data_source_id=self.data_source_id(),
            scope={(IssueEntity.SEASON, season_id)},
        )

    def data_source_id(self) -> int:
        """Get or create this provider's `data_sources` row."""
        if self._source_id is not None:
            return self._source_id
        with self.session_factory() as session, session.begin():
            source = session.scalars(
                select(DataSource).where(DataSource.code == self.provider.code)
            ).one_or_none()
            if source is None:
                source = DataSource(
                    code=self.provider.code,
                    name=self.provider.name,
                    base_url=self.provider.base_url,
                )
                session.add(source)
            else:
                _set_fields(
                    source,
                    {"name": self.provider.name, "base_url": self.provider.base_url},
                    keep_when_none={"base_url"},
                )
            session.flush()
            self._source_id = source.id
        return self._source_id

    # -- run orchestration -------------------------------------------------------------

    def _run[R](self, operation: _Operation[R]) -> RunStats:
        source_id = self.data_source_id()
        with self._runs.track(source_id, operation.job, operation.parameters) as stats:
            result = operation.fetch()
            stats.fetched = len(result.records) + len(result.rejected)
            issues: list[IssueDraft] = []
            checked: set[tuple[IssueEntity, str]] = set()

            for rejected in result.rejected:
                stats.fail(rejected.external_id, f"malformed: {rejected.reason}")
                issues.append(
                    _record_issue(
                        IssueType.MALFORMED_RECORD,
                        IssueSeverity.ERROR,
                        operation.entity,
                        rejected.external_id,
                        f"malformed provider record: {rejected.reason}",
                    )
                )
                if rejected.external_id:
                    checked.add((operation.entity, rejected.external_id))

            records = self._drop_duplicate_ids(operation, result.records, stats, issues)
            extra_issues, extra_scope = operation.after_fetch(result)
            issues.extend(extra_issues)
            checked |= extra_scope

            with self.session_factory() as session, session.begin():
                session.execute(
                    text("SELECT pg_advisory_xact_lock(:namespace, :source)"),
                    {"namespace": ADVISORY_LOCK_NAMESPACE, "source": source_id},
                )
                for record in records:
                    reference = operation.identify(record)
                    checked.add((operation.entity, reference))
                    self._warnings = []
                    try:
                        normalised = operation.normalise(record)
                        with session.begin_nested():
                            stats.record(operation.apply(session, source_id, normalised))
                        issues.extend(self._warnings)
                    except normalize.NormalizationError as exc:
                        stats.fail(reference, str(exc))
                        issues.append(
                            _record_issue(
                                IssueType.MALFORMED_RECORD,
                                IssueSeverity.ERROR,
                                operation.entity,
                                reference,
                                f"not stored: {exc}",
                                provider_record=_dump(record),
                            )
                        )
                    except RecordError as exc:
                        stats.fail(reference, str(exc))
                        if exc.issue is not None:
                            issues.append(exc.issue)
                    except (IntegrityError, DataError) as exc:
                        stats.fail(reference, f"rejected by database: {exc.orig}")
                        issues.append(
                            _record_issue(
                                IssueType.MALFORMED_RECORD,
                                IssueSeverity.ERROR,
                                operation.entity,
                                reference,
                                f"rejected by database: {exc.orig}",
                                provider_record=_dump(record),
                            )
                        )

            detector = ingestion_detector(operation.job)
            present = self.issues.record(
                issues, detector=detector, data_source_id=source_id, run_id=stats.run_id
            )
            self.issues.resolve_absent(
                detector=detector, present=present, data_source_id=source_id, scope=checked
            )
        return stats

    def _drop_duplicate_ids[R](
        self,
        operation: _Operation[R],
        records: Sequence[R],
        stats: RunStats,
        issues: list[IssueDraft],
    ) -> list[R]:
        """Keep the first record for each provider ID; flag and skip the others."""
        first: dict[str, R] = {}
        for record in records:
            reference = operation.identify(record)
            if reference not in first:
                first[reference] = record
                continue
            stats.fail(reference, "duplicate provider id in one response; later copy skipped")
            issues.append(
                _record_issue(
                    IssueType.DUPLICATE_PROVIDER_ID,
                    IssueSeverity.WARNING,
                    operation.entity,
                    reference,
                    "provider returned this id more than once in one response",
                    kept=_dump(first[reference]),
                    skipped=_dump(record),
                    identical=_dump(first[reference]) == _dump(record),
                )
            )
        return list(first.values())

    def _warn(self, issue: IssueDraft) -> None:
        self._warnings.append(issue)

    # -- normalisation with context ------------------------------------------------------

    def _normalise_season(self, record: ProviderSeason, competition_id: str) -> ProviderSeason:
        season, recognised = normalize.season(record)
        if season.competition_external_id != competition_id:
            raise RecordError(
                f"season belongs to competition {season.competition_external_id!r}, "
                f"not {competition_id!r}",
                _record_issue(
                    IssueType.INCONSISTENT_COMPETITION,
                    IssueSeverity.ERROR,
                    IssueEntity.SEASON,
                    season.external_id,
                    "season listed under another competition; not stored",
                    provider_record=_dump(record),
                    requested_competition=competition_id,
                ),
            )
        if not recognised:
            self._warn(
                _record_issue(
                    IssueType.INCOMPLETE_RECORD,
                    IssueSeverity.WARNING,
                    IssueEntity.SEASON,
                    season.external_id,
                    f"season name {season.name!r} is not in a recognised format; kept as is",
                )
            )
        if season.start_date is None or season.end_date is None:
            self._warn(
                IssueDraft(
                    issue_type=IssueType.INCOMPLETE_RECORD,
                    severity=IssueSeverity.INFO,
                    entity_type=IssueEntity.SEASON,
                    external_id=season.external_id,
                    key="season_dates_missing",
                    message="season has no start or end date",
                )
            )
        return season

    def _normalise_match(
        self, record: ProviderMatch, competition_id: str, season_id: str
    ) -> ProviderMatch:
        match = normalize.match(record)
        if (match.competition_external_id, match.season_external_id) != (
            competition_id,
            season_id,
        ):
            raise RecordError(
                f"match belongs to {match.competition_external_id}/{match.season_external_id}, "
                f"not {competition_id}/{season_id}",
                _record_issue(
                    IssueType.INCONSISTENT_COMPETITION,
                    IssueSeverity.ERROR,
                    IssueEntity.MATCH,
                    match.external_id,
                    "match listed under another competition or season; not stored",
                    provider_record=_dump(record),
                    requested={"competition": competition_id, "season": season_id},
                ),
            )
        corrupt = rules.kickoff_is_corrupt(match, self.clock())
        if corrupt is not None:
            raise RecordError(corrupt.message, corrupt)
        return match

    # -- per-record upserts --------------------------------------------------------------

    def _resolve[E](
        self,
        session: Session,
        entity: type[E],
        mapping: type[Any],
        mapping_fk: InstrumentedAttribute[int],
        source_id: int,
        external_id: str,
        label: str,
        *,
        referrer: tuple[IssueEntity, str],
    ) -> E:
        found = _find_by_external_id(session, entity, mapping, mapping_fk, source_id, external_id)
        if found is None:
            issue_type = (
                IssueType.MISSING_TEAM if label == "team" else IssueType.INCONSISTENT_COMPETITION
            )
            raise UnresolvedReferenceError(
                f"unknown {label} {external_id!r}; sync it first",
                IssueDraft(
                    issue_type=issue_type,
                    severity=IssueSeverity.ERROR,
                    entity_type=referrer[0],
                    external_id=referrer[1],
                    key=f"unknown_{label}:{external_id}",
                    message=f"references unknown {label} {external_id!r}; not stored",
                    details={f"{label}_external_id": external_id},
                ),
            )
        return found

    def _apply_competition(
        self, session: Session, source_id: int, record: ProviderCompetition
    ) -> Outcome:
        values = {
            "name": record.name,
            "type": record.type,
            "country": record.country,
            "logo_url": record.logo_url,
        }
        competition = _find_by_external_id(
            session,
            Competition,
            CompetitionExternalId,
            CompetitionExternalId.competition_id,
            source_id,
            record.external_id,
        )
        if competition is None:
            competition = Competition(**values)
            competition.external_ids.append(
                CompetitionExternalId(data_source_id=source_id, external_id=record.external_id)
            )
            session.add(competition)
            return "inserted"
        changed = _set_fields(competition, values, keep_when_none={"logo_url", "country"})
        return "updated" if changed else "unchanged"

    def _apply_season(self, session: Session, source_id: int, record: ProviderSeason) -> Outcome:
        competition = self._resolve(
            session,
            Competition,
            CompetitionExternalId,
            CompetitionExternalId.competition_id,
            source_id,
            record.competition_external_id,
            "competition",
            referrer=(IssueEntity.SEASON, record.external_id),
        )
        season = _find_by_external_id(
            session,
            Season,
            SeasonExternalId,
            SeasonExternalId.season_id,
            source_id,
            record.external_id,
        )
        outcome: Outcome = "unchanged"
        if season is None:
            # The season may already exist (e.g. created from another provider).
            season = session.scalars(
                select(Season).where(
                    Season.competition_id == competition.id, Season.name == record.name
                )
            ).one_or_none()
            if season is None:
                season = Season(competition=competition, name=record.name)
                session.add(season)
                outcome = "inserted"
            else:
                outcome = "updated"
            season.external_ids.append(
                SeasonExternalId(data_source_id=source_id, external_id=record.external_id)
            )

        if record.is_current and not season.is_current:
            # At most one current season per competition (partial unique index).
            session.flush()
            session.execute(
                update(Season)
                .where(Season.competition_id == competition.id, Season.id != season.id)
                .values(is_current=False, updated_at=func.now())
            )
        changed = _set_fields(
            season,
            {
                "name": record.name,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "is_current": record.is_current,
            },
            keep_when_none={"start_date", "end_date"},
        )
        if outcome == "unchanged" and changed:
            outcome = "updated"
        return outcome

    def _apply_team(self, session: Session, source_id: int, record: ProviderTeam) -> Outcome:
        values = {
            "name": record.name,
            "short_name": record.short_name,
            "code": record.code,
            "country": record.country,
            "logo_url": record.logo_url,
            "is_national_team": record.is_national_team,
        }
        team = _find_by_external_id(
            session, Team, TeamExternalId, TeamExternalId.team_id, source_id, record.external_id
        )
        if team is None:
            team = Team(**values)
            team.external_ids.append(
                TeamExternalId(data_source_id=source_id, external_id=record.external_id)
            )
            session.add(team)
            return "inserted"
        changed = _set_fields(
            team, values, keep_when_none={"short_name", "code", "country", "logo_url"}
        )
        return "updated" if changed else "unchanged"

    def _apply_match(self, session: Session, source_id: int, record: ProviderMatch) -> Outcome:
        referrer = (IssueEntity.MATCH, record.external_id)
        competition = self._resolve(
            session,
            Competition,
            CompetitionExternalId,
            CompetitionExternalId.competition_id,
            source_id,
            record.competition_external_id,
            "competition",
            referrer=referrer,
        )
        season = self._resolve(
            session,
            Season,
            SeasonExternalId,
            SeasonExternalId.season_id,
            source_id,
            record.season_external_id,
            "season",
            referrer=referrer,
        )
        home, away = (
            self._resolve(
                session,
                Team,
                TeamExternalId,
                TeamExternalId.team_id,
                source_id,
                team_id,
                "team",
                referrer=referrer,
            )
            for team_id in (record.home_team_external_id, record.away_team_external_id)
        )

        def pair(score: ScorePair | None) -> tuple[int | None, int | None]:
            return (score.home, score.away) if score else (None, None)

        home_score, away_score = pair(record.score)
        home_ht, away_ht = pair(record.half_time_score)
        home_pen, away_pen = pair(record.penalty_score)
        values: dict[str, Any] = {
            "competition_id": competition.id,
            "season_id": season.id,
            "home_team_id": home.id,
            "away_team_id": away.id,
            "kickoff_at": record.kickoff_at,
            "status": record.status,
            "matchday": record.matchday,
            "stage": record.stage,
            "home_score": home_score,
            "away_score": away_score,
            "home_half_time_score": home_ht,
            "away_half_time_score": away_ht,
            "home_penalty_score": home_pen,
            "away_penalty_score": away_pen,
            "venue": record.venue,
            "referee": record.referee,
        }
        descriptive = {"matchday", "stage", "venue", "referee"}

        match = _find_by_external_id(
            session, Match, MatchExternalId, MatchExternalId.match_id, source_id, record.external_id
        )
        outcome: Outcome
        if match is None:
            # Same fixture already stored (e.g. from another provider): link it.
            match = session.scalars(
                select(Match).where(
                    Match.home_team_id == home.id,
                    Match.away_team_id == away.id,
                    Match.kickoff_at == record.kickoff_at,
                )
            ).one_or_none()
            if match is not None and match.season_id != season.id:
                raise RecordConflictError(
                    f"fixture already stored as match {match.id} in another season; "
                    "not linking automatically",
                    _record_issue(
                        IssueType.DUPLICATE_MATCH,
                        IssueSeverity.ERROR,
                        IssueEntity.MATCH,
                        record.external_id,
                        f"same fixture already stored as match {match.id} in another season",
                        existing_match_id=match.id,
                        provider_record=_dump(record),
                    ),
                )
            if match is None:
                match = Match(**{k: v for k, v in values.items() if v is not None})
                session.add(match)
                outcome = "inserted"
            else:
                _set_fields(match, values, keep_when_none=descriptive)
                outcome = "updated"
            match.external_ids.append(
                MatchExternalId(data_source_id=source_id, external_id=record.external_id)
            )
        else:
            changed = _set_fields(match, values, keep_when_none=descriptive)
            outcome = "updated" if changed else "unchanged"

        session.flush()
        for warning in rules.match_warnings(
            record, season_start=season.start_date, season_end=season.end_date, now=self.clock()
        ):
            self._warn(dataclasses.replace(warning, entity_id=match.id))
        return outcome

    def _apply_team_statistics(
        self, session: Session, source_id: int, record: ProviderTeamMatchStatistics
    ) -> Outcome:
        referrer = (IssueEntity.TEAM_MATCH_STATISTICS, record.external_id)
        match = self._resolve(
            session,
            Match,
            MatchExternalId,
            MatchExternalId.match_id,
            source_id,
            record.match_external_id,
            "match",
            referrer=referrer,
        )
        team = self._resolve(
            session,
            Team,
            TeamExternalId,
            TeamExternalId.team_id,
            source_id,
            record.team_external_id,
            "team",
            referrer=referrer,
        )
        if team.id not in (match.home_team_id, match.away_team_id):
            raise RecordError(
                f"team {record.team_external_id!r} did not play match {record.match_external_id!r}",
                _record_issue(
                    IssueType.INCONSISTENT_STATISTICS,
                    IssueSeverity.ERROR,
                    IssueEntity.TEAM_MATCH_STATISTICS,
                    record.external_id,
                    "statistics are for a team that did not play this match; not stored",
                    provider_record=_dump(record),
                ),
            )
        violations = rules.statistics_violations(record)
        if violations:
            raise RecordError(violations[0].message, violations[0])

        values: dict[str, Any] = {name: getattr(record, name) for name in STATISTIC_FIELDS}
        row = session.scalars(
            select(TeamMatchStatistics).where(
                TeamMatchStatistics.match_id == match.id,
                TeamMatchStatistics.team_id == team.id,
                TeamMatchStatistics.data_source_id == source_id,
            )
        ).one_or_none()
        if row is None:
            session.add(
                TeamMatchStatistics(
                    match_id=match.id,
                    team_id=team.id,
                    data_source_id=source_id,
                    extra=dict(record.extra),
                    **{k: v for k, v in values.items() if v is not None},
                )
            )
            return "inserted"
        changed = _set_fields(row, values, keep_when_none=STATISTIC_FIELDS)
        if record.extra and {**row.extra, **record.extra} != row.extra:
            row.extra = {**row.extra, **record.extra}
            changed = True
        return "updated" if changed else "unchanged"

    def _matches_needing_statistics(self, season_id: str, *, retry_missing: bool) -> list[str]:
        source_id = self.data_source_id()
        has_statistics = exists().where(
            TeamMatchStatistics.match_id == Match.id,
            TeamMatchStatistics.data_source_id == source_id,
        )
        known_gap = exists().where(
            DataQualityIssue.data_source_id == source_id,
            DataQualityIssue.detector == STATISTICS_DETECTOR,
            DataQualityIssue.issue_type == IssueType.MISSING_STATISTICS,
            DataQualityIssue.entity_type == IssueEntity.MATCH,
            DataQualityIssue.external_id == MatchExternalId.external_id,
            DataQualityIssue.resolved_at.is_(None),
        )
        query = (
            select(MatchExternalId.external_id)
            .join(Match, MatchExternalId.match_id == Match.id)
            .join(SeasonExternalId, SeasonExternalId.season_id == Match.season_id)
            .where(
                MatchExternalId.data_source_id == source_id,
                SeasonExternalId.data_source_id == source_id,
                SeasonExternalId.external_id == season_id,
                Match.status == MatchStatus.FINISHED,
                ~has_statistics,
            )
            .order_by(Match.kickoff_at)
        )
        if not retry_missing:
            query = query.where(~known_gap)
        with self.session_factory() as session:
            return list(session.scalars(query))


def _no_extra_issues(_: object) -> tuple[list[IssueDraft], set[tuple[IssueEntity, str]]]:
    return [], set()


def _dump(record: object) -> Any:
    model_dump = getattr(record, "model_dump", None)
    return model_dump(mode="json") if model_dump else repr(record)


def _leading_year(name: str) -> int | None:
    digits = name[:4]
    return int(digits) if digits.isdigit() else None


def summarise(runs: Iterable[RunStats]) -> IngestionStatus:
    """Worst status across runs: failed > partial > succeeded."""
    statuses = {run.status for run in runs}
    for status in (IngestionStatus.FAILED, IngestionStatus.PARTIAL):
        if status in statuses:
            return status
    return IngestionStatus.SUCCEEDED
