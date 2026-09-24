"""Database-wide data-quality checks.

Ingestion validates one record at a time. The audit looks at the stored data as a
whole, for problems that only show up across records: near-duplicate fixtures,
teams playing twice at once, possible duplicate teams, kickoffs outside their
season, and statistics that contradict their match.

Every finding becomes (or refreshes) a `data_quality_issues` row with detector
`audit`. Open audit issues that are no longer found are marked resolved. Nothing is
ever deleted or modified in the football tables.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, aliased, sessionmaker

from app.ingestion.normalize import name_key
from app.models import Match, Season, Team, TeamMatchStatistics
from app.models.enums import IssueEntity, IssueSeverity, IssueType, MatchStatus
from app.quality.issues import AUDIT, IssueDraft, IssueStore
from app.quality.rules import MAX_PLAUSIBLE_GOALS, SEASON_DATE_TOLERANCE

#: Same two teams meeting twice within this window is probably one fixture twice.
DUPLICATE_FIXTURE_WINDOW = timedelta(hours=48)
#: A team cannot play two matches this close together.
TEAM_CLASH_WINDOW = timedelta(hours=12)
#: Two teams' possession should add up to about 100%.
POSSESSION_TOLERANCE = 2
#: A scheduled match this long past kickoff probably has a stale status.
STALE_STATUS_AFTER = timedelta(days=3)


@dataclass
class AuditResult:
    issues: list[IssueDraft] = field(default_factory=list)
    resolved: int = 0

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            counts[issue.issue_type.value] += 1
        return dict(sorted(counts.items()))


def _match_issue(
    match_id: int,
    issue_type: IssueType,
    severity: IssueSeverity,
    key: str,
    message: str,
    **details: object,
) -> IssueDraft:
    return IssueDraft(
        issue_type=issue_type,
        severity=severity,
        entity_type=IssueEntity.MATCH,
        entity_id=match_id,
        key=key,
        message=message,
        details=dict(details),
    )


def duplicate_fixtures(session: Session) -> list[IssueDraft]:
    """Same home and away team within 48 hours: the same fixture stored twice?"""
    other = aliased(Match)
    rows = session.execute(
        select(Match.id, other.id, Match.kickoff_at, other.kickoff_at).join(
            other,
            and_(
                other.home_team_id == Match.home_team_id,
                other.away_team_id == Match.away_team_id,
                other.id > Match.id,
                func.abs(func.extract("epoch", other.kickoff_at - Match.kickoff_at))
                < DUPLICATE_FIXTURE_WINDOW.total_seconds(),
            ),
        )
    ).all()
    return [
        _match_issue(
            later,
            IssueType.DUPLICATE_MATCH,
            IssueSeverity.WARNING,
            f"near_duplicate_of:{earlier}",
            f"same fixture as match {earlier} within 48 hours; possible duplicate",
            other_match_id=earlier,
            kickoffs=[a.isoformat(), b.isoformat()],
        )
        for earlier, later, a, b in rows
    ]


def team_clashes(session: Session) -> list[IssueDraft]:
    """A team with two different matches within 12 hours."""
    other = aliased(Match)
    shares_a_team = or_(
        Match.home_team_id == other.home_team_id,
        Match.home_team_id == other.away_team_id,
        Match.away_team_id == other.home_team_id,
        Match.away_team_id == other.away_team_id,
    )
    rows = session.execute(
        select(Match.id, other.id).join(
            other,
            and_(
                other.id > Match.id,
                shares_a_team,
                func.abs(func.extract("epoch", other.kickoff_at - Match.kickoff_at))
                < TEAM_CLASH_WINDOW.total_seconds(),
                # Same pairing is reported by duplicate_fixtures instead.
                ~and_(
                    other.home_team_id == Match.home_team_id,
                    other.away_team_id == Match.away_team_id,
                ),
            ),
        )
    ).all()
    return [
        _match_issue(
            later,
            IssueType.DUPLICATE_MATCH,
            IssueSeverity.WARNING,
            f"team_clash_with:{earlier}",
            f"a team plays both this match and match {earlier} within 12 hours",
            other_match_id=earlier,
        )
        for earlier, later in rows
    ]


def score_problems(session: Session) -> list[IssueDraft]:
    issues = []
    rows = session.execute(
        select(
            Match.id,
            Match.status,
            Match.home_score,
            Match.away_score,
            Match.home_half_time_score,
            Match.away_half_time_score,
            Match.home_penalty_score,
            Match.away_penalty_score,
        ).where(
            or_(
                Match.home_score > MAX_PLAUSIBLE_GOALS,
                Match.away_score > MAX_PLAUSIBLE_GOALS,
                Match.home_half_time_score > Match.home_score,
                Match.away_half_time_score > Match.away_score,
                and_(
                    Match.home_penalty_score.is_not(None),
                    or_(
                        Match.home_score != Match.away_score,
                        Match.home_penalty_score == Match.away_penalty_score,
                    ),
                ),
            )
        )
    ).all()
    for match_id, _, hs, as_, hht, aht, hp, ap in rows:
        score = f"{hs}-{as_}"
        if hs is not None and max(hs, as_) > MAX_PLAUSIBLE_GOALS:
            issues.append(
                _match_issue(
                    match_id,
                    IssueType.IMPOSSIBLE_SCORE,
                    IssueSeverity.WARNING,
                    "implausible_goals",
                    f"implausible score {score}",
                )
            )
        if hht is not None and hs is not None and (hht > hs or aht > as_):
            issues.append(
                _match_issue(
                    match_id,
                    IssueType.IMPOSSIBLE_SCORE,
                    IssueSeverity.ERROR,
                    "half_time_exceeds_full_time",
                    f"half-time {hht}-{aht} exceeds final score {score}",
                )
            )
        if hp is not None and (hs != as_ or hp == ap):
            issues.append(
                _match_issue(
                    match_id,
                    IssueType.IMPOSSIBLE_SCORE,
                    IssueSeverity.ERROR,
                    "invalid_shootout",
                    f"penalty shoot-out {hp}-{ap} after score {score}",
                )
            )
    return issues


def date_problems(session: Session, now: datetime) -> list[IssueDraft]:
    issues = []
    tolerance_days = SEASON_DATE_TOLERANCE.days
    # Calendar date in UTC, independent of the database session's time zone.
    kickoff_day = func.date(func.timezone("UTC", Match.kickoff_at))
    outside = session.execute(
        select(Match.id, Match.kickoff_at, Season.start_date, Season.end_date)
        .join(Season, Season.id == Match.season_id)
        .where(
            or_(
                kickoff_day < Season.start_date - tolerance_days,
                kickoff_day > Season.end_date + tolerance_days,
            )
        )
    ).all()
    for match_id, kickoff, start, end in outside:
        issues.append(
            _match_issue(
                match_id,
                IssueType.INVALID_DATE,
                IssueSeverity.WARNING,
                "outside_season",
                f"kickoff {kickoff.date()} is outside its season ({start} to {end})",
            )
        )
    future_finished = session.scalars(
        select(Match.id).where(Match.status == MatchStatus.FINISHED, Match.kickoff_at > now)
    ).all()
    issues.extend(
        _match_issue(
            match_id,
            IssueType.INVALID_DATE,
            IssueSeverity.ERROR,
            "finished_in_future",
            "match is marked finished but kicks off in the future",
        )
        for match_id in future_finished
    )
    stale = session.scalars(
        select(Match.id).where(
            Match.status == MatchStatus.SCHEDULED,
            Match.kickoff_at < now - STALE_STATUS_AFTER,
        )
    ).all()
    issues.extend(
        _match_issue(
            match_id,
            IssueType.INCOMPLETE_RECORD,
            IssueSeverity.WARNING,
            "stale_scheduled_status",
            "kickoff was more than 3 days ago but the match is still 'scheduled'",
        )
        for match_id in stale
    )
    return issues


def overlapping_seasons(session: Session) -> list[IssueDraft]:
    """Two seasons of one competition whose date ranges overlap."""
    other = aliased(Season)
    rows = session.execute(
        select(Season.id, other.id, Season.name, other.name).join(
            other,
            and_(
                other.competition_id == Season.competition_id,
                other.id > Season.id,
                other.start_date < Season.end_date,
                Season.start_date < other.end_date,
            ),
        )
    ).all()
    return [
        IssueDraft(
            issue_type=IssueType.INCONSISTENT_COMPETITION,
            severity=IssueSeverity.WARNING,
            entity_type=IssueEntity.SEASON,
            entity_id=later,
            key=f"overlaps:{earlier}",
            message=(
                f"season '{later_name}' overlaps season '{earlier_name}' of the same competition"
            ),
            details={"other_season_id": earlier},
        )
        for earlier, later, earlier_name, later_name in rows
    ]


def possible_duplicate_teams(session: Session) -> list[IssueDraft]:
    """Different team rows whose names match once accents, case and affixes go."""
    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for team_id, name, country in session.execute(select(Team.id, Team.name, Team.country)):
        groups[(name_key(name), (country or "").casefold())].append((team_id, name))
    issues = []
    for rows in groups.values():
        if len(rows) < 2:
            continue
        rows.sort()
        first_id = rows[0][0]
        for team_id, name in rows[1:]:
            issues.append(
                IssueDraft(
                    issue_type=IssueType.POSSIBLE_DUPLICATE_TEAM,
                    severity=IssueSeverity.WARNING,
                    entity_type=IssueEntity.TEAM,
                    entity_id=team_id,
                    key=f"same_name_as:{first_id}",
                    message=f"team '{name}' may be the same club as team {first_id}",
                    details={"team_ids": [tid for tid, _ in rows]},
                )
            )
    return issues


def statistics_problems(session: Session) -> list[IssueDraft]:
    issues: list[IssueDraft] = []
    wrong_team = session.execute(
        select(TeamMatchStatistics.id, TeamMatchStatistics.match_id)
        .join(Match, Match.id == TeamMatchStatistics.match_id)
        .where(
            TeamMatchStatistics.team_id != Match.home_team_id,
            TeamMatchStatistics.team_id != Match.away_team_id,
        )
    ).all()
    issues.extend(
        IssueDraft(
            issue_type=IssueType.INCONSISTENT_STATISTICS,
            severity=IssueSeverity.ERROR,
            entity_type=IssueEntity.TEAM_MATCH_STATISTICS,
            entity_id=stats_id,
            key="team_not_in_match",
            message=f"statistics for a team that did not play match {match_id}",
        )
        for stats_id, match_id in wrong_team
    )
    possession = session.execute(
        select(
            TeamMatchStatistics.match_id,
            TeamMatchStatistics.data_source_id,
            func.sum(TeamMatchStatistics.possession),
        )
        .where(TeamMatchStatistics.possession.is_not(None))
        .group_by(TeamMatchStatistics.match_id, TeamMatchStatistics.data_source_id)
        .having(
            func.count() == 2,
            func.abs(func.sum(TeamMatchStatistics.possession) - 100) > POSSESSION_TOLERANCE,
        )
    ).all()
    issues.extend(
        _match_issue(
            match_id,
            IssueType.INCONSISTENT_STATISTICS,
            IssueSeverity.WARNING,
            f"possession_sum:{source_id}",
            f"possession adds up to {total}%, not 100%",
            data_source_id=source_id,
        )
        for match_id, source_id, total in possession
    )
    return issues


CHECKS = (
    duplicate_fixtures,
    team_clashes,
    score_problems,
    overlapping_seasons,
    possible_duplicate_teams,
    statistics_problems,
)


def run_audit(session_factory: sessionmaker[Session], now: datetime | None = None) -> AuditResult:
    """Run every check, record findings and resolve issues no longer found."""
    now = now or datetime.now(UTC)
    result = AuditResult()
    with session_factory() as session:
        for check in CHECKS:
            result.issues.extend(check(session))
        result.issues.extend(date_problems(session, now))
    store = IssueStore(session_factory, clock=lambda: now)
    present = store.record(result.issues, detector=AUDIT)
    result.resolved = store.resolve_absent(detector=AUDIT, present=present)
    return result
