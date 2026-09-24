"""Producing, storing and reading predictions for the API.

One request becomes:

1. **Resolve the match** from `match_id`, or from team names with an optional
   competition and date.
2. **Compute features** as of a cutoff: now for a match that has not kicked off,
   the kickoff for one that has. The cutoff is floored to the minute, so repeated
   requests in the same minute reuse the same feature snapshot, and the snapshot is
   saved (identical values are recognised, not duplicated).
3. **Load the active model version** of the requested (or only) model name, from
   its artifact, cached per version.
4. **Predict**, deriving every market from the one score grid (docs/prediction-engine.md).
5. **Store** the prediction, its full score grid and its over/under lines, and
   return it. A prediction from the same model version and the same feature
   snapshot is reused instead of recomputed unless `refresh` is set.

Features are always computed from information available at the cutoff: the same
leak-free path used for training (Sprint 4).
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, joinedload, sessionmaker

from app.features.builder import FeatureBuilder, FeatureRow
from app.features.config import FeatureConfig
from app.features.store import FeatureStore
from app.features.timeline import Target
from app.ingestion.normalize import name_key
from app.models import (
    Competition,
    DataIngestionRun,
    FeatureSnapshot,
    Match,
    ModelVersion,
    Prediction,
    PredictionOverUnderProbability,
    PredictionResult,
    PredictionScoreProbability,
    Season,
    Team,
)
from app.models.enums import IngestionStatus, MatchStatus
from app.registry.loading import ArtifactUnavailableError, ModelCache
from app.registry.service import ModelRegistry, RegistryError
from app.schemas.predictions import (
    BothTeamsToScore,
    DataFreshness,
    MatchSelector,
    ModelReference,
    OverUnderProbability,
    PredictionRead,
    ScoreProbability,
    StoredPredictionRead,
)

logger = logging.getLogger(__name__)

#: How far ahead an unnamed date search looks for a scheduled meeting.
SEARCH_AHEAD = timedelta(days=400)
TOP_SCORES = 5


class PredictionError(RuntimeError):
    """A request that cannot be served (no match, no model, no features)."""

    def __init__(self, detail: str, *, status_code: int = 404) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class PredictionOutcome:
    prediction: PredictionRead
    created: bool


def floor_to_minute(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


def resolve_match(session: Session, selector: MatchSelector, *, now: datetime) -> Match:
    """The match a selector identifies, or a `PredictionError` explaining why not."""
    if selector.match_id is not None:
        match = session.get(Match, selector.match_id)
        if match is None:
            raise PredictionError(f"match {selector.match_id} was not found")
        return match

    home = _find_team(session, selector.home_team or "")
    away = _find_team(session, selector.away_team or "")
    query = select(Match).where(Match.home_team_id == home.id, Match.away_team_id == away.id)
    query = _narrow(session, query, selector, now=now)
    matches = list(session.scalars(query.order_by(Match.kickoff_at).limit(5)))
    if not matches:
        raise PredictionError(
            f"no match found for {home.name} v {away.name}"
            + (f" on {selector.date}" if selector.date else " (none scheduled)")
        )
    if len(matches) > 1:
        options = ", ".join(f"{m.id} ({m.kickoff_at:%Y-%m-%d})" for m in matches)
        raise PredictionError(
            f"{home.name} v {away.name} matches several fixtures: {options}; "
            "give a date or a match_id",
            status_code=409,
        )
    return matches[0]


def _narrow(
    session: Session, query: Select[tuple[Match]], selector: MatchSelector, *, now: datetime
) -> Select[tuple[Match]]:
    if selector.competition:
        competition = _find_competition(session, selector.competition)
        query = query.where(Match.competition_id == competition.id)
    if selector.date is not None:
        start = datetime.combine(selector.date, datetime.min.time(), tzinfo=UTC)
        return query.where(Match.kickoff_at >= start, Match.kickoff_at < start + timedelta(days=1))
    # No date: the next meeting that has not kicked off yet.
    return query.where(Match.kickoff_at >= now, Match.kickoff_at < now + SEARCH_AHEAD)


def _find_team(session: Session, name: str) -> Team:
    key = name_key(name)
    teams = [t for t in session.scalars(select(Team)) if name_key(t.name) == key]
    if not teams:
        raise PredictionError(f"team {name!r} was not found")
    if len(teams) > 1:
        raise PredictionError(
            f"team {name!r} matches several teams: {', '.join(t.name for t in teams)}",
            status_code=409,
        )
    return teams[0]


def _find_competition(session: Session, name: str) -> Competition:
    key = name_key(name)
    found = [c for c in session.scalars(select(Competition)) if name_key(c.name) == key]
    if not found:
        raise PredictionError(f"competition {name!r} was not found")
    return found[0]


def select_model(session: Session, model_name: str | None) -> ModelVersion:
    """The active version of `model_name`, or of the only active model."""
    registry = ModelRegistry(session)
    if model_name is not None:
        try:
            return registry.get(model_name)
        except RegistryError as exc:
            raise PredictionError(str(exc), status_code=404) from exc
    active = list(session.scalars(select(ModelVersion).where(ModelVersion.is_active)))
    if not active:
        raise PredictionError(
            "no model is active; activate one with `python -m app.ml registry activate`",
            status_code=409,
        )
    if len(active) > 1:
        names = ", ".join(sorted(row.name for row in active))
        raise PredictionError(
            f"several models are active ({names}); name one in the request",
            status_code=409,
        )
    return active[0]


class PredictionService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        cache: ModelCache,
        config: FeatureConfig | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.cache = cache
        self.config = config or FeatureConfig()

    def predict(
        self,
        session: Session,
        matches: list[Match],
        model_version: ModelVersion,
        *,
        refresh: bool = False,
        now: datetime | None = None,
    ) -> list[PredictionOutcome]:
        """One outcome per match, in the order given."""
        now = now or datetime.now(UTC)
        if model_version.feature_set_version != self.config.version():
            raise PredictionError(
                f"model {model_version.name} {model_version.version} needs features "
                f"{model_version.feature_set_version}, but this service builds "
                f"{self.config.version()}",
                status_code=409,
            )
        snapshots = self._snapshots(session, matches, now=now)
        outcomes: list[PredictionOutcome | None] = []
        pending: list[tuple[Match, FeatureSnapshot]] = []
        for match in matches:
            snapshot = snapshots[match.id]
            existing = None if refresh else self._existing(session, match, model_version, snapshot)
            if existing is None:
                outcomes.append(None)  # filled in from `pending` below
                pending.append((match, snapshot))
            else:
                reused = self._read(session, existing, now=now).model_copy(update={"reused": True})
                outcomes.append(PredictionOutcome(reused, created=False))
        created = iter(self._create(session, pending, model_version, now=now))
        return [outcome if outcome is not None else next(created) for outcome in outcomes]

    # Features ---------------------------------------------------------------------

    def _snapshots(
        self, session: Session, matches: list[Match], *, now: datetime
    ) -> dict[int, FeatureSnapshot]:
        targets = [self._target(session, match, now=now) for match in matches]
        wanted = {t.match_id: t.cutoff for t in targets}
        stored = {
            snapshot.match_id: snapshot
            for snapshot in session.scalars(
                select(FeatureSnapshot).where(
                    FeatureSnapshot.match_id.in_(wanted),
                    FeatureSnapshot.feature_set_version == self.config.version(),
                    FeatureSnapshot.data_cutoff_at.in_(set(wanted.values())),
                )
            )
            if snapshot.data_cutoff_at == wanted[snapshot.match_id]
        }
        missing = [t for t in targets if t.match_id not in stored]
        if missing:
            rows = self._build(missing)
            FeatureStore(self.session_factory).save(rows)
            session.expire_all()
            for snapshot in session.scalars(
                select(FeatureSnapshot).where(
                    FeatureSnapshot.match_id.in_([t.match_id for t in missing]),
                    FeatureSnapshot.feature_set_version == self.config.version(),
                    FeatureSnapshot.data_cutoff_at.in_({t.cutoff for t in missing}),
                )
            ):
                if snapshot.data_cutoff_at == wanted[snapshot.match_id]:
                    stored[snapshot.match_id] = snapshot
        missing_still = [m.id for m in matches if m.id not in stored]
        if missing_still:
            raise PredictionError(
                f"features could not be computed for match(es) {missing_still}",
                status_code=422,
            )
        return stored

    def _build(self, targets: list[Target]) -> list[FeatureRow]:
        try:
            return FeatureBuilder(self.session_factory, self.config).build(targets)
        except Exception as exc:  # the builder raises on leakage or impossible targets
            logger.exception("Feature computation failed")
            raise PredictionError(
                f"features could not be computed: {exc}", status_code=422
            ) from exc

    def _target(self, session: Session, match: Match, *, now: datetime) -> Target:
        season = session.get(Season, match.season_id)
        competition = session.get(Competition, match.competition_id)
        if season is None or competition is None:
            raise PredictionError(f"match {match.id} has no competition or season")
        cutoff = min(floor_to_minute(now), match.kickoff_at)
        return Target(
            match_id=match.id,
            competition_id=match.competition_id,
            competition_type=competition.type,
            season_id=match.season_id,
            season_start=season.start_date,
            season_end=season.end_date,
            kickoff_at=match.kickoff_at,
            cutoff=cutoff,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            matchday=match.matchday,
        )

    # Predicting and storing ---------------------------------------------------------

    def _create(
        self,
        session: Session,
        pending: list[tuple[Match, FeatureSnapshot]],
        model_version: ModelVersion,
        *,
        now: datetime,
    ) -> list[PredictionOutcome]:
        if not pending:
            return []
        # Imported lazily: the API image installs the serving extra, not the full ML one.
        from app.ml.dataset import Dataset, Example
        from app.ml.engine import GoalPredictionEngine

        try:
            model, _ = self.cache.get(model_version)
        except ArtifactUnavailableError as exc:
            raise PredictionError(str(exc), status_code=503) from exc
        # A dataset must be in chronological order — the invariant every temporal
        # guarantee rests on — but a caller may ask for matches in any order. Sort
        # for the model, then put the answers back in the order that was asked for.
        ordered = sorted(
            enumerate(pending), key=lambda item: (item[1][0].kickoff_at, item[1][0].id)
        )
        examples = tuple(
            Example(
                match_id=match.id,
                kickoff_at=match.kickoff_at,
                season_id=match.season_id,
                competition_id=match.competition_id,
                features=snapshot.features,
                # Targets are unused when predicting; the result may not exist yet.
                home_goals=match.home_score or 0,
                away_goals=match.away_score or 0,
            )
            for _, (match, snapshot) in ordered
        )
        data = Dataset(examples, model_version.feature_set_version or self.config.version())
        predicted = GoalPredictionEngine(model).predict(data)
        by_request: dict[int, Any] = {
            position: prediction
            for (position, _), prediction in zip(ordered, predicted, strict=True)
        }
        outcomes = []
        for position, (match, snapshot) in enumerate(pending):
            prediction = by_request[position]
            row = self._store(session, match, snapshot, model_version, prediction, now=now)
            logger.info(
                "Predicted match %s with %s %s (cutoff %s)",
                match.id,
                model_version.name,
                model_version.version,
                snapshot.data_cutoff_at.isoformat(),
            )
            outcomes.append(PredictionOutcome(self._read(session, row, now=now), created=True))
        return outcomes

    def _store(
        self,
        session: Session,
        match: Match,
        snapshot: FeatureSnapshot,
        model_version: ModelVersion,
        prediction: Any,
        *,
        now: datetime,
    ) -> Prediction:
        probabilities = [
            round(prediction.home_win_probability, 6),
            round(prediction.draw_probability, 6),
            round(prediction.away_win_probability, 6),
        ]
        probabilities[probabilities.index(max(probabilities))] += 1 - sum(probabilities)
        row = Prediction(
            match_id=match.id,
            model_version_id=model_version.id,
            feature_snapshot_id=snapshot.id,
            predicted_at=now,
            predicted_home_goals=prediction.most_probable_score.home_goals,
            predicted_away_goals=prediction.most_probable_score.away_goals,
            expected_home_goals=_decimal(prediction.expected_home_goals, 3),
            expected_away_goals=_decimal(prediction.expected_away_goals, 3),
            home_win_probability=_decimal(probabilities[0]),
            draw_probability=_decimal(probabilities[1]),
            away_win_probability=_decimal(probabilities[2]),
            btts_probability=_decimal(prediction.btts_probability.yes),
        )
        matrix = prediction.score_probability_matrix
        row.score_probabilities = [
            PredictionScoreProbability(
                home_goals=h, away_goals=a, probability=_decimal(float(matrix[h, a]))
            )
            for h in range(matrix.shape[0])
            for a in range(matrix.shape[1])
        ]
        row.over_under_probabilities = [
            PredictionOverUnderProbability(
                line=Decimal(str(entry.line)),
                over_probability=_decimal(entry.over),
                under_probability=_decimal(entry.under),
            )
            for entry in prediction.over_under_probabilities
        ]
        session.add(row)
        session.flush()
        return row

    def _existing(
        self,
        session: Session,
        match: Match,
        model_version: ModelVersion,
        snapshot: FeatureSnapshot,
    ) -> Prediction | None:
        """A prediction already made from these features by this model version."""
        return session.scalars(
            select(Prediction)
            .where(
                Prediction.match_id == match.id,
                Prediction.model_version_id == model_version.id,
                Prediction.feature_snapshot_id == snapshot.id,
            )
            .order_by(Prediction.predicted_at.desc())
            .limit(1)
        ).one_or_none()

    # Reading ------------------------------------------------------------------------

    def read(self, session: Session, prediction_id: int) -> PredictionRead:
        row = session.get(Prediction, prediction_id)
        if row is None:
            raise PredictionError(f"prediction {prediction_id} was not found")
        return self._read(session, row, now=datetime.now(UTC))

    def _read(self, session: Session, row: Prediction, *, now: datetime) -> PredictionRead:
        match = row.match
        model_version = row.model_version
        cutoff = row.feature_snapshot.data_cutoff_at if row.feature_snapshot else row.predicted_at
        matrix = [[0.0] * (max_goals_of(row) + 1) for _ in range(max_goals_of(row) + 1)]
        for cell in row.score_probabilities:
            matrix[cell.home_goals][cell.away_goals] = float(cell.probability)
        scores = sorted(
            (
                ScoreProbability(
                    home_goals=cell.home_goals,
                    away_goals=cell.away_goals,
                    probability=float(cell.probability),
                )
                for cell in row.score_probabilities
            ),
            key=lambda s: -s.probability,
        )
        return PredictionRead(
            id=row.id,
            match_id=match.id,
            kickoff_at=match.kickoff_at,
            competition=match.competition.name if match.competition else None,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            expected_home_goals=float(row.expected_home_goals),
            expected_away_goals=float(row.expected_away_goals),
            most_probable_score=ScoreProbability(
                home_goals=row.predicted_home_goals,
                away_goals=row.predicted_away_goals,
                probability=matrix[row.predicted_home_goals][row.predicted_away_goals],
            ),
            top_scores=scores[:TOP_SCORES],
            score_probability_matrix=matrix,
            max_goals=max_goals_of(row),
            home_win_probability=float(row.home_win_probability),
            draw_probability=float(row.draw_probability),
            away_win_probability=float(row.away_win_probability),
            over_under_probabilities=[
                OverUnderProbability(
                    line=float(line.line),
                    over=float(line.over_probability),
                    under=float(line.under_probability),
                )
                for line in row.over_under_probabilities
            ],
            btts_probability=BothTeamsToScore(
                yes=float(row.btts_probability), no=1 - float(row.btts_probability)
            ),
            model=ModelReference(
                id=model_version.id,
                name=model_version.name,
                version=model_version.version,
                algorithm=model_version.algorithm,
                feature_set_version=model_version.feature_set_version,
            ),
            predicted_at=row.predicted_at,
            data_freshness=freshness(session, cutoff=cutoff, match=match, now=now),
            reused=False,
        )


def max_goals_of(row: Prediction) -> int:
    return max((cell.home_goals for cell in row.score_probabilities), default=0)


def freshness(session: Session, *, cutoff: datetime, match: Match, now: datetime) -> DataFreshness:
    latest_result = session.scalar(
        select(func.max(Match.kickoff_at)).where(
            Match.status == MatchStatus.FINISHED,
            Match.kickoff_at + FeatureConfig().result_delay <= cutoff,
        )
    )
    last_ingestion = session.scalar(
        select(func.max(DataIngestionRun.finished_at)).where(
            DataIngestionRun.status == IngestionStatus.SUCCEEDED
        )
    )
    return DataFreshness(
        feature_cutoff_at=cutoff,
        latest_result_at=latest_result,
        last_ingestion_at=last_ingestion,
        age_seconds=max((now - cutoff).total_seconds(), 0.0),
        match_started=match.kickoff_at <= now,
    )


def _decimal(value: float, places: int = 6) -> Decimal:
    return Decimal(f"{value:.{places}f}")


def list_stored(
    session: Session,
    *,
    match_id: int | None = None,
    model_name: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[int, list[StoredPredictionRead]]:
    """Stored predictions, newest first, with their result when there is one."""
    query = (
        select(Prediction, Match, ModelVersion, PredictionResult)
        .join(Match, Match.id == Prediction.match_id)
        .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
        .outerjoin(PredictionResult, PredictionResult.prediction_id == Prediction.id)
        # Each row names its competition and both teams.
        .options(
            joinedload(Match.competition),
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
    )
    if match_id is not None:
        query = query.where(Prediction.match_id == match_id)
    if model_name:
        query = query.where(ModelVersion.name == model_name)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = session.execute(
        query.order_by(Prediction.predicted_at.desc(), Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return total, [
        StoredPredictionRead(
            id=prediction.id,
            match_id=match.id,
            kickoff_at=match.kickoff_at,
            competition=match.competition.name if match.competition else None,
            home_team=match.home_team.name,
            away_team=match.away_team.name,
            home_win_probability=float(prediction.home_win_probability),
            draw_probability=float(prediction.draw_probability),
            away_win_probability=float(prediction.away_win_probability),
            most_probable_score=(
                f"{prediction.predicted_home_goals}-{prediction.predicted_away_goals}"
            ),
            expected_home_goals=float(prediction.expected_home_goals),
            expected_away_goals=float(prediction.expected_away_goals),
            model_name=version.name,
            model_version=version.version,
            predicted_at=prediction.predicted_at,
            actual_score=(
                f"{match.home_score}-{match.away_score}"
                if match.home_score is not None and match.away_score is not None
                else None
            ),
            outcome_correct=result.outcome_correct if result is not None else None,
        )
        for prediction, match, version, result in rows
    ]
