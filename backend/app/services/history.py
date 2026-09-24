"""Prediction history: every stored prediction, with its result once there is one.

Nothing new is computed about a match here. A row joins three things that were
written when they happened — the prediction (Sprint 11), the match result (the
ingestion jobs) and the evaluation (`prediction_results`, Sprint 9/12) — so the
history is a record, not a recalculation. The only derived figure is the
probability the model gave to the outcome that happened, which is picked out of
the prediction's own three probabilities.

The summary aggregates every row matching the filters, not only the page returned,
so a filtered view reports its own accuracy rather than the page's.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Float, Select, case, cast, func, or_, select
from sqlalchemy.orm import Session, aliased

from app.models import Competition, Match, ModelVersion, Prediction, PredictionResult, Season, Team
from app.schemas.history import HistorySummary, PredictionHistoryRow

logger = logging.getLogger(__name__)

OUTCOMES = ("home", "draw", "away")

_home_team = aliased(Team, name="home_team")
_away_team = aliased(Team, name="away_team")


@dataclass(frozen=True)
class HistoryFilters:
    """What to show. Every field is optional; together they narrow the same query."""

    competition_id: int | None = None
    season_id: int | None = None
    team_id: int | None = None
    model_name: str | None = None
    model_version_id: int | None = None
    from_date: date | None = None
    to_date: date | None = None
    #: True: only evaluated predictions. False: only those still waiting.
    evaluated: bool | None = None
    #: True: only the ones whose most likely outcome happened; False: only the misses.
    outcome_correct: bool | None = None


def _base() -> Select[Any]:
    return (
        select(
            Prediction,
            Match,
            ModelVersion,
            PredictionResult,
            Competition,
            Season,
            _home_team,
            _away_team,
        )
        .join(Match, Match.id == Prediction.match_id)
        .join(ModelVersion, ModelVersion.id == Prediction.model_version_id)
        .outerjoin(PredictionResult, PredictionResult.prediction_id == Prediction.id)
        .outerjoin(Competition, Competition.id == Match.competition_id)
        .outerjoin(Season, Season.id == Match.season_id)
        .join(_home_team, _home_team.id == Match.home_team_id)
        .join(_away_team, _away_team.id == Match.away_team_id)
    )


def _filtered(query: Select[Any], filters: HistoryFilters) -> Select[Any]:
    if filters.competition_id is not None:
        query = query.where(Match.competition_id == filters.competition_id)
    if filters.season_id is not None:
        query = query.where(Match.season_id == filters.season_id)
    if filters.team_id is not None:
        query = query.where(
            or_(Match.home_team_id == filters.team_id, Match.away_team_id == filters.team_id)
        )
    if filters.model_name:
        query = query.where(ModelVersion.name == filters.model_name)
    if filters.model_version_id is not None:
        query = query.where(Prediction.model_version_id == filters.model_version_id)
    if filters.from_date is not None:
        query = query.where(Match.kickoff_at >= _start_of(filters.from_date))
    if filters.to_date is not None:
        query = query.where(Match.kickoff_at < _start_of(filters.to_date) + timedelta(days=1))
    if filters.evaluated is True:
        query = query.where(PredictionResult.prediction_id.is_not(None))
    if filters.evaluated is False:
        query = query.where(PredictionResult.prediction_id.is_(None))
    if filters.outcome_correct is not None:
        query = query.where(PredictionResult.outcome_correct.is_(filters.outcome_correct))
    return query


def _share(column: Any) -> Any:
    """A boolean as 1.0 or 0.0 so it can be averaged.

    PostgreSQL refuses to cast boolean to double precision, and an unevaluated row
    is NULL here rather than 0, so it is left out of the average instead of counting
    as a miss.
    """
    return case((column.is_(True), 1.0), (column.is_(False), 0.0), else_=None)


def _start_of(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def history(
    session: Session, filters: HistoryFilters, *, limit: int = 25, offset: int = 0
) -> tuple[int, list[PredictionHistoryRow]]:
    """Predictions matching the filters, most recent kickoff first."""
    query = _filtered(_base(), filters)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = session.execute(
        query.order_by(Match.kickoff_at.desc(), Prediction.id.desc()).limit(limit).offset(offset)
    ).all()
    return total, [_row(*row) for row in rows]


def _row(
    prediction: Prediction,
    match: Match,
    version: ModelVersion,
    result: PredictionResult | None,
    competition: Competition | None,
    season: Season | None,
    home_team: Team,
    away_team: Team,
) -> PredictionHistoryRow:
    probabilities = (
        float(prediction.home_win_probability),
        float(prediction.draw_probability),
        float(prediction.away_win_probability),
    )
    actual = _outcome_of(match)
    metrics = result.metrics if result is not None else {}
    return PredictionHistoryRow(
        prediction_id=prediction.id,
        match_id=match.id,
        kickoff_at=match.kickoff_at,
        competition_id=match.competition_id,
        competition=competition.name if competition else None,
        season_id=match.season_id,
        season=season.name if season else None,
        home_team_id=match.home_team_id,
        away_team_id=match.away_team_id,
        home_team=home_team.name,
        away_team=away_team.name,
        predicted_home_goals=prediction.predicted_home_goals,
        predicted_away_goals=prediction.predicted_away_goals,
        expected_home_goals=float(prediction.expected_home_goals),
        expected_away_goals=float(prediction.expected_away_goals),
        home_win_probability=probabilities[0],
        draw_probability=probabilities[1],
        away_win_probability=probabilities[2],
        btts_probability=float(prediction.btts_probability),
        predicted_outcome=OUTCOMES[probabilities.index(max(probabilities))],
        status=match.status.value,
        home_score=match.home_score,
        away_score=match.away_score,
        actual_outcome=actual,
        evaluated_at=result.evaluated_at if result else None,
        outcome_correct=result.outcome_correct if result else None,
        exact_score_correct=result.exact_score_correct if result else None,
        btts_correct=result.btts_correct if result else None,
        brier_score=float(result.brier_score) if result else None,
        log_loss=float(result.log_loss) if result else None,
        ranked_probability_score=float(result.ranked_probability_score) if result else None,
        actual_score_probability=(
            float(result.actual_score_probability)
            if result and result.actual_score_probability is not None
            else None
        ),
        home_goals_error=_number(metrics.get("home_goals_error")),
        away_goals_error=_number(metrics.get("away_goals_error")),
        actual_outcome_probability=(
            probabilities[OUTCOMES.index(actual)] if actual is not None else None
        ),
        model_id=version.id,
        model_name=version.name,
        model_version=version.version,
        predicted_at=prediction.predicted_at,
    )


def _outcome_of(match: Match) -> str | None:
    if match.home_score is None or match.away_score is None:
        return None
    if match.home_score > match.away_score:
        return "home"
    return "draw" if match.home_score == match.away_score else "away"


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def summary(session: Session, filters: HistoryFilters) -> HistorySummary:
    """Totals over everything matching the filters.

    The averages cover evaluated rows only — an unplayed match has no error — while
    `predictions` counts every stored row, so the two numbers can be compared.
    """
    evaluated = PredictionResult.prediction_id.is_not(None)
    home_error = cast(Prediction.expected_home_goals, Float) - Match.home_score
    away_error = cast(Prediction.expected_away_goals, Float) - Match.away_score
    query = _filtered(
        _base().with_only_columns(
            func.count(),
            func.count(PredictionResult.prediction_id),
            func.avg(_share(PredictionResult.outcome_correct)),
            func.avg(_share(PredictionResult.exact_score_correct)),
            func.avg(_share(PredictionResult.btts_correct)),
            func.avg(PredictionResult.log_loss),
            func.avg(PredictionResult.brier_score),
            func.avg(PredictionResult.ranked_probability_score),
            func.avg(home_error).filter(evaluated),
            func.avg(away_error).filter(evaluated),
            func.avg((func.abs(home_error) + func.abs(away_error)) / 2).filter(evaluated),
            func.min(Match.kickoff_at),
            func.max(Match.kickoff_at),
        ),
        filters,
    )
    row = session.execute(query).one()
    total, scored = int(row[0] or 0), int(row[1] or 0)
    return HistorySummary(
        predictions=total,
        evaluated=scored,
        pending=total - scored,
        accuracy=_number(row[2]),
        exact_score_accuracy=_number(row[3]),
        btts_accuracy=_number(row[4]),
        log_loss=_number(row[5]),
        brier_score=_number(row[6]),
        ranked_probability_score=_number(row[7]),
        mean_home_goals_error=_number(row[8]),
        mean_away_goals_error=_number(row[9]),
        mean_absolute_goals_error=_number(row[10]),
        first_kickoff_at=row[11],
        last_kickoff_at=row[12],
    )
