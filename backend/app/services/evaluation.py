"""Scoring stored predictions once their match has been played.

A prediction is evaluated when its match is finished and the result would be known
(kickoff + the feature system's result delay). The scores come from the stored
probabilities and score grid, so a `prediction_results` row always describes exactly
what was served.
"""

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.features.config import FeatureConfig
from app.models import Match, Prediction, PredictionResult
from app.models.enums import MatchStatus

logger = logging.getLogger(__name__)
EPS = 1e-15


@dataclass(frozen=True)
class Evaluation:
    evaluated: int
    match_ids: list[int]


def pending(session: Session, *, now: datetime, limit: int) -> list[tuple[Prediction, Match]]:
    """Stored predictions whose match has finished and whose result is known."""
    delay = FeatureConfig().result_delay
    rows = session.execute(
        select(Prediction, Match)
        .join(Match, Match.id == Prediction.match_id)
        .outerjoin(PredictionResult, PredictionResult.prediction_id == Prediction.id)
        .where(
            PredictionResult.prediction_id.is_(None),
            Match.status == MatchStatus.FINISHED,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
            Match.kickoff_at + delay <= now,
        )
        .order_by(Match.kickoff_at, Prediction.id)
        .limit(limit)
    ).all()
    return [(prediction, match) for prediction, match in rows]


def evaluate_pending(
    session: Session, *, now: datetime | None = None, limit: int = 500
) -> Evaluation:
    now = now or datetime.now(UTC)
    evaluated = []
    for prediction, match in pending(session, now=now, limit=limit):
        session.add(PredictionResult(**result_values(prediction, match, now=now)))
        evaluated.append(match.id)
    session.flush()
    return Evaluation(evaluated=len(evaluated), match_ids=evaluated)


def result_values(prediction: Prediction, match: Match, *, now: datetime) -> dict[str, Any]:
    """Accuracy, Brier score, log loss, RPS and the probability of the actual score."""
    if match.home_score is None or match.away_score is None:
        raise ValueError(f"match {match.id} has no result")
    home, away = match.home_score, match.away_score
    probabilities = [
        float(prediction.home_win_probability),
        float(prediction.draw_probability),
        float(prediction.away_win_probability),
    ]
    outcome = 0 if home > away else 1 if home == away else 2
    actual = [1.0 if index == outcome else 0.0 for index in range(3)]
    brier = sum((p - a) ** 2 for p, a in zip(probabilities, actual, strict=True))
    cumulative = [sum(probabilities[: i + 1]) for i in range(2)]
    cumulative_actual = [sum(actual[: i + 1]) for i in range(2)]
    rps = sum((c - a) ** 2 for c, a in zip(cumulative, cumulative_actual, strict=True)) / 2
    score_probability = next(
        (
            float(cell.probability)
            for cell in prediction.score_probabilities
            if (cell.home_goals, cell.away_goals) == (home, away)
        ),
        0.0,
    )
    total = home + away
    per_line = {
        f"over_{float(line.line)}_correct": (float(line.over_probability) > 0.5)
        == (total > float(line.line))
        for line in prediction.over_under_probabilities
    }
    return {
        "prediction_id": prediction.id,
        "evaluated_at": now,
        "outcome_correct": probabilities.index(max(probabilities)) == outcome,
        "exact_score_correct": (prediction.predicted_home_goals, prediction.predicted_away_goals)
        == (home, away),
        "btts_correct": (float(prediction.btts_probability) > 0.5) == (home > 0 and away > 0),
        "brier_score": _decimal(brier),
        "log_loss": _decimal(-math.log(max(probabilities[outcome], EPS))),
        "ranked_probability_score": _decimal(rps),
        "actual_score_probability": _decimal(min(max(score_probability, 0.0), 1.0)),
        "metrics": {
            **per_line,
            "home_goals_error": float(prediction.expected_home_goals) - home,
            "away_goals_error": float(prediction.expected_away_goals) - away,
        },
    }


def accuracy_since(session: Session, *, since: datetime) -> dict[str, float]:
    """Headline accuracy of everything evaluated since a moment (for reports)."""
    row = session.execute(
        select(
            func.count(),
            # PostgreSQL will not cast boolean to double precision.
            func.avg(case((PredictionResult.outcome_correct, 1.0), else_=0.0)),
            func.avg(PredictionResult.log_loss),
        ).where(PredictionResult.evaluated_at >= since)
    ).one()
    count = int(row[0] or 0)
    return {
        "evaluated": count,
        "accuracy": float(row[1]) if row[1] is not None else 0.0,
        "log_loss": float(row[2]) if row[2] is not None else 0.0,
    }


def _decimal(value: float, places: int = 6) -> Decimal:
    return Decimal(f"{value:.{places}f}")
