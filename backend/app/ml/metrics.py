"""Evaluation metrics for match predictions.

1X2: accuracy, log loss, Brier score (multi-class: sum over the three outcomes, range
0 to 2) and ranked probability score (RPS, which respects home < draw < away order).
Goals: MAE and RMSE of expected goals against actual goals (home, away, total).
Exact scores: accuracy of the most likely score, top-3 and top-5 coverage, and the
mean log probability given to the actual score.
Derived markets (from the same score grid): accuracy, Brier score (single-probability
form, range 0 to 1), log loss and calibration error for over 1.5 / 2.5 / 3.5 total
goals and for both teams to score.
Calibration: expected calibration error (ECE, 10 equal-width bins); for 1X2 the mean
of the home, draw and away ECE (`calibration.classwise_ece`).
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.ml.calibration import classwise_ece, expected_calibration_error
from app.ml.dataset import Dataset, FloatArray
from app.ml.probability import MatchPrediction

#: Over/under lines scored by `evaluate` (0.5 is omitted: 0-0 is its only "under").
EVALUATED_LINES = (1.5, 2.5, 3.5)
#: Fewest matches a competition needs before its own figures are reported.
MIN_COMPETITION_MATCHES = 30

#: Probability floor used for log scores, so one impossible outcome cannot give -inf.
EPS = 1e-15


def metrics_by_competition(
    predictions: Sequence[MatchPrediction], data: Dataset
) -> dict[str, dict[str, Any]]:
    """The figures that show how far to trust a model in each competition, keyed by its id.

    Competitions with fewer than `MIN_COMPETITION_MATCHES` matches are left out: a rate
    from so few says little, and showing it would suggest otherwise.
    """
    if len(predictions) != len(data):
        raise ValueError("one prediction per match is required")
    measured: dict[str, dict[str, Any]] = {}
    for competition_id in data.competition_ids:
        chosen = [
            index
            for index, example in enumerate(data.examples)
            if example.competition_id == competition_id
        ]
        if len(chosen) < MIN_COMPETITION_MATCHES:
            continue
        subset = Dataset(tuple(data.examples[i] for i in chosen), data.feature_set_version)
        metrics = evaluate([predictions[i] for i in chosen], subset)
        described = subset.describe()
        measured[str(competition_id)] = {
            "matches": len(chosen),
            "accuracy": metrics["accuracy"],
            "log_loss": metrics["log_loss"],
            "exact_score_accuracy": metrics["exact_score_accuracy"],
            "home_win_rate": described["home_win_rate"],
            "start": described["start"],
            "end": described["end"],
        }
    return measured


def evaluate(predictions: Sequence[MatchPrediction], data: Dataset) -> dict[str, float]:
    if len(predictions) != len(data):
        raise ValueError("one prediction per match is required")
    if not predictions:
        raise ValueError("nothing to evaluate")
    probs = np.array([p.outcomes for p in predictions])
    outcomes = data.outcomes
    one_hot = np.eye(3)[outcomes]
    n = len(predictions)

    expected_home = np.array([p.expected_home_goals for p in predictions])
    expected_away = np.array([p.expected_away_goals for p in predictions])
    home, away = data.home_goals, data.away_goals

    actual_score_probs = np.array(
        [
            p.score_probability(int(h), int(a))
            for p, h, a in zip(predictions, home, away, strict=True)
        ]
    )
    top = [p.top_scores(5) for p in predictions]
    hits = [
        [(sh, sa) == (int(h), int(a)) for sh, sa, _ in scores]
        for scores, h, a in zip(top, home, away, strict=True)
    ]
    cumulative = np.cumsum(probs, axis=1)[:, :2]
    cumulative_actual = np.cumsum(one_hot, axis=1)[:, :2]

    def rmse(error: FloatArray) -> float:
        return float(math.sqrt(np.mean(error**2)))

    markets: dict[str, float] = {}
    total = home + away
    for line in EVALUATED_LINES:
        name = f"over_{str(line).replace('.', '_')}"
        markets |= _binary(
            name, np.array([p.over_under(line)[0] for p in predictions]), total > line
        )
    markets |= _binary("btts", np.array([p.btts[0] for p in predictions]), (home > 0) & (away > 0))

    return {
        "matches": float(n),
        "accuracy": float(np.mean(probs.argmax(axis=1) == outcomes)),
        "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(n), outcomes], EPS, 1)))),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece": classwise_ece(probs, outcomes),
        "rps": float(np.mean(np.sum((cumulative - cumulative_actual) ** 2, axis=1) / 2)),
        "home_goals_mae": float(np.mean(np.abs(expected_home - home))),
        "away_goals_mae": float(np.mean(np.abs(expected_away - away))),
        "total_goals_mae": float(np.mean(np.abs(expected_home + expected_away - home - away))),
        "home_goals_rmse": rmse(expected_home - home),
        "away_goals_rmse": rmse(expected_away - away),
        "total_goals_rmse": rmse(expected_home + expected_away - home - away),
        "exact_score_accuracy": float(np.mean([row[0] for row in hits])),
        "top3_score_coverage": float(np.mean([any(row[:3]) for row in hits])),
        "top5_score_coverage": float(np.mean([any(row[:5]) for row in hits])),
        "actual_score_log_prob": float(np.mean(np.log(np.clip(actual_score_probs, EPS, 1)))),
        **markets,
    }


def _binary(name: str, probability: FloatArray, happened: np.ndarray) -> dict[str, float]:
    """Accuracy (yes when p > 0.5), Brier score, log loss and calibration error of a
    yes/no market."""
    outcome = happened.astype(float)
    p_actual = np.where(happened, probability, 1 - probability)
    return {
        f"{name}_accuracy": float(np.mean((probability > 0.5) == happened)),
        f"{name}_brier": float(np.mean((probability - outcome) ** 2)),
        f"{name}_log_loss": float(-np.mean(np.log(np.clip(p_actual, EPS, 1)))),
        f"{name}_ece": expected_calibration_error(probability, happened),
    }


def match_log_losses(predictions: Sequence[MatchPrediction], data: Dataset) -> FloatArray:
    """1X2 log loss of each match (their mean is `log_loss`), for paired comparisons."""
    if len(predictions) != len(data):
        raise ValueError("one prediction per match is required")
    probs = np.array([p.outcomes for p in predictions], dtype=float).reshape(-1, 3)
    losses: FloatArray = -np.log(np.clip(probs[np.arange(len(data)), data.outcomes], EPS, 1))
    return losses


BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 0


def paired_bootstrap(
    losses: Sequence[float],
    reference: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Mean per-match difference `losses - reference` (negative = better than the
    reference), its 95% percentile interval over matches resampled with replacement,
    and the share of resamples in which the difference is negative."""
    diff = np.asarray(losses, dtype=float) - np.asarray(reference, dtype=float)
    if diff.size == 0:
        raise ValueError("no matches to compare")
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(samples, diff.size))].mean(axis=1)
    return {
        "difference": float(diff.mean()),
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "better_share": float((means < 0).mean()),
    }


#: The metric used for model selection (on validation data only). Lower is better.
SELECTION_METRIC = "log_loss"
