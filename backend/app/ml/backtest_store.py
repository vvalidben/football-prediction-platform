"""Writes backtest predictions to the database, in real-world order.

Each (model, calibration variant) of a backtest gets its own inactive
`model_versions` row, named `backtest_<model>_<variant>`. As the backtest moves
forward:

- `save` writes each prediction when it is made (`predicted_at` = the step time,
  always before kickoff), with its full exact-score grid and over/under lines;
- `reveal(t)` writes a `prediction_results` row for every stored prediction whose
  result is known by t (`evaluated_at` = kickoff + result delay), reading the
  actual score from the `matches` table.

`metrics_from_store` recomputes the metrics from the stored rows alone, so the
reported figures can be checked against what was persisted.
"""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker

from app.ml.backtest import BacktestConfig, BacktestRecord, BacktestResult, ModelSpec
from app.ml.dataset import AWAY, DRAW, HOME, Dataset, Example
from app.ml.metrics import evaluate
from app.ml.probability import (
    DEFAULT_LINES,
    MAX_GOALS,
    MatchPrediction,
    normalise,
    reconcile,
)
from app.models import (
    Competition,
    FeatureSnapshot,
    Match,
    ModelVersion,
    Prediction,
    PredictionOverUnderProbability,
    PredictionResult,
    PredictionScoreProbability,
)
from app.registry.service import ModelRegistry, NewModelVersion

EPS = 1e-15


def _competitions(session: Session, ids: Sequence[int]) -> list[dict[str, Any]]:
    names = {
        i: n
        for i, n in session.execute(
            select(Competition.id, Competition.name).where(Competition.id.in_(ids))
        )
    }
    return [{"id": i, "name": names.get(i)} for i in ids]


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]


def _decimal(value: float, places: int = 6) -> Decimal:
    return Decimal(f"{value:.{places}f}")


class BacktestStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        models: Sequence[ModelSpec],
        config: BacktestConfig,
        data: Dataset,
        algorithms: Mapping[str, str],
        run: str | None = None,
    ) -> None:
        """`data` is the backtest's dataset; `algorithms` maps each model label to its
        implementation class name."""
        feature_set_version = data.feature_set_version
        self.factory = session_factory
        self.config = config
        self.feature_set_version = feature_set_version
        self.run = run or f"{datetime.now(UTC):%Y%m%d%H%M%S}"
        #: (model label, variant) -> model_versions.id
        self.versions: dict[tuple[str, str], int] = {}
        #: prediction id -> (match id, revealed_at), not yet evaluated
        self._pending: dict[int, tuple[int, datetime]] = {}
        with self.factory() as session, session.begin():
            self._snapshots: dict[int, int] = {
                match_id: snapshot_id
                for match_id, snapshot_id in session.execute(
                    select(FeatureSnapshot.match_id, FeatureSnapshot.id)
                    .join(Match, Match.id == FeatureSnapshot.match_id)
                    .where(
                        FeatureSnapshot.feature_set_version == feature_set_version,
                        FeatureSnapshot.data_cutoff_at == Match.kickoff_at,
                    )
                )
            }
            period = data.where(lambda e: config.end is None or e.kickoff_at < config.end)
            registry = ModelRegistry(session)
            for spec in models:
                for variant in config.calibration:
                    row = registry.register(
                        NewModelVersion(
                            name=f"backtest_{spec.name}_{variant}",
                            version=f"{self.run}-{_digest(spec.label, variant)}",
                            algorithm=algorithms[spec.label],
                            description=(
                                f"backtest of {spec.label}, calibration {variant}, "
                                f"{config.window} window from {config.start.date()} "
                                "(evaluation record: no artifact, cannot be activated)"
                            ),
                            feature_set_version=feature_set_version,
                            dataset_version=period.fingerprint(),
                            # Updated by finish() to the span actually trained on.
                            training_data_start=period.examples[0].kickoff_at.date(),
                            training_data_end=period.examples[-1].kickoff_at.date(),
                            competitions=_competitions(session, period.competition_ids),
                            hyperparameters={
                                "model": spec.name,
                                "params": spec.params,
                                "calibration": variant,
                                "backtest": config.describe(),
                            },
                            metrics={},
                            artifact_uri=None,
                            trained_at=datetime.now(UTC),
                        )
                    )
                    self.versions[(spec.label, variant)] = row.id

    def save(self, records: Sequence[BacktestRecord]) -> None:
        if not records:
            return
        with self.factory() as session, session.begin():
            ids = session.scalars(
                insert(Prediction).returning(Prediction.id, sort_by_parameter_order=True),
                [self._prediction_row(r) for r in records],
            ).all()
            scores: list[dict[str, Any]] = []
            lines: list[dict[str, Any]] = []
            for prediction_id, record in zip(ids, records, strict=True):
                grid = record.prediction.grid
                scores += [
                    {
                        "prediction_id": prediction_id,
                        "home_goals": h,
                        "away_goals": a,
                        "probability": _decimal(float(grid[h, a])),
                    }
                    for h in range(MAX_GOALS + 1)
                    for a in range(MAX_GOALS + 1)
                ]
                for line in DEFAULT_LINES:
                    over, under = record.prediction.over_under(line)
                    lines.append(
                        {
                            "prediction_id": prediction_id,
                            "line": Decimal(str(line)),
                            "over_probability": _decimal(over),
                            "under_probability": _decimal(under),
                        }
                    )
                self._pending[prediction_id] = (record.example.match_id, record.revealed_at)
            session.execute(insert(PredictionScoreProbability), scores)
            session.execute(insert(PredictionOverUnderProbability), lines)

    def _prediction_row(self, record: BacktestRecord) -> dict[str, Any]:
        p = record.prediction
        home, away, _ = p.most_probable_score
        outcomes = [round(x, 6) for x in p.outcomes]
        # Rounding to 6 places can leave the three a few millionths from 1: put the
        # remainder on the largest, well inside the database tolerance.
        outcomes[int(np.argmax(outcomes))] += 1 - sum(outcomes)
        return {
            "match_id": record.example.match_id,
            "model_version_id": self.versions[(record.model, record.variant)],
            "feature_snapshot_id": self._snapshots.get(record.example.match_id),
            "predicted_at": record.predicted_at,
            "predicted_home_goals": home,
            "predicted_away_goals": away,
            "expected_home_goals": _decimal(p.expected_home_goals, 3),
            "expected_away_goals": _decimal(p.expected_away_goals, 3),
            "home_win_probability": _decimal(outcomes[0]),
            "draw_probability": _decimal(outcomes[1]),
            "away_win_probability": _decimal(outcomes[2]),
            "btts_probability": _decimal(p.btts[0]),
        }

    def reveal(self, up_to: datetime) -> None:
        """Evaluate every stored prediction whose result is known by `up_to`."""
        due = [pid for pid, (_, revealed_at) in self._pending.items() if revealed_at <= up_to]
        if not due:
            return
        with self.factory() as session, session.begin():
            predictions = {
                p.id: p for p in session.scalars(select(Prediction).where(Prediction.id.in_(due)))
            }
            matches = {
                m.id: m
                for m in session.scalars(
                    select(Match).where(Match.id.in_({p.match_id for p in predictions.values()}))
                )
            }
            grids = _load_grids(session, due)
            rows = []
            for pid in due:
                prediction = predictions[pid]
                match = matches[prediction.match_id]
                rows.append(_result_row(pid, prediction, match, grids[pid], self._pending[pid][1]))
            session.execute(insert(PredictionResult), rows)
        for pid in due:
            del self._pending[pid]

    def finish(self, result: BacktestResult) -> None:
        """Record the overall metrics on each backtest model version."""
        with self.factory() as session, session.begin():
            for (label, variant), version_id in self.versions.items():
                row = session.get(ModelVersion, version_id)
                assert row is not None  # noqa: S101 - created in __init__
                chosen = [r for r in result.records if r.model == label and r.variant == variant]
                row.training_metrics = {
                    "backtest": result.metrics(label, variant),
                    "predictions": len(chosen),
                    "calibrated_predictions": sum(1 for r in chosen if r.calibrated_on),
                    "skipped_matches": result.skipped_matches,
                }
                if chosen:
                    row.training_data_start = min(r.train_start for r in chosen).date()
                    row.training_data_end = max(r.train_end for r in chosen).date()


def _load_grids(session: Session, prediction_ids: Sequence[int]) -> dict[int, np.ndarray]:
    grids = {pid: np.zeros((MAX_GOALS + 1, MAX_GOALS + 1)) for pid in prediction_ids}
    for pid, h, a, probability in session.execute(
        select(
            PredictionScoreProbability.prediction_id,
            PredictionScoreProbability.home_goals,
            PredictionScoreProbability.away_goals,
            PredictionScoreProbability.probability,
        ).where(PredictionScoreProbability.prediction_id.in_(prediction_ids))
    ):
        grids[pid][h, a] = float(probability)
    return {pid: normalise(grid) for pid, grid in grids.items()}


def _result_row(
    prediction_id: int,
    prediction: Prediction,
    match: Match,
    grid: np.ndarray,
    revealed_at: datetime,
) -> dict[str, Any]:
    if match.home_score is None or match.away_score is None:
        raise ValueError(f"match {match.id} has no result to reveal")
    hg, ag = match.home_score, match.away_score
    probs = np.array(
        [
            float(prediction.home_win_probability),
            float(prediction.draw_probability),
            float(prediction.away_win_probability),
        ]
    )
    outcome = HOME if hg > ag else DRAW if hg == ag else AWAY
    one_hot = np.eye(3)[outcome]
    cumulative, actual = np.cumsum(probs)[:2], np.cumsum(one_hot)[:2]
    score_p = float(grid[hg, ag]) if hg <= MAX_GOALS and ag <= MAX_GOALS else 0.0
    total = hg + ag
    per_line = {
        f"over_{float(p.line)}_correct": (float(p.over_probability) > 0.5)
        == (total > float(p.line))
        for p in prediction.over_under_probabilities
    }
    return {
        "prediction_id": prediction_id,
        "evaluated_at": revealed_at,
        "outcome_correct": bool(int(np.argmax(probs)) == outcome),
        "exact_score_correct": (prediction.predicted_home_goals, prediction.predicted_away_goals)
        == (hg, ag),
        "btts_correct": (float(prediction.btts_probability) > 0.5) == (hg > 0 and ag > 0),
        "brier_score": _decimal(float(np.sum((probs - one_hot) ** 2))),
        "log_loss": _decimal(float(-np.log(max(probs[outcome], EPS)))),
        "ranked_probability_score": _decimal(float(np.sum((cumulative - actual) ** 2) / 2)),
        "actual_score_probability": _decimal(min(max(score_p, 0.0), 1.0)),
        "metrics": {
            **per_line,
            "home_goals_error": float(prediction.expected_home_goals) - hg,
            "away_goals_error": float(prediction.expected_away_goals) - ag,
        },
    }


def metrics_from_store(session: Session, model_version_id: int) -> dict[str, float]:
    """Recompute the backtest metrics from the stored predictions and match results."""
    rows = session.execute(
        select(Prediction, Match)
        .join(Match, Match.id == Prediction.match_id)
        .join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
        .where(Prediction.model_version_id == model_version_id)
        .order_by(Match.kickoff_at, Match.id)
    ).all()
    if not rows:
        return {}
    grids = _load_grids(session, [p.id for p, _ in rows])
    if any(m.home_score is None or m.away_score is None for _, m in rows):
        raise ValueError("a stored prediction's match has no result")
    examples = tuple(
        Example(
            match_id=m.id,
            kickoff_at=m.kickoff_at,
            season_id=m.season_id,
            competition_id=m.competition_id,
            features={},
            home_goals=m.home_score,
            away_goals=m.away_score,
        )
        for _, m in rows
    )
    # The stored 1X2 columns are the prediction's own probabilities (rounding keeps
    # exact ties, which re-summing 121 rounded cells would not): rescale the stored
    # grid to them, as the models do.
    predictions = [
        MatchPrediction(
            reconcile(
                grids[p.id],
                (
                    float(p.home_win_probability),
                    float(p.draw_probability),
                    float(p.away_win_probability),
                ),
            )
        )
        for p, _ in rows
    ]
    return evaluate(predictions, Dataset(examples, "stored"))
