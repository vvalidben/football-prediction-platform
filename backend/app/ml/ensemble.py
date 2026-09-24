"""Ensembles of the prediction models, with weights chosen on validation data only.

Every model outputs a full score grid, so ensembles combine grids. Over/under, BTTS
and the rest stay derived from one grid and remain consistent.

- `SimpleAverage`: the mean of the members' grids. Nothing is fitted.
- `WeightedAverage`: a weighted mean of the grids (weights >= 0, summing to 1). The
  weights minimise the 1X2 log loss of the mixture; that loss is convex in the
  weights, so the optimum is well defined. A weight can be exactly 0: a member that
  does not help is dropped.
- `Stacking`: a multinomial logistic regression on the members' 1X2 log-odds gives
  the ensemble's 1X2. The simple-average grid is rescaled to it.

**Validation data only.** Combiners are fitted on out-of-fold member predictions:
every match of the rolling-origin folds (`experiment.earlier_folds` plus the
validation period) predicted by a member fitted only on earlier matches, with its
hyperparameters as tuned in `tune_and_select`. The test period is never used.

**Measured, not assumed.** `forward_evaluation` scores every individual model and
every ensemble method on the same held-out folds: fold k is predicted with weights
fitted on folds 1..k-1 only. The first fold only provides training data for the
combiners. The best entry by pooled log loss may be a single model; it is reported
either way. `recommend` goes one step further: an ensemble is recommended only if it
beats the best single model measurably (95% paired bootstrap interval below 0).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.dataset import AWAY, DRAW, HOME, Dataset, FloatArray
from app.ml.experiment import (
    Candidate,
    ExperimentResult,
    Fold,
    earlier_folds,
    pooled,
    warm,
)
from app.ml.metrics import SELECTION_METRIC, evaluate, match_log_losses, paired_bootstrap
from app.ml.predictors import Predictor
from app.ml.probability import MatchPrediction, normalise, reconcile
from app.ml.splits import ChronologicalSplit
from app.ml.tracking import Tracker

#: The models the sprint asked to combine: 1X2 classifiers, the goal model, Elo.
DEFAULT_MEMBERS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "poisson",
    "elo",
)
EPS = 1e-12
#: Weights below this are set to 0 (optimiser noise), then renormalised.
MIN_WEIGHT = 1e-4

Predictions = Mapping[str, Sequence[MatchPrediction]]


def _grids(predictions: Predictions, names: Sequence[str]) -> FloatArray:
    """Array (members, matches, 11, 11)."""
    return np.stack([np.stack([p.grid for p in predictions[name]]) for name in names])


def _outcome_probs(predictions: Predictions, names: Sequence[str]) -> FloatArray:
    """Array (members, matches, 3): home, draw, away."""
    return np.stack(
        [np.array([p.outcomes for p in predictions[name]], dtype=float) for name in names]
    )


class Combiner(ABC):
    """Turns member predictions for the same matches into one prediction per match."""

    method: ClassVar[str]

    def __init__(self, members: Sequence[str]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = tuple(members)
        self.fitted = False

    def fit(self, predictions: Predictions, data: Dataset) -> Self:
        self._check(predictions, data)
        self._fit(predictions, data)
        self.fitted = True
        return self

    def combine(self, predictions: Predictions) -> list[MatchPrediction]:
        if not self.fitted:
            raise RuntimeError(f"{self.method} combiner has not been fitted")
        return self._combine(predictions)

    def _check(self, predictions: Predictions, data: Dataset) -> None:
        missing = set(self.members) - set(predictions)
        if missing:
            raise ValueError(f"no predictions for members {sorted(missing)}")
        if any(len(predictions[m]) != len(data) for m in self.members):
            raise ValueError("every member needs one prediction per match")
        if len(data) == 0:
            raise ValueError("cannot fit a combiner on no matches")

    @abstractmethod
    def _fit(self, predictions: Predictions, data: Dataset) -> None: ...

    @abstractmethod
    def _combine(self, predictions: Predictions) -> list[MatchPrediction]: ...

    @abstractmethod
    def params(self) -> dict[str, Any]: ...


class _LinearPool(Combiner):
    weights: FloatArray

    def _combine(self, predictions: Predictions) -> list[MatchPrediction]:
        mixed = np.tensordot(self.weights, _grids(predictions, self.members), axes=1)
        return [MatchPrediction(normalise(grid)) for grid in mixed]

    def weight_map(self) -> dict[str, float]:
        return {m: float(w) for m, w in zip(self.members, self.weights, strict=True)}

    def params(self) -> dict[str, Any]:
        return {"method": self.method, "weights": self.weight_map()}


class SimpleAverage(_LinearPool):
    method = "simple_average"

    def __init__(self, members: Sequence[str]) -> None:
        super().__init__(members)
        self.weights = np.full(len(self.members), 1 / len(self.members))

    def _fit(self, predictions: Predictions, data: Dataset) -> None:
        return  # nothing to learn


class WeightedAverage(_LinearPool):
    method = "weighted_average"

    def __init__(self, members: Sequence[str]) -> None:
        super().__init__(members)
        self.weights = np.full(len(self.members), 1 / len(self.members))

    def _fit(self, predictions: Predictions, data: Dataset) -> None:
        probs = _outcome_probs(predictions, self.members)
        # p[i, m]: member m's probability of what actually happened in match i.
        p = probs[:, np.arange(len(data)), data.outcomes].T
        self.weights = optimal_weights(p)


def optimal_weights(p: FloatArray) -> FloatArray:
    """Weights on the simplex minimising -mean(log(p @ w)), where p[i, m] is member
    m's probability of the observed outcome of match i."""
    members = p.shape[1]
    if members == 1:
        return np.ones(1)

    def loss(w: FloatArray) -> float:
        return float(-np.mean(np.log(np.clip(p @ w, EPS, None))))

    def grad(w: FloatArray) -> FloatArray:
        gradient: FloatArray = -np.mean(p / np.clip(p @ w, EPS, None)[:, None], axis=0)
        return gradient

    result = minimize(
        loss,
        np.full(members, 1 / members),
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * members,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)}
        ],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    weights = np.clip(result.x, 0, None)
    weights[weights < MIN_WEIGHT] = 0.0
    total = weights.sum()
    return weights / total if total > 0 else np.full(members, 1 / members)


class Stacking(Combiner):
    """Multinomial logistic regression on the members' 1X2 log-odds."""

    method = "stacking"

    def __init__(self, members: Sequence[str], c: float = 1.0) -> None:
        super().__init__(members)
        self.c = c
        self.meta: Pipeline = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=5000)
        )

    def _features(self, predictions: Predictions) -> FloatArray:
        probs = np.clip(_outcome_probs(predictions, self.members), EPS, 1)
        log_home_draw = np.log(probs[:, :, HOME] / probs[:, :, DRAW])
        log_away_draw = np.log(probs[:, :, AWAY] / probs[:, :, DRAW])
        return np.concatenate([log_home_draw, log_away_draw]).T  # (matches, 2 * members)

    def _fit(self, predictions: Predictions, data: Dataset) -> None:
        if set(np.unique(data.outcomes)) != {HOME, DRAW, AWAY}:
            raise ValueError("stacking needs home wins, draws and away wins")
        self.meta.fit(self._features(predictions), data.outcomes)

    def _combine(self, predictions: Predictions) -> list[MatchPrediction]:
        target = self.meta.predict_proba(self._features(predictions))
        average = _grids(predictions, self.members).mean(axis=0)
        return [
            MatchPrediction(reconcile(normalise(grid), (float(t[0]), float(t[1]), float(t[2]))))
            for grid, t in zip(average, target, strict=True)
        ]

    def params(self) -> dict[str, Any]:
        coefficients = self.meta[-1].coef_ if self.fitted else None
        return {
            "method": self.method,
            "C": self.c,
            "coefficients": coefficients.round(4).tolist() if coefficients is not None else None,
        }


COMBINERS: dict[str, Callable[[Sequence[str]], Combiner]] = {
    SimpleAverage.method: SimpleAverage,
    WeightedAverage.method: WeightedAverage,
    Stacking.method: Stacking,
}


class EnsembleModel(Predictor):
    """Members plus a fitted combiner, behind the common model interface.

    `train(data)` refits every member on `data` and keeps the combiner (weights or
    meta-model) as fitted on validation data."""

    name = "ensemble"

    def __init__(self, members: Mapping[str, Predictor], combiner: Combiner) -> None:
        super().__init__()
        if tuple(members) != combiner.members:
            raise ValueError("members must match the combiner's members, in order")
        if not combiner.fitted:
            raise ValueError("the combiner must be fitted (on validation data) first")
        self.members = dict(members)
        self.combiner = combiner
        self.fitted = all(m.fitted for m in self.members.values())
        self._collect_features()

    def _collect_features(self) -> None:
        seen: dict[str, None] = {}
        for member in self.members.values():
            seen.update(dict.fromkeys(member.used_features))
        self.used_features = tuple(seen)

    def params(self) -> dict[str, Any]:
        return {
            **self.combiner.params(),
            "members": {name: m.params() for name, m in self.members.items()},
        }

    def _fit(self, train: Dataset) -> None:
        for member in self.members.values():
            member.fit(train)
        self._collect_features()

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        return self.combiner.combine({n: m.predict(data) for n, m in self.members.items()})


# Out-of-fold predictions and forward-chained evaluation ------------------------------


def ensemble_folds(split: ChronologicalSplit, cv_folds: int, min_team_history: int) -> list[Fold]:
    """The Sprint 6 rolling-origin folds, oldest first, ending with (train, validation)."""
    return [
        *earlier_folds(split.train, cv_folds - 1, min_team_history),
        Fold(warm(split.train, min_team_history), split.validation),
    ]


def out_of_fold_predictions(
    builders: Mapping[str, Callable[[], Predictor]], folds: Sequence[Fold]
) -> list[dict[str, list[MatchPrediction]]]:
    """For each fold, each member fitted on the fold's training data predicts its
    held-out block. No member ever predicts a match it was trained on."""
    return [
        {name: build().fit(fold.train).predict(fold.validation) for name, build in builders.items()}
        for fold in folds
    ]


def _concat(datasets: Sequence[Dataset]) -> Dataset:
    total = datasets[0]
    for data in datasets[1:]:
        total = total + data
    return total


def _stack(
    oof: Sequence[Mapping[str, Sequence[MatchPrediction]]], names: Sequence[str]
) -> dict[str, list[MatchPrediction]]:
    return {name: [p for fold in oof for p in fold[name]] for name in names}


@dataclass
class ForwardResult:
    #: Pooled metrics over the evaluated folds, for individuals and ensemble methods.
    metrics: dict[str, dict[str, float]]
    #: Per evaluated fold: the weights / parameters each combiner was fitted to.
    fitted: list[dict[str, dict[str, Any]]]
    evaluated_folds: int
    #: Per-match 1X2 log loss over the evaluated folds, for paired comparisons.
    losses: dict[str, list[float]] = field(default_factory=dict)


def forward_evaluation(
    oof: Sequence[Mapping[str, Sequence[MatchPrediction]]],
    folds: Sequence[Fold],
    members: Sequence[str],
    methods: Sequence[str],
) -> ForwardResult:
    """Fold k (k >= 2) is predicted by combiners fitted on folds 1..k-1 only."""
    if len(folds) < 2:
        raise ValueError("forward evaluation needs at least 2 folds (use --cv-folds >= 2)")
    per_entry: dict[str, list[dict[str, float]]] = {n: [] for n in [*members, *methods]}
    losses: dict[str, list[float]] = {n: [] for n in per_entry}
    fitted: list[dict[str, dict[str, Any]]] = []

    def score(name: str, predictions: Sequence[MatchPrediction], held_out: Dataset) -> None:
        per_entry[name].append(evaluate(predictions, held_out))
        losses[name] += match_log_losses(predictions, held_out).tolist()

    for k in range(1, len(folds)):
        held_out = folds[k].validation
        for name in members:
            score(name, oof[k][name], held_out)
        history = _concat([f.validation for f in folds[:k]])
        history_predictions = _stack(oof[:k], members)
        fold_params = {}
        for method in methods:
            combiner = COMBINERS[method](members).fit(history_predictions, history)
            score(method, combiner.combine(oof[k]), held_out)
            fold_params[method] = combiner.params()
        fitted.append(fold_params)
    return ForwardResult(
        metrics={name: pooled(values) for name, values in per_entry.items()},
        fitted=fitted,
        evaluated_folds=len(folds) - 1,
        losses=losses,
    )


# The ensemble stage of an experiment ---------------------------------------------------


@dataclass
class EnsembleCandidate:
    method: str
    combiner: Combiner
    model: EnsembleModel
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float] = field(default_factory=dict)
    test_losses: list[float] = field(default_factory=list)
    run_id: str | None = None


@dataclass
class EnsembleResult:
    members: list[str]
    folds: list[dict[str, Any]]
    forward: ForwardResult
    candidates: list[EnsembleCandidate]
    #: Best ensemble method by forward validation log loss.
    best_ensemble: str
    #: Best entry overall (individual model or ensemble) by forward validation log loss.
    best_overall: str
    #: What to use, and why (see `recommend`).
    recommendation: dict[str, Any] = field(default_factory=dict)

    def candidate(self, method: str) -> EnsembleCandidate:
        return next(c for c in self.candidates if c.method == method)

    @property
    def best_individual(self) -> str:
        methods = {c.method for c in self.candidates}
        return next(name for name, _ in self.ranking if name not in methods)

    @property
    def ranking(self) -> list[tuple[str, float]]:
        return sorted(
            ((name, m[SELECTION_METRIC]) for name, m in self.forward.metrics.items()),
            key=lambda kv: kv[1],
        )


def run_ensembles(
    split: ChronologicalSplit,
    experiment: ExperimentResult,
    candidates: Sequence[Candidate],
    *,
    cv_folds: int,
    min_team_history: int = 3,
    members: Sequence[str] = DEFAULT_MEMBERS,
    methods: Sequence[str] = tuple(COMBINERS),
    tracker: Tracker | None = None,
) -> EnsembleResult:
    """Build, evaluate and track ensembles of the tuned members of `experiment`."""
    available = {c.name for c in experiment.candidates}
    missing = [m for m in members if m not in available]
    if missing:
        raise ValueError(f"ensemble members were not trained: {missing}")
    if len(members) < 2:
        raise ValueError("an ensemble needs at least two members")
    folds = ensemble_folds(split, cv_folds, min_team_history)
    builds = {c.name: c.build for c in candidates}
    builders: dict[str, Callable[[], Predictor]] = {
        name: _builder(builds[name], experiment.candidate(name).params) for name in members
    }
    oof = out_of_fold_predictions(builders, folds)
    forward = forward_evaluation(oof, folds, members, methods)

    all_validation = _concat([f.validation for f in folds])
    all_predictions = _stack(oof, members)
    final_members: dict[str, Predictor] = {}
    for name in members:
        model = experiment.candidate(name).final_model
        assert model is not None  # noqa: S101 - set by evaluate_on_test
        final_members[name] = model
    results = []
    for method in methods:
        combiner = COMBINERS[method](members).fit(all_predictions, all_validation)
        model = EnsembleModel(final_members, combiner)
        entry = EnsembleCandidate(method, combiner, model, forward.metrics[method])
        if len(split.test):
            predictions = model.predict(split.test)
            entry.test_metrics = evaluate(predictions, split.test)
            entry.test_losses = match_log_losses(predictions, split.test).tolist()
        results.append(entry)

    ranking = sorted(forward.metrics, key=lambda n: forward.metrics[n][SELECTION_METRIC])
    best_ensemble = next(n for n in ranking if n in methods)
    best_individual = next(n for n in ranking if n not in methods)
    result = EnsembleResult(
        members=list(members),
        folds=[{"train": f.train.describe(), "validation": f.validation.describe()} for f in folds],
        forward=forward,
        candidates=results,
        best_ensemble=best_ensemble,
        best_overall=ranking[0],
        recommendation=recommend(forward, best_ensemble, best_individual),
    )
    if tracker is not None:
        _track(result, experiment, split, tracker)
    return result


def recommend(forward: ForwardResult, best_ensemble: str, best_individual: str) -> dict[str, Any]:
    """An ensemble is recommended only if it *measurably* beats the best single model
    on forward validation: the 95% paired bootstrap interval of its per-match log-loss
    difference must lie entirely below 0. Otherwise the simpler single model is
    recommended, even when the ensemble's point estimate is slightly lower."""
    gain = paired_bootstrap(forward.losses[best_ensemble], forward.losses[best_individual])
    measurable = gain["high"] < 0
    return {
        "recommended": best_ensemble if measurable else best_individual,
        "best_ensemble": best_ensemble,
        "best_individual": best_individual,
        "difference": gain["difference"],
        "interval": [gain["low"], gain["high"]],
        "rule": "ensemble only if its 95% forward-validation interval vs the best "
        "individual model lies entirely below 0",
        "reason": (
            f"{best_ensemble} beats {best_individual} measurably on forward validation"
            if measurable
            else f"{best_ensemble} does not measurably beat {best_individual} on forward "
            f"validation (difference {gain['difference']:+.4f}, 95% interval "
            f"[{gain['low']:+.4f}, {gain['high']:+.4f}]); the single model is simpler"
        ),
    }


def _builder(build: Callable[..., Predictor], params: Mapping[str, Any]) -> Callable[[], Predictor]:
    return lambda: build(**params)


def _track(
    result: EnsembleResult,
    experiment: ExperimentResult,
    split: ChronologicalSplit,
    tracker: Tracker,
) -> None:
    fit_period = experiment.candidate(result.members[0]).fit_period or {}
    for entry in result.candidates:
        tags = {
            "stage": "ensemble",
            "candidate": f"ensemble_{entry.method}",
            "model_type": "EnsembleModel",
            "best_ensemble": str(entry.method == result.best_ensemble).lower(),
            "best_overall": str(entry.method == result.best_overall).lower(),
            "recommended": str(entry.method == result.recommendation["recommended"]).lower(),
            "selection_metric": f"forward_validation_{SELECTION_METRIC}",
        }
        with tracker.run(f"ensemble {entry.method}", tags) as run_id:
            entry.run_id = run_id
            tracker.log_params(
                {
                    "model_type": "EnsembleModel",
                    "method": entry.method,
                    "members": ",".join(result.members),
                    "feature_set_version": split.train.feature_set_version,
                    "combiner_fit_start": result.folds[0]["validation"]["start"],
                    "combiner_fit_end": result.folds[-1]["validation"]["end"],
                    "combiner_fit_matches": sum(f["validation"]["matches"] for f in result.folds),
                    "fit_start": fit_period.get("start"),
                    "fit_end": fit_period.get("end"),
                    "test_start": split.test.describe()["start"],
                    "test_end": split.test.describe()["end"],
                    "forward_folds": result.forward.evaluated_folds,
                    **{
                        f"weight_{name}": round(weight, 6)
                        for name, weight in entry.combiner.params().get("weights", {}).items()
                    },
                }
            )
            tracker.log_metrics(entry.validation_metrics, "validation")
            if entry.test_metrics:
                tracker.log_metrics(entry.test_metrics, "test")
            tracker.log_dict(
                {
                    "combiner": entry.combiner.params(),
                    "members": {
                        name: {
                            "params": experiment.candidate(name).params,
                            "run_id": experiment.candidate(name).run_id,
                        }
                        for name in result.members
                    },
                    "folds": result.folds,
                    "forward_fitted": [fold[entry.method] for fold in result.forward.fitted],
                },
                "ensemble.json",
            )
            tracker.log_dict({"features": list(entry.model.used_features)}, "features.json")
            tracker.log_model(
                entry.model,
                {
                    "feature_set_version": split.train.feature_set_version,
                    "method": entry.method,
                    "members": result.members,
                },
            )
