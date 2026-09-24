"""Why a model produced the prediction it did: per-feature contributions.

**What an explanation is.** It describes *the model's behaviour*: how this fitted
model turned this match's feature vector into its output, relative to a stated
baseline. A contribution is not a cause. "The home team's recent form contributed
+0.31 to the model's home-win score" means the model's arithmetic moved that way;
it does not mean recent form makes a team win, and it does not mean that changing
that feature in the world would change the result. Features are correlated, so
credit given to one of them is partly credit for everything it moves with.

**Methods, and where each is valid.**

- `logistic_regression`: Shapley values of a linear model, each contribution being
  the coefficient times the standardised value. Exact: the baseline plus the
  contributions is the model's own score.
- `poisson`: the same, on each side's log goal rate. Exact.
- `elo`: one feature, so its contribution is the whole difference from the
  baseline. Exact by definition.
- `random_forest`, `xgboost`, `lightgbm`: TreeSHAP (the `shap` package), when it
  is installed. Exact for the tree model.
- `historical_baseline`, `ensemble`: refused, with the reason.

The linear method is SHAP's linear explainer with the independent-features
assumption; the baseline is the average training match, which is where the fitted
intercept applies. `StandardScaler` centres the training data exactly, so the
contributions add up to the model's own score with no residual.

`historical_baseline` reads no features, and an ensemble mixes several models'
score grids: the mixture's probability is not the weighted sum of its members'
log-odds, so there is no honest additive decomposition to report. Both refuse
rather than return something that looks like an explanation.

**What is explained.** For the models with a 1X2 classifier the contributions are
the classifier's, which is exactly the prediction's match-result probabilities
(the score grid is rescaled to them). The goal regressors that shape the score
distribution are not covered. For `poisson` there is no classifier: the two log
goal rates are explained instead, and the 1X2 follows from them.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np

from app.features.registry import DESCRIPTIONS, GROUPS
from app.ml.dataset import AWAY, DRAW, HOME, Dataset, FloatArray
from app.ml.predictors import (
    EloModel,
    HistoricalBaseline,
    LogisticModel,
    NotFittedError,
    PoissonModel,
    Predictor,
)

#: The additive scale a target's contributions are expressed on.
SCORE = "score"
PROBABILITY = "probability"
LOG_RATE = "log goal rate"

#: Suffix of the flag a linear pipeline adds for a feature that had missing values.
MISSING_SUFFIX = "__missing"

OUTCOME_KEYS = ("home_win", "draw", "away_win")
OUTCOME_LABELS = ("Home win", "Draw", "Away win")

CAVEAT = (
    "These figures describe how the model combined the data it was given. "
    "They are not causes: a contribution does not mean the feature made the result "
    "happen, and features that move together share the credit between them."
)

EPS = 1e-12


class ExplanationUnavailableError(RuntimeError):
    """This model cannot be explained by any method available here."""


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    group: str
    #: The feature's value for this match; None when the data was missing.
    value: float | None
    #: Signed, on the target's `unit` scale. Positive pushes the target up.
    contribution: float


@dataclass(frozen=True)
class TargetExplanation:
    """One thing the model outputs, decomposed into feature contributions."""

    key: str
    label: str
    #: The scale the contributions add on.
    unit: str
    #: The model's output at the baseline, and for this match, on `unit`.
    baseline: float
    predicted: float
    #: The same two numbers in readable terms, when there are any.
    output_unit: str | None
    baseline_output: float | None
    predicted_output: float | None
    #: Every feature the model used, largest absolute contribution first.
    contributions: tuple[Contribution, ...]

    @property
    def total(self) -> float:
        return self.baseline + sum(c.contribution for c in self.contributions)


@dataclass(frozen=True)
class Explanation:
    model: str
    #: Short name of the method, e.g. "linear Shapley values".
    method: str
    #: What the method does, in one sentence.
    method_detail: str
    #: What the contributions are measured against.
    baseline_description: str
    targets: tuple[TargetExplanation, ...]
    features_used: int


@dataclass(frozen=True)
class FeatureImportance:
    feature: str
    label: str
    group: str
    importance: float


@dataclass(frozen=True)
class ImportanceReport:
    """How much the model relies on each feature in general, across all matches."""

    model: str
    method: str
    method_detail: str
    entries: tuple[FeatureImportance, ...]


def label_of(feature: str) -> str:
    """A readable name for a feature, from the feature registry."""
    described = DESCRIPTIONS.get(feature)
    if described:
        return described[0].upper() + described[1:]
    if feature.endswith(MISSING_SUFFIX):
        base = feature[: -len(MISSING_SUFFIX)]
        return f"{label_of(base)} — not available for this match"
    return feature.replace("_", " ").capitalize()


def group_of(feature: str) -> str:
    base = feature[: -len(MISSING_SUFFIX)] if feature.endswith(MISSING_SUFFIX) else feature
    return GROUPS.get(base, "other")


def explain(model: Predictor, data: Dataset, index: int = 0) -> Explanation:
    """Explain one match's prediction, `data.examples[index]`.

    Raises `ExplanationUnavailable` when no valid method applies to this model.
    """
    if not model.fitted:
        raise NotFittedError(f"{model.name} has not been fitted")
    if not 0 <= index < len(data):
        raise IndexError(f"index {index} is outside the dataset")
    if isinstance(model, HistoricalBaseline):
        raise ExplanationUnavailableError(
            "the historical baseline reads no features: it predicts the same score "
            "distribution for every match, so there is nothing to attribute"
        )
    if isinstance(model, LogisticModel):
        return _explain_logistic(model, data, index)
    if isinstance(model, PoissonModel):
        return _explain_poisson(model, data, index)
    if isinstance(model, EloModel):
        return _explain_elo(model, data, index)
    if _is_tree_ensemble(model):
        return _explain_trees(model, data, index)
    raise ExplanationUnavailableError(
        f"no explanation method is available for {model.name}; an ensemble mixes "
        "several models' score grids, and the mixture's probabilities are not the "
        "sum of its members' contributions"
    )


def importance(model: Predictor) -> ImportanceReport:
    """How much the model relies on each feature overall (not for one match)."""
    if not model.fitted:
        raise NotFittedError(f"{model.name} has not been fitted")
    if _is_tree_ensemble(model):
        values = model.feature_importances()  # type: ignore[attr-defined]
        return ImportanceReport(
            model=model.name,
            method="impurity importance",
            method_detail=(
                "the tree library's own importance measure, normalised to sum to 1. "
                "It says how much the 1X2 classifier's splits used each feature, not "
                "how a feature affects any single prediction."
            ),
            entries=_ranked(values),
        )
    if isinstance(model, LogisticModel):
        imputer, _, estimator = _pipeline_parts(model.classifier)
        weights = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        return ImportanceReport(
            model=model.name,
            method="standardised coefficients",
            method_detail=(
                "the average absolute coefficient across the three outcomes, on "
                "standardised features: how far the model's score moves per standard "
                "deviation of the feature."
            ),
            entries=_ranked(_named(_column_names(imputer, model.used_features), weights)),
        )
    if isinstance(model, PoissonModel):
        names: list[str] = []
        stacked: list[FloatArray] = []
        for pipeline in (model.goals.home, model.goals.away):
            imputer, _, estimator = _pipeline_parts(pipeline)
            names = _column_names(imputer, model.used_features)
            stacked.append(np.abs(np.asarray(estimator.coef_, dtype=float)))
        weights = np.mean(stacked, axis=0)
        return ImportanceReport(
            model=model.name,
            method="standardised coefficients",
            method_detail=(
                "the average absolute coefficient of the two goal regressions, on "
                "standardised features: how far the log goal rate moves per standard "
                "deviation of the feature."
            ),
            entries=_ranked(_named(names, weights)),
        )
    if isinstance(model, EloModel):
        return ImportanceReport(
            model=model.name,
            method="single feature",
            method_detail=(
                "the Elo model's match result depends on one feature, the rating "
                "difference, so it carries all of the weight."
            ),
            entries=_ranked({EloModel.FEATURE: 1.0}),
        )
    raise ExplanationUnavailableError(f"no importance measure is available for {model.name}")


# Linear models ----------------------------------------------------------------------


def _pipeline_parts(pipeline: Any) -> tuple[Any, Any, Any]:
    """The imputer, scaler and estimator of a `_regression_pipeline`."""
    return (
        pipeline.named_steps["simpleimputer"],
        pipeline.named_steps["standardscaler"],
        pipeline.steps[-1][1],
    )


def _column_names(imputer: Any, features: tuple[str, ...]) -> list[str]:
    """The columns the estimator sees: the kept features, then missingness flags."""
    statistics = np.asarray(imputer.statistics_, dtype=float)
    names = [name for name, value in zip(features, statistics, strict=True) if not np.isnan(value)]
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        names += [f"{features[i]}{MISSING_SUFFIX}" for i in indicator.features_]
    return names


def _linear_contributions(pipeline: Any, row: FloatArray) -> tuple[FloatArray, FloatArray]:
    """`(baseline, contributions)` for one row: coefficient x standardised value.

    The scaler centres the training data, so the contributions of a linear model are
    exactly its Shapley values against the average training match, and
    `baseline + contributions.sum(axis=-1)` is the model's own score.
    """
    imputer, scaler, estimator = _pipeline_parts(pipeline)
    z = scaler.transform(imputer.transform(row.reshape(1, -1)))[0]
    coefficients = np.asarray(estimator.coef_, dtype=float)
    intercept = np.asarray(estimator.intercept_, dtype=float)
    return intercept, coefficients * z


def _values_of(data: Dataset, index: int, features: tuple[str, ...]) -> dict[str, float | None]:
    stored = data.examples[index].features
    return {name: _as_float(stored.get(name)) for name in features}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _contributions(
    names: list[str], values: dict[str, float | None], weights: FloatArray
) -> tuple[Contribution, ...]:
    rows = [
        Contribution(
            feature=name,
            label=label_of(name),
            group=group_of(name),
            value=_flag_value(name, values),
            contribution=float(weight),
        )
        for name, weight in zip(names, weights, strict=True)
    ]
    return tuple(sorted(rows, key=lambda row: -abs(row.contribution)))


def _flag_value(name: str, values: dict[str, float | None]) -> float | None:
    """A missingness flag is 1 when the feature it refers to has no value."""
    if not name.endswith(MISSING_SUFFIX):
        return values.get(name)
    return 1.0 if values.get(name[: -len(MISSING_SUFFIX)]) is None else 0.0


def _softmax(scores: FloatArray) -> FloatArray:
    shifted = np.exp(scores - scores.max())
    return np.asarray(shifted / shifted.sum(), dtype=np.float64)


def _explain_logistic(model: LogisticModel, data: Dataset, index: int) -> Explanation:
    row = data.matrix(model.used_features)[index]
    intercept, contributions = _linear_contributions(model.classifier, row)
    imputer, _, _ = _pipeline_parts(model.classifier)
    names = _column_names(imputer, model.used_features)
    values = _values_of(data, index, model.used_features)
    scores = intercept + contributions.sum(axis=1)
    baseline_probabilities = _softmax(intercept)
    probabilities = _softmax(scores)
    targets = tuple(
        TargetExplanation(
            key=OUTCOME_KEYS[outcome],
            label=OUTCOME_LABELS[outcome],
            unit=SCORE,
            baseline=float(intercept[outcome]),
            predicted=float(scores[outcome]),
            output_unit=PROBABILITY,
            baseline_output=float(baseline_probabilities[outcome]),
            predicted_output=float(probabilities[outcome]),
            contributions=_contributions(names, values, contributions[outcome]),
        )
        for outcome in (HOME, DRAW, AWAY)
    )
    return Explanation(
        model=model.name,
        method="linear Shapley values",
        method_detail=(
            "each contribution is the model's coefficient multiplied by the "
            "standardised feature value, which for a linear model is its exact "
            "Shapley value; they add up to the model's score for that outcome."
        ),
        baseline_description="an average match in the model's training data",
        targets=targets,
        features_used=len(names),
    )


def _explain_poisson(model: PoissonModel, data: Dataset, index: int) -> Explanation:
    row = data.matrix(model.used_features)[index]
    values = _values_of(data, index, model.used_features)
    targets = []
    names: list[str] = []
    for key, label, pipeline in (
        ("home_goals", "Home goals", model.goals.home),
        ("away_goals", "Away goals", model.goals.away),
    ):
        intercept, contributions = _linear_contributions(pipeline, row)
        imputer, _, _ = _pipeline_parts(pipeline)
        names = _column_names(imputer, model.used_features)
        log_rate = float(intercept[0] if intercept.ndim else intercept) + float(contributions.sum())
        targets.append(
            TargetExplanation(
                key=key,
                label=label,
                unit=LOG_RATE,
                baseline=float(np.ravel(intercept)[0]),
                predicted=log_rate,
                output_unit="goals",
                baseline_output=float(np.exp(np.ravel(intercept)[0])),
                predicted_output=float(np.exp(log_rate)),
                contributions=_contributions(names, values, np.ravel(contributions)),
            )
        )
    return Explanation(
        model=model.name,
        method="linear Shapley values",
        method_detail=(
            "each contribution is the regression's coefficient multiplied by the "
            "standardised feature value, on the log goal rate; they add up to the "
            "model's rate for that side, from which every other market follows."
        ),
        baseline_description="an average match in the model's training data",
        targets=tuple(targets),
        features_used=len(names),
    )


def _explain_elo(model: EloModel, data: Dataset, index: int) -> Explanation:
    value = _as_float(data.examples[index].features.get(EloModel.FEATURE))
    x = 0.0 if value is None else value / 400.0
    # The model's own ordered-logit function, so the explanation and the prediction
    # can never drift apart.
    probabilities = model._probabilities
    predicted = probabilities(np.array([x]), model.slope, *model.cutpoints)[0]
    baseline = probabilities(np.array([0.0]), model.slope, *model.cutpoints)[0]
    targets = tuple(
        TargetExplanation(
            key=OUTCOME_KEYS[outcome],
            label=OUTCOME_LABELS[outcome],
            unit=SCORE,
            baseline=_logit(float(baseline[outcome])),
            predicted=_logit(float(predicted[outcome])),
            output_unit=PROBABILITY,
            baseline_output=float(baseline[outcome]),
            predicted_output=float(predicted[outcome]),
            contributions=(
                Contribution(
                    feature=EloModel.FEATURE,
                    label=label_of(EloModel.FEATURE),
                    group=group_of(EloModel.FEATURE),
                    value=value,
                    contribution=_logit(float(predicted[outcome]))
                    - _logit(float(baseline[outcome])),
                ),
            ),
        )
        for outcome in (HOME, DRAW, AWAY)
    )
    return Explanation(
        model=model.name,
        method="single feature",
        method_detail=(
            "the model's match result depends on one feature, the rating difference, "
            "so its contribution is the whole difference between this match's "
            "probabilities and the baseline's."
        ),
        baseline_description="two equally rated teams (rating difference of zero)",
        targets=targets,
        features_used=1,
    )


def _logit(probability: float) -> float:
    clipped = min(max(probability, EPS), 1 - EPS)
    return float(np.log(clipped / (1 - clipped)))


# Tree ensembles ---------------------------------------------------------------------


def _is_tree_ensemble(model: Predictor) -> bool:
    """True for the tree models, without importing them (the serving image has no
    xgboost or lightgbm, so `app.ml.trees` may not be importable here)."""
    return hasattr(model, "feature_importances") and hasattr(model, "classifier")


def shap_available() -> bool:
    return find_spec("shap") is not None


def _explain_trees(model: Any, data: Dataset, index: int) -> Explanation:
    if not shap_available():
        raise ExplanationUnavailableError(
            f"explaining {model.name} needs TreeSHAP, and the shap package is not "
            "installed in this environment"
        )
    import shap  # an optional dependency: imported only where it is used

    row = data.matrix(model.used_features)[index : index + 1]
    explainer = shap.TreeExplainer(model.classifier)
    values = np.asarray(explainer.shap_values(row), dtype=float)
    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if values.ndim != 3 or values.shape[2] != len(base):  # pragma: no cover - guard
        raise ExplanationUnavailableError(
            f"TreeSHAP returned an unexpected shape {values.shape} for {model.name}"
        )
    names = list(model.used_features)
    values_by_name = _values_of(data, index, model.used_features)
    unit = PROBABILITY if _outputs_probabilities(model) else SCORE
    totals = base + values[0].sum(axis=0)
    outputs = totals if unit == PROBABILITY else _softmax(totals)
    baseline_outputs = base if unit == PROBABILITY else _softmax(base)
    targets = tuple(
        TargetExplanation(
            key=OUTCOME_KEYS[outcome],
            label=OUTCOME_LABELS[outcome],
            unit=unit,
            baseline=float(base[outcome]),
            predicted=float(totals[outcome]),
            output_unit=PROBABILITY,
            baseline_output=float(baseline_outputs[outcome]),
            predicted_output=float(outputs[outcome]),
            contributions=_contributions(names, values_by_name, values[0][:, outcome]),
        )
        for outcome in (HOME, DRAW, AWAY)
    )
    return Explanation(
        model=model.name,
        method="TreeSHAP",
        method_detail=(
            "exact Shapley values for tree ensembles (Lundberg et al.), computed by "
            "the shap package over the 1X2 classifier's trees; they add up to the "
            "classifier's own output."
        ),
        baseline_description="the average prediction over the model's training data",
        targets=targets,
        features_used=len(names),
    )


def _outputs_probabilities(model: Any) -> bool:
    """Random forests give TreeSHAP probabilities; the boosters give margins."""
    return bool(model.name == "random_forest")


# Shared -----------------------------------------------------------------------------


def _named(names: list[str], weights: FloatArray) -> dict[str, float]:
    return {name: float(weight) for name, weight in zip(names, weights, strict=True)}


def _ranked(values: dict[str, float]) -> tuple[FeatureImportance, ...]:
    total = sum(abs(value) for value in values.values())
    scale = total if total > 0 else 1.0
    entries = [
        FeatureImportance(
            feature=name,
            label=label_of(name),
            group=group_of(name),
            importance=abs(value) / scale,
        )
        for name, value in values.items()
    ]
    return tuple(sorted(entries, key=lambda entry: -entry.importance))
