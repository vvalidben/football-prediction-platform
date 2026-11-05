"""Training, tuning, selection and evaluation of the prediction models.

Two phases, deliberately separated so the test set cannot influence any choice:

1. **Select** (`tune_and_select`). This phase receives only train and validation
   data. Each candidate's hyperparameters are tuned, then the candidate with the
   best score is selected. The score is the validation log loss or, with time-aware
   cross-validation (`cv_folds > 1`), the log loss pooled over rolling-origin folds:
   the validation period plus the last training seasons, each predicted by a model
   fitted only on earlier matches.
2. **Report** (`evaluate_on_test`). Each tuned candidate is refitted on train +
   validation and evaluated once on the test period. The selection is already fixed
   and is not revisited.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.ml.dataset import Dataset, Example
from app.ml.feature_sets import CORE_FEATURES, WITH_ODDS
from app.ml.metrics import SELECTION_METRIC, evaluate, match_log_losses, metrics_by_competition
from app.ml.predictors import (
    EloModel,
    HistoricalBaseline,
    LogisticModel,
    PoissonModel,
    Predictor,
)
from app.ml.splits import ChronologicalSplit
from app.ml.tracking import Tracker
from app.ml.trees import LightGBMModel, RandomForestModel, TreeEnsembleModel, XGBoostModel


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[..., Predictor]
    grid: tuple[dict[str, Any], ...] = ({},)


def default_candidates(*, with_odds: bool = False) -> list[Candidate]:
    candidates = [
        Candidate("historical_baseline", HistoricalBaseline, ({"smoothing": 1.0},)),
        Candidate("elo", EloModel),
        Candidate(
            "logistic_regression",
            lambda **p: LogisticModel(CORE_FEATURES, **p),
            tuple({"c": c, "goal_alpha": 10.0} for c in (0.003, 0.01, 0.03, 0.1, 1.0)),
        ),
        Candidate(
            "poisson",
            lambda **p: PoissonModel(CORE_FEATURES, **p),
            tuple({"alpha": a} for a in (1.0, 3.0, 10.0, 30.0)),
        ),
        Candidate(
            "random_forest",
            lambda **p: RandomForestModel(CORE_FEATURES, **p),
            tuple(
                {"max_depth": depth, "min_samples_leaf": leaf}
                for depth in (4, 8)
                for leaf in (10, 30, 60)
            ),
        ),
        Candidate(
            "xgboost",
            lambda **p: XGBoostModel(CORE_FEATURES, **p),
            tuple(
                {"max_depth": depth, "n_estimators": trees}
                for depth in (2, 3)
                for trees in (100, 250, 500)
            ),
        ),
        Candidate(
            "lightgbm",
            lambda **p: LightGBMModel(CORE_FEATURES, **p),
            tuple(
                {"num_leaves": leaves, "n_estimators": trees}
                for leaves in (3, 7)
                for trees in (100, 250, 500)
            ),
        ),
    ]
    if with_odds:
        candidates.append(
            Candidate(
                "logistic_regression_with_odds",
                lambda **p: LogisticModel(WITH_ODDS, **p),
                tuple({"c": c, "goal_alpha": 10.0} for c in (0.01, 0.1, 1.0)),
            )
        )
    return candidates


@dataclass(frozen=True)
class Fold:
    """A model fitted on `train` is evaluated on `validation`, which follows it."""

    train: Dataset
    validation: Dataset


@dataclass
class CandidateResult:
    name: str
    params: dict[str, Any]
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    tuning_trials: int
    #: Metrics pooled over all cross-validation folds (None without cross-validation).
    cv_metrics: dict[str, float] | None = None
    cv_folds: int = 1
    model_type: str = ""
    test_metrics: dict[str, float] | None = None
    #: The test figures per competition (`metrics.metrics_by_competition`).
    test_by_competition: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Per-match 1X2 log loss on the test period, for paired comparisons.
    test_losses: list[float] = field(default_factory=list)
    final_model: Predictor | None = None
    #: The data the final model was actually fitted on (train after warm-up + validation).
    fit_period: dict[str, Any] | None = None
    #: Dataset version (fingerprint) and competitions of that fit data.
    fit_dataset_version: str | None = None
    fit_competitions: list[int] = field(default_factory=list)
    run_id: str | None = None

    @property
    def selection_score(self) -> float:
        return (self.cv_metrics or self.validation_metrics)[SELECTION_METRIC]


@dataclass
class ExperimentResult:
    split: dict[str, Any]
    candidates: list[CandidateResult]
    selected: str
    feature_set_version: str
    selection_metric: str = f"validation_{SELECTION_METRIC}"
    cv_folds: list[dict[str, Any]] = field(default_factory=list)
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def candidate(self, name: str) -> CandidateResult:
        return next(c for c in self.candidates if c.name == name)


def warm(train: Dataset, min_team_history: int) -> Dataset:
    """Training rows where both teams have enough rated history. Early matches, with
    ratings still at their defaults, mostly add noise. Only training is filtered;
    validation and test keep every match."""
    if min_team_history <= 0:
        return train
    return train.where(
        lambda e: (
            (e.features.get("home_rated_matches") or 0) >= min_team_history
            and (e.features.get("away_rated_matches") or 0) >= min_team_history
        )
    )


def _before(cut: datetime) -> Callable[[Example], bool]:
    return lambda e: e.kickoff_at < cut


def _between(start: datetime, end: datetime | None) -> Callable[[Example], bool]:
    return lambda e: e.kickoff_at >= start and (end is None or e.kickoff_at < end)


def earlier_folds(train: Dataset, count: int, min_team_history: int) -> list[Fold]:
    """Rolling-origin folds inside the training period, for time-aware cross-validation.

    The held-out blocks are the last `count` training seasons (ordered by their first
    kickoff). Each block is predicted by a model fitted only on matches before the
    block starts (warm-up filtered); held-out blocks keep every match, as validation
    does. The first season is never held out: nothing precedes it. Like
    `split_by_seasons`, this is meant for single-competition data.
    """
    if count <= 0:
        return []
    first_kickoff: dict[int, datetime] = {}
    for example in train.examples:
        first_kickoff.setdefault(example.season_id, example.kickoff_at)
    cuts = sorted(first_kickoff.values())[1:][-count:]
    folds = []
    for i, cut in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else None
        fit = warm(train.where(_before(cut)), min_team_history)
        held_out = train.where(_between(cut, end))
        if len(fit) and len(held_out):
            folds.append(Fold(fit, held_out))
    return folds


def pooled(metrics: Sequence[dict[str, float]]) -> dict[str, float]:
    """Match-weighted mean of per-fold metrics (as if all folds were one dataset)."""
    total = sum(m["matches"] for m in metrics)
    return {
        key: total if key == "matches" else sum(m[key] * m["matches"] for m in metrics) / total
        for key in metrics[0]
    }


def _model_type(model: Predictor) -> str:
    return type(model).__name__


def tune_and_select(
    train: Dataset,
    validation: Dataset,
    candidates: Sequence[Candidate],
    *,
    tracker: Tracker | None = None,
    folds: Sequence[Fold] = (),
) -> tuple[list[CandidateResult], str]:
    """Phase 1. Sees no test data by construction.

    `folds` are extra, earlier cross-validation folds (from `earlier_folds`, inside
    the training period). Without them, tuning and selection use the validation log
    loss; with them, the log loss pooled over those folds and (train, validation).
    """
    results = []
    for candidate in candidates:
        best: CandidateResult | None = None
        for params in candidate.grid:
            model = candidate.build(**params).fit(train)
            validation_metrics = evaluate(model.predict(validation), validation)
            cv_metrics = None
            if folds:
                fold_metrics = [
                    evaluate(
                        candidate.build(**params).fit(f.train).predict(f.validation),
                        f.validation,
                    )
                    for f in folds
                ]
                cv_metrics = pooled([*fold_metrics, validation_metrics])
            trial = CandidateResult(
                name=candidate.name,
                params=params,
                train_metrics={},
                validation_metrics=validation_metrics,
                tuning_trials=len(candidate.grid),
                cv_metrics=cv_metrics,
                cv_folds=len(folds) + 1,
                model_type=_model_type(model),
            )
            if tracker is not None:
                with tracker.run(
                    f"{candidate.name} tuning",
                    {
                        "stage": "tuning",
                        "candidate": candidate.name,
                        "model_type": trial.model_type,
                    },
                ):
                    tracker.log_params(
                        {
                            **model.params(),
                            **params,
                            "model_type": trial.model_type,
                            "feature_set_version": train.feature_set_version,
                            "train_start": train.describe()["start"],
                            "train_end": train.describe()["end"],
                            "train_matches": len(train),
                            "cv_folds": trial.cv_folds,
                        }
                    )
                    tracker.log_metrics(validation_metrics, "validation")
                    if cv_metrics is not None:
                        tracker.log_metrics(cv_metrics, "cv")
            if best is None or trial.selection_score < best.selection_score:
                trial.train_metrics = evaluate(model.predict(train), train)
                best = trial
        assert best is not None  # noqa: S101 - every candidate has at least one grid entry
        results.append(best)
    selected = min(results, key=lambda r: r.selection_score).name
    return results, selected


def evaluate_on_test(
    results: Sequence[CandidateResult],
    candidates: Sequence[Candidate],
    split: ChronologicalSplit,
    selected: str,
    *,
    train_for_final: Dataset,
    tracker: Tracker | None = None,
    selection_metric: str = f"validation_{SELECTION_METRIC}",
) -> None:
    """Phase 2. Refit each tuned candidate on train + validation, evaluate on test."""
    builders = {c.name: c for c in candidates}
    final_train = train_for_final + split.validation
    for result in results:
        model = builders[result.name].build(**result.params).fit(final_train)
        result.final_model = model
        result.fit_period = final_train.describe()
        result.fit_dataset_version = final_train.fingerprint()
        result.fit_competitions = final_train.competition_ids
        if len(split.test):
            predictions = model.predict(split.test)
            result.test_metrics = evaluate(predictions, split.test)
            result.test_by_competition = metrics_by_competition(predictions, split.test)
            result.test_losses = match_log_losses(predictions, split.test).tolist()
        else:
            result.test_metrics = {}
        if tracker is None:
            continue
        tags = {
            "stage": "final",
            "candidate": result.name,
            "model_type": result.model_type,
            "selected": str(result.name == selected).lower(),
            "selection_metric": selection_metric,
        }
        with tracker.run(f"{result.name} final", tags) as run_id:
            result.run_id = run_id
            dataset_info = {
                "feature_set_version": split.train.feature_set_version,
                "split": split.description,
                "fit_period": final_train.describe(),
                "train_period": split.train.describe(),
                "validation_period": split.validation.describe(),
                "test_period": split.test.describe(),
            }
            tracker.log_params(
                {
                    **model.params(),
                    **result.params,
                    "model_type": result.model_type,
                    "feature_set_version": split.train.feature_set_version,
                    "dataset_version": result.fit_dataset_version,
                    "fit_start": final_train.describe()["start"],
                    "fit_end": final_train.describe()["end"],
                    "fit_matches": len(final_train),
                    "test_start": split.test.describe()["start"],
                    "test_end": split.test.describe()["end"],
                    "cv_folds": result.cv_folds,
                }
            )
            tracker.log_metrics(result.validation_metrics, "validation")
            if result.cv_metrics is not None:
                tracker.log_metrics(result.cv_metrics, "cv")
            tracker.log_metrics(result.test_metrics, "test")
            tracker.log_dict({"features": list(model.used_features)}, "features.json")
            tracker.log_dict(dataset_info, "dataset.json")
            if isinstance(model, TreeEnsembleModel):
                tracker.log_dict(model.feature_importances(), "feature_importances.json")
            tracker.log_model(model, {**dataset_info, "selected": result.name == selected})


def run_experiment(
    split: ChronologicalSplit,
    candidates: Sequence[Candidate] | None = None,
    *,
    tracker: Tracker | None = None,
    min_team_history: int = 3,
    cv_folds: int = 1,
) -> ExperimentResult:
    """`cv_folds` counts all tuning folds: 1 = validation only (no cross-validation);
    n > 1 adds the last n - 1 training seasons as earlier folds."""
    candidates = list(candidates or default_candidates())
    train = warm(split.train, min_team_history)
    folds = earlier_folds(split.train, cv_folds - 1, min_team_history)
    selection_metric = f"{'cv' if folds else 'validation'}_{SELECTION_METRIC}"
    results, selected = tune_and_select(
        train, split.validation, candidates, tracker=tracker, folds=folds
    )
    evaluate_on_test(
        results,
        candidates,
        split,
        selected,
        train_for_final=train,
        tracker=tracker,
        selection_metric=selection_metric,
    )
    return ExperimentResult(
        split=split.describe(),
        candidates=results,
        selected=selected,
        feature_set_version=split.train.feature_set_version,
        selection_metric=selection_metric,
        cv_folds=[
            {"train": f.train.describe(), "validation": f.validation.describe()} for f in folds
        ]
        + [{"train": train.describe(), "validation": split.validation.describe()}],
    )
