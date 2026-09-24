"""Ensembles: weights from validation data only, measured against individual models."""

import dataclasses
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest

from app.ml.dataset import Dataset
from app.ml.engine import ScorePrediction
from app.ml.ensemble import (
    EnsembleModel,
    EnsembleResult,
    SimpleAverage,
    Stacking,
    WeightedAverage,
    ensemble_folds,
    forward_evaluation,
    optimal_weights,
    out_of_fold_predictions,
    run_ensembles,
)
from app.ml.experiment import Candidate, ExperimentResult, run_experiment
from app.ml.feature_sets import CORE_FEATURES
from app.ml.metrics import evaluate
from app.ml.predictors import EloModel, LogisticModel, PoissonModel, Predictor
from app.ml.probability import MatchPrediction
from app.ml.report import render_report
from app.ml.splits import ChronologicalSplit, split_by_seasons
from app.ml.tracking import Tracker
from app.ml.trees import XGBoostModel
from tests.ml.synthetic import make_dataset
from tests.ml.test_score_engine import assert_consistent

MEMBERS = ("logistic_regression", "poisson", "xgboost", "elo")


class Contrarian(EloModel):
    """Elo with home and away swapped: confidently wrong. A useful ensemble drops it."""

    name = "contrarian"

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        return [MatchPrediction(p.grid.T.copy()) for p in super()._predict(data)]


def candidates() -> list[Candidate]:
    return [
        Candidate("elo", EloModel),
        Candidate(
            "logistic_regression",
            lambda **p: LogisticModel(CORE_FEATURES, **p),
            ({"c": 0.01}, {"c": 0.1}),
        ),
        Candidate("poisson", lambda **p: PoissonModel(CORE_FEATURES, **p), ({"alpha": 3.0},)),
        Candidate(
            "xgboost",
            lambda **p: XGBoostModel(CORE_FEATURES, **p),
            ({"max_depth": 2, "n_estimators": 60},),
        ),
        Candidate("contrarian", Contrarian),
    ]


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    """Six seasons: four train, one validation, one test."""
    return split_by_seasons(make_dataset(seasons=6, teams=10))


@pytest.fixture(scope="module")
def experiment(split: ChronologicalSplit) -> ExperimentResult:
    return run_experiment(split, candidates(), cv_folds=3)


@pytest.fixture(scope="module")
def ensemble(split: ChronologicalSplit, experiment: ExperimentResult) -> EnsembleResult:
    return run_ensembles(split, experiment, candidates(), cv_folds=3, members=MEMBERS)


# Weight optimisation --------------------------------------------------------------------


class TestOptimalWeights:
    def test_weights_are_on_the_simplex(self) -> None:
        rng = np.random.default_rng(0)
        weights = optimal_weights(rng.uniform(0.05, 0.9, size=(300, 4)))

        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()

    def test_a_dominant_member_takes_all_the_weight(self) -> None:
        rng = np.random.default_rng(1)
        weak = rng.uniform(0.2, 0.4, 500)
        p = np.column_stack([weak, weak + 0.3])  # member 2 always better

        assert optimal_weights(p) == pytest.approx([0.0, 1.0], abs=1e-6)

    def test_complementary_members_are_mixed_evenly(self) -> None:
        """Each member is right on half the matches: the symmetric optimum is 50/50."""
        p = np.array([[0.9, 0.1]] * 200 + [[0.1, 0.9]] * 200)

        assert optimal_weights(p) == pytest.approx([0.5, 0.5], abs=1e-4)

    def test_optimum_is_no_worse_than_any_member_or_the_average(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(0.05, 0.95, size=(400, 3))
        weights = optimal_weights(p)

        def loss(w: np.ndarray) -> float:
            return float(-np.mean(np.log(p @ w)))

        best = loss(weights)
        assert best <= loss(np.full(3, 1 / 3)) + 1e-9
        for member in np.eye(3):
            assert best <= loss(member) + 1e-9


# Combiners -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def member_predictions(split: ChronologicalSplit) -> dict[str, list[MatchPrediction]]:
    models: dict[str, Predictor] = {
        "elo": EloModel(),
        "poisson": PoissonModel(CORE_FEATURES),
        "contrarian": Contrarian(),
    }
    return {name: m.fit(split.train).predict(split.validation) for name, m in models.items()}


def test_simple_average_is_the_mean_grid(
    split: ChronologicalSplit, member_predictions: dict[str, list[MatchPrediction]]
) -> None:
    combiner = SimpleAverage(["elo", "poisson"]).fit(member_predictions, split.validation)

    combined = combiner.combine(member_predictions)

    for i, prediction in enumerate(combined):
        mean = (member_predictions["elo"][i].grid + member_predictions["poisson"][i].grid) / 2
        np.testing.assert_allclose(prediction.grid, mean, atol=1e-12)


def test_weighted_average_drops_a_harmful_member(
    split: ChronologicalSplit, member_predictions: dict[str, list[MatchPrediction]]
) -> None:
    combiner = WeightedAverage(["elo", "poisson", "contrarian"]).fit(
        member_predictions, split.validation
    )

    weights = combiner.params()["weights"]
    assert weights["contrarian"] < 0.02
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weight_one_reproduces_the_member(
    split: ChronologicalSplit, member_predictions: dict[str, list[MatchPrediction]]
) -> None:
    combiner = WeightedAverage(["elo", "poisson"]).fit(member_predictions, split.validation)
    combiner.weights = np.array([1.0, 0.0])

    for mixed, elo in zip(
        combiner.combine(member_predictions), member_predictions["elo"], strict=True
    ):
        np.testing.assert_allclose(mixed.grid, elo.grid, atol=1e-12)


@pytest.mark.parametrize("combiner_class", [SimpleAverage, WeightedAverage, Stacking])
def test_every_combiner_yields_consistent_predictions(
    combiner_class: Any,
    split: ChronologicalSplit,
    member_predictions: dict[str, list[MatchPrediction]],
) -> None:
    combiner = combiner_class(["elo", "poisson", "contrarian"]).fit(
        member_predictions, split.validation
    )

    for prediction in combiner.combine(member_predictions):
        assert_consistent(ScorePrediction.from_grid(prediction.grid))


def test_stacking_uses_its_meta_model_for_1x2(
    split: ChronologicalSplit, member_predictions: dict[str, list[MatchPrediction]]
) -> None:
    combiner = Stacking(["elo", "poisson"]).fit(member_predictions, split.validation)

    combined = combiner.combine(member_predictions)
    target = combiner.meta.predict_proba(combiner._features(member_predictions))

    np.testing.assert_allclose([p.outcomes for p in combined], target, atol=1e-9)


def test_combiners_refuse_bad_input(
    split: ChronologicalSplit, member_predictions: dict[str, list[MatchPrediction]]
) -> None:
    with pytest.raises(ValueError, match="no predictions for members"):
        WeightedAverage(["elo", "xgboost"]).fit(member_predictions, split.validation)
    with pytest.raises(ValueError, match="one prediction per match"):
        half = split.validation.where(lambda e: e.match_id % 2 == 0)
        WeightedAverage(["elo"]).fit(member_predictions, half)
    with pytest.raises(RuntimeError, match="not been fitted"):
        WeightedAverage(["elo"]).combine(member_predictions)


# Validation-only fitting -----------------------------------------------------------------


def test_combiners_are_fitted_on_out_of_fold_validation_data_only(
    split: ChronologicalSplit,
) -> None:
    folds = ensemble_folds(split, cv_folds=3, min_team_history=3)
    test_ids = {e.match_id for e in split.test.examples}

    for fold in folds:
        held_out = {e.match_id for e in fold.validation.examples}
        assert not held_out & test_ids
        assert not held_out & {e.match_id for e in fold.train.examples}  # out of fold
        assert fold.train.examples[-1].kickoff_at < fold.validation.examples[0].kickoff_at
    assert folds[-1].validation is split.validation


def test_forward_step_weights_ignore_later_blocks(split: ChronologicalSplit) -> None:
    """Changing the results of the last block must not change the weights fitted
    before it: step k only sees blocks 1..k."""
    folds = ensemble_folds(split, cv_folds=3, min_team_history=3)
    builders = {"elo": EloModel, "poisson": lambda: PoissonModel(CORE_FEATURES)}
    oof = out_of_fold_predictions(builders, folds)
    normal = forward_evaluation(oof, folds, list(builders), ["weighted_average", "stacking"])
    last = folds[-1]
    flipped = Dataset(
        tuple(
            dataclasses.replace(e, home_goals=e.away_goals, away_goals=e.home_goals + 2)
            for e in last.validation.examples
        ),
        last.validation.feature_set_version,
    )
    poisoned = [*folds[:-1], dataclasses.replace(last, validation=flipped)]

    result = forward_evaluation(oof, poisoned, list(builders), ["weighted_average", "stacking"])

    assert result.fitted == normal.fitted  # weights at every step unchanged
    assert result.metrics != normal.metrics  # only the scores of the last block moved


def test_poisoned_test_labels_change_nothing_but_test_figures(
    split: ChronologicalSplit, experiment: ExperimentResult, ensemble: EnsembleResult
) -> None:
    flipped = tuple(
        dataclasses.replace(e, home_goals=e.away_goals, away_goals=e.home_goals + 3)
        for e in split.test.examples
    )
    poisoned = ChronologicalSplit(
        split.train, split.validation, Dataset(flipped, split.test.feature_set_version), "p"
    )
    poisoned_experiment = run_experiment(poisoned, candidates(), cv_folds=3)

    result = run_ensembles(poisoned, poisoned_experiment, candidates(), cv_folds=3, members=MEMBERS)

    assert result.forward.metrics == ensemble.forward.metrics
    assert result.forward.fitted == ensemble.forward.fitted
    assert result.best_overall == ensemble.best_overall
    for before, after in zip(ensemble.candidates, result.candidates, strict=True):
        assert before.combiner.params() == after.combiner.params()
        assert before.test_metrics != after.test_metrics


# Measured comparison --------------------------------------------------------------------


def test_every_approach_is_scored_on_the_same_matches(ensemble: EnsembleResult) -> None:
    expected = sum(f["validation"]["matches"] for f in ensemble.folds[1:])

    assert set(ensemble.forward.metrics) == {
        *MEMBERS,
        "simple_average",
        "weighted_average",
        "stacking",
    }
    for metrics in ensemble.forward.metrics.values():
        assert metrics["matches"] == expected
    assert ensemble.forward.evaluated_folds == len(ensemble.folds) - 1 == 2


def test_best_overall_is_the_lowest_forward_log_loss(ensemble: EnsembleResult) -> None:
    ranking = ensemble.ranking

    assert ensemble.best_overall == ranking[0][0]
    assert [loss for _, loss in ranking] == sorted(loss for _, loss in ranking)
    methods = {c.method for c in ensemble.candidates}
    assert ensemble.best_ensemble == next(name for name, _ in ranking if name in methods)


def test_a_single_model_can_win(split: ChronologicalSplit, experiment: ExperimentResult) -> None:
    """More members are not assumed to help: an ensemble with a harmful member is
    reported as worse than its best member when it is."""
    result = run_ensembles(
        split,
        experiment,
        candidates(),
        cv_folds=3,
        members=("elo", "contrarian"),
        methods=("simple_average",),
    )

    assert (
        result.forward.metrics["simple_average"]["log_loss"]
        > result.forward.metrics["elo"]["log_loss"]
    )
    assert result.best_overall == "elo"


def test_run_ensembles_needs_trained_members_and_two_folds(
    split: ChronologicalSplit, experiment: ExperimentResult
) -> None:
    with pytest.raises(ValueError, match="not trained"):
        run_ensembles(split, experiment, candidates(), cv_folds=3, members=("elo", "lightgbm"))
    with pytest.raises(ValueError, match="at least 2 folds"):
        run_ensembles(split, experiment, candidates(), cv_folds=1, members=MEMBERS)


# The ensemble model -------------------------------------------------------------------------


def test_final_ensembles_use_the_refitted_members(
    split: ChronologicalSplit, experiment: ExperimentResult, ensemble: EnsembleResult
) -> None:
    for entry in ensemble.candidates:
        for name in MEMBERS:
            assert entry.model.members[name] is experiment.candidate(name).final_model
        assert entry.test_metrics["matches"] == len(split.test)
        direct = entry.combiner.combine(
            {n: m.predict(split.test) for n, m in entry.model.members.items()}
        )
        assert evaluate(direct, split.test) == entry.test_metrics


def test_ensemble_model_has_the_common_interface(
    split: ChronologicalSplit, ensemble: EnsembleResult, tmp_path: Path
) -> None:
    model = ensemble.candidate("weighted_average").model
    probs = model.predict_proba(split.test)

    model.save(tmp_path / "ens", {"feature_set_version": split.test.feature_set_version})
    loaded = EnsembleModel.load(
        tmp_path / "ens", feature_set_version=split.test.feature_set_version
    )

    assert probs.shape == (len(split.test), 3)
    np.testing.assert_array_equal(loaded.predict_proba(split.test), probs)
    assert loaded.combiner.params() == model.combiner.params()
    # train() refits the members but keeps the validation-fitted weights.
    before = model.combiner.params()
    refit = loaded.train(split.train + split.validation)
    assert refit.combiner.params() == before


def test_ensemble_model_rejects_mismatched_parts(
    member_predictions: Any, split: ChronologicalSplit
) -> None:
    unfitted = WeightedAverage(["elo", "poisson"])
    members = {
        "elo": EloModel().fit(split.train),
        "poisson": PoissonModel(CORE_FEATURES).fit(split.train),
    }

    with pytest.raises(ValueError, match="fitted"):
        EnsembleModel(members, unfitted)
    fitted = WeightedAverage(["poisson", "elo"]).fit(member_predictions, split.validation)
    with pytest.raises(ValueError, match="in order"):
        EnsembleModel(members, fitted)


# Tracking and report ----------------------------------------------------------------------


def test_ensembles_are_tracked_as_their_own_models(
    split: ChronologicalSplit, experiment: ExperimentResult, tmp_path: Path
) -> None:
    tracker = Tracker("ensembles", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    result = run_ensembles(
        split, experiment, candidates(), cv_folds=3, members=MEMBERS, tracker=tracker
    )

    runs = mlflow.search_runs(
        experiment_ids=[tracker.experiment_id],
        filter_string="tags.stage = 'ensemble'",
        output_format="list",
    )
    assert {r.data.tags["candidate"] for r in runs} == {
        "ensemble_simple_average",
        "ensemble_weighted_average",
        "ensemble_stacking",
    }
    run = next(r for r in runs if r.data.tags["candidate"] == "ensemble_weighted_average")
    assert run.data.params["model_type"] == "EnsembleModel"
    assert run.data.params["members"] == ",".join(MEMBERS)
    assert {f"weight_{m}" for m in MEMBERS} <= set(run.data.params)
    assert {"validation_log_loss", "test_log_loss", "test_over_2_5_brier"} <= set(run.data.metrics)
    details = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/ensemble.json")
    assert set(details["members"]) == set(MEMBERS)
    assert len(details["forward_fitted"]) == 2
    local = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="model", dst_path=str(tmp_path / "dl")
    )
    loaded = EnsembleModel.load(Path(local), feature_set_version=split.train.feature_set_version)
    np.testing.assert_array_equal(
        loaded.predict_proba(split.test),
        result.candidate("weighted_average").model.predict_proba(split.test),
    )


def test_report_has_the_ensemble_section(
    experiment: ExperimentResult, ensemble: EnsembleResult
) -> None:
    report = render_report(experiment, ensemble=ensemble)

    for heading in (
        "## Ensembles",
        "### Forward-chained validation (the basis for every choice)",
        "### Weights",
        "### Test (reported only)",
    ):
        assert heading in report
    assert f"**{ensemble.best_overall} (" in report
    assert "(combiner training only)" in report


def test_forward_losses_support_paired_comparisons(ensemble: EnsembleResult) -> None:
    from app.ml.report import forward_comparisons

    expected = sum(f["validation"]["matches"] for f in ensemble.folds[1:])
    for name, losses in ensemble.forward.losses.items():
        assert len(losses) == expected
        assert np.mean(losses) == pytest.approx(ensemble.forward.metrics[name]["log_loss"])
    rows = forward_comparisons(ensemble)
    reference = ensemble.best_individual
    assert reference not in rows and reference not in {c.method for c in ensemble.candidates}
    for name, row in rows.items():
        gap = (
            ensemble.forward.metrics[name]["log_loss"]
            - ensemble.forward.metrics[reference]["log_loss"]
        )
        assert row["difference"] == pytest.approx(gap)
        assert row["low"] <= row["difference"] <= row["high"]


def test_an_ensemble_is_recommended_only_when_it_measurably_wins(
    ensemble: EnsembleResult,
) -> None:
    from app.ml.metrics import paired_bootstrap

    advice = ensemble.recommendation
    gain = paired_bootstrap(
        ensemble.forward.losses[ensemble.best_ensemble],
        ensemble.forward.losses[ensemble.best_individual],
    )
    assert advice["interval"] == [gain["low"], gain["high"]]
    expected = ensemble.best_ensemble if gain["high"] < 0 else ensemble.best_individual
    assert advice["recommended"] == expected


def test_a_tie_recommends_the_single_model() -> None:
    from app.ml.ensemble import ForwardResult, recommend

    rng = np.random.default_rng(3)
    base = rng.uniform(0.5, 1.5, 1000)
    noise = rng.normal(0, 0.05, 1000)

    def forward(gain: float) -> ForwardResult:
        ensemble = (base - gain + noise).tolist()
        losses = {"elo": base.tolist(), "weighted_average": ensemble}
        return ForwardResult(metrics={}, fitted=[], evaluated_folds=3, losses=losses)

    tie, clear = forward(0.0002), forward(0.05)

    assert recommend(tie, "weighted_average", "elo")["recommended"] == "elo"
    assert recommend(clear, "weighted_average", "elo")["recommended"] == "weighted_average"
