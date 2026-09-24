"""Tuning, selection on validation only, test reporting, persistence and tracking."""

import dataclasses
import inspect
from pathlib import Path

import mlflow
import numpy as np
import pytest

from app.ml.dataset import Dataset, Example
from app.ml.experiment import (
    Candidate,
    default_candidates,
    run_experiment,
    tune_and_select,
)
from app.ml.feature_sets import CORE_FEATURES
from app.ml.persistence import load_model, save_model
from app.ml.predictors import EloModel, HistoricalBaseline, LogisticModel
from app.ml.splits import ChronologicalSplit, split_by_seasons
from app.ml.tracking import Tracker
from app.ml.trees import TreeEnsembleModel
from tests.ml.synthetic import make_dataset


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    return split_by_seasons(make_dataset(seasons=4, teams=10))


def small_candidates() -> list[Candidate]:
    return [
        Candidate("historical_baseline", HistoricalBaseline),
        Candidate("elo", EloModel),
        Candidate(
            "logistic_regression",
            lambda **p: LogisticModel(CORE_FEATURES, **p),
            ({"c": 0.01}, {"c": 1.0}),
        ),
    ]


class TestSelection:
    def test_selection_phase_cannot_see_test_data(self) -> None:
        parameters = inspect.signature(tune_and_select).parameters

        assert list(parameters)[:3] == ["train", "validation", "candidates"]
        assert "test" not in parameters and "split" not in parameters

    def test_poisoned_test_labels_do_not_change_the_selection(
        self, split: ChronologicalSplit
    ) -> None:
        normal = run_experiment(split, small_candidates())
        # Invert every test result: any model chosen on test data would now differ.
        flipped = tuple(
            dataclasses.replace(e, home_goals=e.away_goals, away_goals=e.home_goals + 3)
            for e in split.test.examples
        )
        poisoned = ChronologicalSplit(
            split.train,
            split.validation,
            Dataset(flipped, split.test.feature_set_version),
            "poisoned",
        )

        result = run_experiment(poisoned, small_candidates())

        assert result.selected == normal.selected
        for before, after in zip(normal.candidates, result.candidates, strict=True):
            assert before.params == after.params
            assert before.validation_metrics == after.validation_metrics
            assert before.test_metrics != after.test_metrics  # test changed, choices did not

    def test_selected_model_has_the_best_validation_log_loss(
        self, split: ChronologicalSplit
    ) -> None:
        result = run_experiment(split, small_candidates())

        best = min(result.candidates, key=lambda c: c.validation_metrics["log_loss"])
        assert result.selected == best.name
        assert result.selected != "historical_baseline"

    def test_hyperparameters_are_tuned_on_validation(self, split: ChronologicalSplit) -> None:
        result = run_experiment(split, small_candidates())

        logistic = result.candidate("logistic_regression")
        assert logistic.tuning_trials == 2
        losses = {}
        for c in (0.01, 1.0):
            model = LogisticModel(CORE_FEATURES, c=c).fit(split.train)
            from app.ml.metrics import evaluate

            losses[c] = evaluate(model.predict(split.validation), split.validation)["log_loss"]
        assert logistic.params["c"] == min(losses, key=losses.__getitem__)

    def test_final_models_are_refit_on_train_and_validation(
        self, split: ChronologicalSplit
    ) -> None:
        result = run_experiment(split, small_candidates(), min_team_history=0)

        for candidate in result.candidates:
            assert candidate.final_model is not None and candidate.final_model.fitted
            assert candidate.test_metrics and candidate.test_metrics["matches"] == len(split.test)

    def test_warm_up_filter_only_touches_training(self, split: ChronologicalSplit) -> None:
        cold = (
            tuple(
                dataclasses.replace(e, features={**e.features, "home_rated_matches": 0})
                for e in split.train.examples[:20]
            )
            + split.train.examples[20:]
        )
        cold_split = ChronologicalSplit(
            Dataset(cold, split.train.feature_set_version), split.validation, split.test, "cold"
        )

        result = run_experiment(cold_split, small_candidates()[:1], min_team_history=3)

        assert result.candidate("historical_baseline").validation_metrics["matches"] == len(
            split.validation
        )

    def test_default_candidates_cover_the_baselines_and_tree_models(self) -> None:
        names = [c.name for c in default_candidates()]

        assert names == [
            "historical_baseline",
            "elo",
            "logistic_regression",
            "poisson",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]
        for candidate in default_candidates():
            if candidate.name in ("random_forest", "xgboost", "lightgbm"):
                assert len(candidate.grid) > 1  # tuned, not fixed
                model = candidate.build(**candidate.grid[0])
                assert isinstance(model, TreeEnsembleModel)
                assert model.features == CORE_FEATURES
        assert "logistic_regression_with_odds" in [
            c.name for c in default_candidates(with_odds=True)
        ]


class TestPersistence:
    def test_saved_model_predicts_identically_after_loading(
        self, split: ChronologicalSplit, tmp_path: Path
    ) -> None:
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(split.train)
        save_model(model, tmp_path / "m", {"feature_set_version": split.train.feature_set_version})

        loaded = load_model(tmp_path / "m", feature_set_version=split.train.feature_set_version)

        before, after = model.predict(split.test), loaded.model.predict(split.test)
        assert all(np.array_equal(a.grid, b.grid) for a, b in zip(before, after, strict=True))
        assert loaded.metadata["model"] == "logistic_regression"
        assert loaded.metadata["params"] == {
            "C": 0.1,
            "goal_alpha": 1.0,
            "features": len(CORE_FEATURES),
        }
        assert "elo_diff" in loaded.metadata["used_features"]

    def test_loading_for_other_features_is_refused(
        self, split: ChronologicalSplit, tmp_path: Path
    ) -> None:
        save_model(EloModel().fit(split.train), tmp_path / "m", {"feature_set_version": "v1-aaa"})

        with pytest.raises(ValueError, match="trained on features"):
            load_model(tmp_path / "m", feature_set_version="v1-bbb")

    def test_unfitted_models_cannot_be_saved(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="fitted"):
            save_model(EloModel(), tmp_path / "m", {})


class TestTracking:
    def test_runs_record_params_metrics_features_periods_and_model(
        self, split: ChronologicalSplit, tmp_path: Path
    ) -> None:
        uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
        tracker = Tracker("test-experiment", uri)

        result = run_experiment(split, small_candidates(), tracker=tracker)

        runs = mlflow.search_runs(
            experiment_ids=[tracker.experiment_id],
            filter_string="tags.stage = 'final'",
            output_format="list",
        )
        assert {r.data.tags["candidate"] for r in runs} == {
            "historical_baseline",
            "elo",
            "logistic_regression",
        }
        selected = [r for r in runs if r.data.tags["selected"] == "true"]
        assert [r.data.tags["candidate"] for r in selected] == [result.selected]
        run = next(r for r in runs if r.data.tags["candidate"] == "logistic_regression")
        assert run.data.params["feature_set_version"] == split.train.feature_set_version
        assert "C" in run.data.params and run.data.params["fit_matches"]
        assert {"validation_log_loss", "test_log_loss", "test_top3_score_coverage"} <= set(
            run.data.metrics
        )
        # The logged model can be downloaded and used.
        local = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model", dst_path=str(tmp_path / "dl")
        )
        loaded = load_model(Path(local), feature_set_version=split.train.feature_set_version)
        assert len(loaded.model.predict(split.test)) == len(split.test)
        features = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/features.json")
        assert "elo_diff" in features["features"]
        tuning = mlflow.search_runs(
            experiment_ids=[tracker.experiment_id],
            filter_string="tags.stage = 'tuning'",
            output_format="list",
        )
        assert len(tuning) == 4  # 1 + 1 + 2 grid points


def test_dataset_examples_carry_no_future_fields() -> None:
    """Examples expose features and targets only; targets are never features."""
    example = make_dataset(seasons=1, teams=4).examples[0]

    assert {"home_goals", "away_goals"}.isdisjoint(example.features)
    assert isinstance(example, Example)
