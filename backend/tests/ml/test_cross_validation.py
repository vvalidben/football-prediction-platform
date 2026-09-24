"""Time-aware cross-validation, tracking of the Sprint 6 fields, and the comparison report."""

import dataclasses
from pathlib import Path

import mlflow
import numpy as np
import pytest

from app.ml.dataset import Dataset
from app.ml.experiment import (
    Candidate,
    earlier_folds,
    pooled,
    run_experiment,
    warm,
)
from app.ml.feature_sets import CORE_FEATURES
from app.ml.metrics import evaluate, match_log_losses, paired_bootstrap
from app.ml.predictors import EloModel, HistoricalBaseline, LogisticModel
from app.ml.report import comparisons, render_report
from app.ml.splits import ChronologicalSplit, split_by_seasons
from app.ml.tracking import Tracker
from app.ml.trees import XGBoostModel
from tests.ml.synthetic import make_dataset


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    """Six seasons: four train, one validation, one test."""
    return split_by_seasons(make_dataset(seasons=6, teams=10))


def candidates() -> list[Candidate]:
    return [
        Candidate("historical_baseline", HistoricalBaseline),
        Candidate("elo", EloModel),
        Candidate(
            "logistic_regression",
            lambda **p: LogisticModel(CORE_FEATURES, **p),
            ({"c": 0.01}, {"c": 1.0}),
        ),
        Candidate(
            "xgboost",
            lambda **p: XGBoostModel(CORE_FEATURES, **p),
            ({"max_depth": 2, "n_estimators": 50}, {"max_depth": 3, "n_estimators": 150}),
        ),
    ]


class TestRollingOriginFolds:
    def test_each_fold_predicts_a_later_training_season_from_earlier_matches(
        self, split: ChronologicalSplit
    ) -> None:
        folds = earlier_folds(split.train, 2, min_team_history=0)

        assert len(folds) == 2
        seasons = [{e.season_id for e in f.validation.examples} for f in folds]
        assert seasons == [{3}, {4}]  # the last two of four training seasons
        for fold in folds:
            assert fold.train.examples[-1].kickoff_at < fold.validation.examples[0].kickoff_at
            assert fold.validation.examples[-1].kickoff_at < split.validation.examples[0].kickoff_at
        assert len(folds[1].train) > len(folds[0].train)  # expanding window

    def test_folds_never_reach_the_validation_or_test_period(
        self, split: ChronologicalSplit
    ) -> None:
        train_ids = {e.match_id for e in split.train.examples}

        for fold in earlier_folds(split.train, 10, min_team_history=3):
            assert {e.match_id for e in fold.train.examples} <= train_ids
            assert {e.match_id for e in fold.validation.examples} <= train_ids

    def test_the_first_season_is_never_held_out(self, split: ChronologicalSplit) -> None:
        folds = earlier_folds(split.train, 10, min_team_history=0)

        assert len(folds) == 3  # seasons 2, 3 and 4
        assert earlier_folds(split.train, 0, min_team_history=0) == []

    def test_only_fold_training_data_is_warm_up_filtered(self, split: ChronologicalSplit) -> None:
        cold = tuple(
            dataclasses.replace(e, features={**e.features, "home_rated_matches": 0})
            if e.season_id == 3
            else e
            for e in split.train.examples
        )
        train = Dataset(cold, split.train.feature_set_version)

        last = earlier_folds(train, 1, min_team_history=3)[0]

        assert not any(e.season_id == 3 for e in last.train.examples)
        assert len(last.validation) == sum(1 for e in train.examples if e.season_id == 4)
        assert len(warm(train, 3)) < len(train)

    def test_pooled_metrics_weight_folds_by_matches(self) -> None:
        result = pooled([{"matches": 100.0, "log_loss": 1.0}, {"matches": 300.0, "log_loss": 0.8}])

        assert result == {"matches": 400.0, "log_loss": pytest.approx(0.85)}


class TestCrossValidatedSelection:
    def test_selection_uses_the_pooled_cross_validation_score(
        self, split: ChronologicalSplit
    ) -> None:
        result = run_experiment(split, candidates(), cv_folds=3)

        assert result.selection_metric == "cv_log_loss"
        assert len(result.cv_folds) == 3
        for candidate in result.candidates:
            assert candidate.cv_metrics is not None and candidate.cv_folds == 3
            assert candidate.selection_score == candidate.cv_metrics["log_loss"]
            validation_matches = candidate.validation_metrics["matches"]
            assert candidate.cv_metrics["matches"] > validation_matches
        best = min(result.candidates, key=lambda c: c.selection_score)
        assert result.selected == best.name

    def test_hyperparameters_are_tuned_on_the_pooled_folds(self, split: ChronologicalSplit) -> None:
        result = run_experiment(split, candidates(), cv_folds=3)
        train = warm(split.train, 3)
        folds = earlier_folds(split.train, 2, 3)

        scores = {}
        for c in (0.01, 1.0):
            per_fold = [
                evaluate(
                    LogisticModel(CORE_FEATURES, c=c).fit(f.train).predict(f.validation),
                    f.validation,
                )
                for f in folds
            ]
            model = LogisticModel(CORE_FEATURES, c=c).fit(train)
            per_fold.append(evaluate(model.predict(split.validation), split.validation))
            scores[c] = pooled(per_fold)["log_loss"]
        assert result.candidate("logistic_regression").params["c"] == min(
            scores, key=scores.__getitem__
        )

    def test_poisoned_test_labels_change_nothing_with_cross_validation(
        self, split: ChronologicalSplit
    ) -> None:
        normal = run_experiment(split, candidates(), cv_folds=3)
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

        result = run_experiment(poisoned, candidates(), cv_folds=3)

        assert result.selected == normal.selected
        for before, after in zip(normal.candidates, result.candidates, strict=True):
            assert before.params == after.params
            assert before.cv_metrics == after.cv_metrics
            assert before.validation_metrics == after.validation_metrics
            assert before.test_metrics != after.test_metrics

    def test_cv_folds_1_is_validation_only(self, split: ChronologicalSplit) -> None:
        result = run_experiment(split, candidates()[:2], cv_folds=1)

        assert result.selection_metric == "validation_log_loss"
        assert all(c.cv_metrics is None for c in result.candidates)


def test_runs_record_model_type_periods_cv_and_test_metrics(
    split: ChronologicalSplit, tmp_path: Path
) -> None:
    tracker = Tracker("sprint6", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    result = run_experiment(split, candidates(), tracker=tracker, cv_folds=2)

    final = mlflow.search_runs(
        experiment_ids=[tracker.experiment_id],
        filter_string="tags.stage = 'final'",
        output_format="list",
    )
    run = next(r for r in final if r.data.tags["candidate"] == "xgboost")
    assert run.data.tags["model_type"] == run.data.params["model_type"] == "XGBoostModel"
    assert run.data.tags["selection_metric"] == "cv_log_loss"
    params = run.data.params
    assert params["feature_set_version"] == split.train.feature_set_version
    assert {"max_depth", "n_estimators", "learning_rate", "fit_start", "fit_end"} <= set(params)
    assert params["cv_folds"] == "2"
    assert {"validation_log_loss", "cv_log_loss", "test_log_loss", "test_rps"} <= set(
        run.data.metrics
    )
    importances = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/feature_importances.json")
    assert "elo_diff" in importances
    tuning = mlflow.search_runs(
        experiment_ids=[tracker.experiment_id],
        filter_string="tags.stage = 'tuning' and tags.candidate = 'xgboost'",
        output_format="list",
    )
    assert len(tuning) == 2
    assert all({"train_start", "train_end", "model_type"} <= set(t.data.params) for t in tuning)
    assert all("cv_log_loss" in t.data.metrics for t in tuning)
    assert result.candidate("xgboost").run_id in {r.info.run_id for r in final}


class TestReport:
    def test_paired_bootstrap_of_identical_models_is_zero(self) -> None:
        losses = list(np.linspace(0.2, 2.0, 200))

        row = paired_bootstrap(losses, losses)

        assert row == {"difference": 0.0, "low": 0.0, "high": 0.0, "better_share": 0.0}

    def test_paired_bootstrap_detects_a_consistent_improvement(self) -> None:
        rng = np.random.default_rng(1)
        reference = rng.uniform(0.3, 2.0, 300)
        better = reference - 0.05 + rng.normal(0, 0.02, 300)

        row = paired_bootstrap(better.tolist(), reference.tolist())

        assert row["difference"] == pytest.approx(-0.05, abs=0.01)
        assert row["low"] < row["difference"] < row["high"] < 0
        assert row["better_share"] == 1.0

    def test_report_shows_every_model_and_the_protocol(self, split: ChronologicalSplit) -> None:
        result = run_experiment(split, candidates(), cv_folds=3)

        report = render_report(result)

        for name in ("historical_baseline", "elo", "logistic_regression", "xgboost"):
            assert report.count(name) >= 4  # selection, validation, CV and test tables
        for heading in (
            "## Selection (no test data)",
            "## Cross-validation metrics",
            "## Test metrics (reported only)",
            "## Test log loss compared with the selected model",
            "## Most important features",
            "## Derived markets on test",
        ):
            assert heading in report
        assert f"**{result.selected}** (selected)" in report
        assert "It influenced no choice." in report

    def test_comparisons_use_per_match_test_losses(self, split: ChronologicalSplit) -> None:
        result = run_experiment(split, candidates())
        selected = result.candidate(result.selected)
        assert selected.final_model is not None and selected.test_metrics is not None

        losses = match_log_losses(selected.final_model.predict(split.test), split.test)

        assert selected.test_losses == pytest.approx(losses.tolist())
        assert np.mean(losses) == pytest.approx(selected.test_metrics["log_loss"])
        diffs = comparisons(result)
        assert set(diffs) == {c.name for c in result.candidates} - {result.selected}
        for name, row in diffs.items():
            other = result.candidate(name).test_metrics
            assert other is not None
            gap = other["log_loss"] - np.mean(losses)
            assert row["difference"] == pytest.approx(gap)


def test_console_summary_names_the_actual_selection_metric(
    split: ChronologicalSplit, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.ml.__main__ import print_results

    print_results(run_experiment(split, candidates(), cv_folds=3))

    output = capsys.readouterr().out
    assert "selected on cv_log_loss" in output
    assert "pooled over 3 rolling-origin folds" in output
    assert "95% paired bootstrap interval" in output
