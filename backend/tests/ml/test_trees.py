"""Tree-ensemble models, and the common model interface shared by every model."""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from app.ml.dataset import Dataset
from app.ml.feature_sets import CORE_FEATURES
from app.ml.metrics import evaluate
from app.ml.predictors import (
    EloModel,
    HistoricalBaseline,
    LogisticModel,
    NotFittedError,
    PoissonModel,
    Predictor,
)
from app.ml.splits import ChronologicalSplit, split_by_seasons
from app.ml.trees import LightGBMModel, RandomForestModel, TreeEnsembleModel, XGBoostModel
from tests.ml.synthetic import make_dataset

TREES: dict[str, Callable[[], TreeEnsembleModel]] = {
    "random_forest": lambda: RandomForestModel(CORE_FEATURES, n_estimators=100),
    "xgboost": lambda: XGBoostModel(CORE_FEATURES, n_estimators=150),
    "lightgbm": lambda: LightGBMModel(CORE_FEATURES, n_estimators=150),
}
ALL_MODELS: dict[str, Callable[[], Predictor]] = {
    "historical_baseline": HistoricalBaseline,
    "elo": EloModel,
    "logistic_regression": lambda: LogisticModel(CORE_FEATURES, c=0.1),
    "poisson": lambda: PoissonModel(CORE_FEATURES),
    **TREES,
}


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    return split_by_seasons(make_dataset(seasons=4, teams=12))


@pytest.fixture(scope="module", params=list(TREES))
def tree(request: pytest.FixtureRequest, split: ChronologicalSplit) -> TreeEnsembleModel:
    return TREES[request.param]().fit(split.train)


class TestTreeModels:
    def test_predictions_are_normalised_consistent_distributions(
        self, tree: TreeEnsembleModel, split: ChronologicalSplit
    ) -> None:
        predictions = tree.predict(split.validation)

        assert len(predictions) == len(split.validation)
        for p in predictions:
            assert p.grid.sum() == pytest.approx(1.0, abs=1e-9)
            assert sum(p.outcomes) == pytest.approx(1.0, abs=1e-9)
            assert all(0 < q < 1 for q in p.outcomes)
            assert 0 < p.expected_home_goals < 6 and 0 < p.expected_away_goals < 6

    def test_classifier_probabilities_are_kept_exactly(
        self, tree: TreeEnsembleModel, split: ChronologicalSplit
    ) -> None:
        """The score grid is rescaled to the classifier's 1X2, not the reverse."""
        x = split.validation.matrix(tree.used_features)

        np.testing.assert_allclose(
            tree.predict_proba(split.validation), tree.classifier.predict_proba(x), atol=1e-9
        )

    def test_repeated_predictions_are_identical(
        self, tree: TreeEnsembleModel, split: ChronologicalSplit
    ) -> None:
        for _ in range(3):
            np.testing.assert_array_equal(
                tree.predict_proba(split.validation), tree.predict_proba(split.validation)
            )

    @pytest.mark.parametrize("name", list(TREES))
    def test_refitting_gives_identical_predictions(
        self, name: str, split: ChronologicalSplit
    ) -> None:
        first = TREES[name]().fit(split.train).predict_proba(split.validation)
        second = TREES[name]().fit(split.train).predict_proba(split.validation)

        np.testing.assert_array_equal(first, second)

    def test_beats_the_historical_baseline(
        self, tree: TreeEnsembleModel, split: ChronologicalSplit
    ) -> None:
        baseline = HistoricalBaseline().fit(split.train)

        base = evaluate(baseline.predict(split.validation), split.validation)
        score = evaluate(tree.predict(split.validation), split.validation)

        assert score["log_loss"] < base["log_loss"]
        assert score["total_goals_mae"] < base["total_goals_mae"] + 0.05

    def test_uses_the_shared_feature_vector_and_drops_unobserved_features(
        self, tree: TreeEnsembleModel
    ) -> None:
        assert tree.features == CORE_FEATURES
        assert set(tree.used_features) <= set(CORE_FEATURES)
        assert "elo_diff" in tree.used_features
        assert "h2h_weighted_points" not in tree.used_features  # always None here

    def test_missing_values_need_no_imputation(self, split: ChronologicalSplit) -> None:
        holes = tuple(
            dataclasses.replace(e, features={**e.features, "elo_diff": None}) if i % 3 == 0 else e
            for i, e in enumerate(split.train.examples)
        )
        train = Dataset(holes, split.train.feature_set_version)

        for build in TREES.values():
            probs = build().fit(train).predict_proba(split.validation)
            assert np.isfinite(probs).all()

    def test_feature_importances_rank_the_signal_first(self, tree: TreeEnsembleModel) -> None:
        importances = tree.feature_importances()

        assert set(importances) == set(tree.used_features)
        assert sum(importances.values()) == pytest.approx(1.0)
        assert list(importances.values()) == sorted(importances.values(), reverse=True)
        top = list(importances)[:3]
        assert {"elo_diff", "rating_expected_home_goals", "rating_expected_away_goals"} & set(top)

    def test_hyperparameters_are_tracked_and_validated(self) -> None:
        model = XGBoostModel(CORE_FEATURES, max_depth=2)

        assert model.params()["max_depth"] == 2
        assert model.params()["learning_rate"] == XGBoostModel.DEFAULTS["learning_rate"]
        assert model.params()["features"] == len(CORE_FEATURES)
        with pytest.raises(ValueError, match="unknown xgboost hyperparameters"):
            XGBoostModel(CORE_FEATURES, max_dept=2)

    def test_training_needs_all_three_outcomes(self, split: ChronologicalSplit) -> None:
        no_draws = split.train.where(lambda e: e.home_goals != e.away_goals)

        with pytest.raises(ValueError, match="draws"):
            LightGBMModel(CORE_FEATURES).fit(no_draws)


@pytest.mark.parametrize("name", list(ALL_MODELS))
class TestCommonInterface:
    """train() / predict() / predict_proba() / save() / load() on every model."""

    def test_train_predict_and_predict_proba(self, name: str, split: ChronologicalSplit) -> None:
        model = ALL_MODELS[name]()
        with pytest.raises(NotFittedError):
            model.predict_proba(split.validation)

        assert model.train(split.train) is model and model.fitted
        probs = model.predict_proba(split.validation)
        predictions = model.predict(split.validation)

        assert probs.shape == (len(split.validation), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs, [p.outcomes for p in predictions])

    def test_save_and_load_round_trip(
        self, name: str, split: ChronologicalSplit, tmp_path: Path
    ) -> None:
        model = ALL_MODELS[name]().train(split.train)
        model.save(tmp_path / "v1", {"feature_set_version": split.train.feature_set_version})

        loaded = type(model).load(
            tmp_path / "v1", feature_set_version=split.train.feature_set_version
        )

        assert type(loaded) is type(model)
        np.testing.assert_array_equal(
            loaded.predict_proba(split.validation), model.predict_proba(split.validation)
        )

    def test_saving_never_overwrites_an_existing_model(
        self, name: str, split: ChronologicalSplit, tmp_path: Path
    ) -> None:
        first = ALL_MODELS[name]().train(split.train)
        first.save(tmp_path / "v1", {"feature_set_version": "v1-a"})
        before = (tmp_path / "v1" / "model.joblib").read_bytes()
        second = ALL_MODELS[name]().train(split.validation)

        with pytest.raises(FileExistsError, match="already holds a saved model"):
            second.save(tmp_path / "v1", {"feature_set_version": "v1-a"})

        assert (tmp_path / "v1" / "model.joblib").read_bytes() == before


def test_loading_as_the_wrong_model_class_is_refused(
    split: ChronologicalSplit, tmp_path: Path
) -> None:
    EloModel().train(split.train).save(tmp_path / "elo", {"feature_set_version": "v1-a"})

    with pytest.raises(TypeError, match="not a XGBoostModel"):
        XGBoostModel.load(tmp_path / "elo")
    assert isinstance(Predictor.load(tmp_path / "elo"), EloModel)
