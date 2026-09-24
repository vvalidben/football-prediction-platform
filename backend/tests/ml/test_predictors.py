"""Training and prediction for each baseline model."""

from collections.abc import Callable

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
from app.ml.splits import split_by_seasons
from tests.ml.synthetic import make_dataset

MODELS: dict[str, Callable[[], Predictor]] = {
    "historical_baseline": HistoricalBaseline,
    "elo": EloModel,
    "logistic_regression": lambda: LogisticModel(CORE_FEATURES, c=0.1),
    "poisson": lambda: PoissonModel(CORE_FEATURES, alpha=1.0),
}


@pytest.fixture(scope="module")
def data() -> Dataset:
    return make_dataset(seasons=4, teams=12)


@pytest.fixture(scope="module", params=list(MODELS))
def fitted(request: pytest.FixtureRequest, data: Dataset) -> tuple[Predictor, Dataset]:
    split = split_by_seasons(data)
    return MODELS[request.param]().fit(split.train), split.validation


def test_every_prediction_is_a_normalised_consistent_distribution(
    fitted: tuple[Predictor, Dataset],
) -> None:
    model, validation = fitted

    predictions = model.predict(validation)

    assert len(predictions) == len(validation)
    for p in predictions:
        assert p.grid.sum() == pytest.approx(1.0, abs=1e-9)
        assert (p.grid >= 0).all()
        assert sum(p.outcomes) == pytest.approx(1.0, abs=1e-9)
        assert all(0 < q < 1 for q in p.outcomes)
        assert 0 < p.expected_home_goals < 6
        assert 0 < p.expected_away_goals < 6


def test_predictions_are_deterministic(fitted: tuple[Predictor, Dataset]) -> None:
    model, validation = fitted

    first, second = model.predict(validation), model.predict(validation)

    assert all(np.array_equal(a.grid, b.grid) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("name", ["elo", "logistic_regression", "poisson"])
def test_feature_models_beat_the_historical_baseline(data: Dataset, name: str) -> None:
    split = split_by_seasons(data)
    baseline = HistoricalBaseline().fit(split.train)
    model = MODELS[name]().fit(split.train)

    base = evaluate(baseline.predict(split.validation), split.validation)
    score = evaluate(model.predict(split.validation), split.validation)

    assert score["log_loss"] < base["log_loss"]
    assert score["total_goals_mae"] < base["total_goals_mae"] + 0.05


def test_elo_model_learns_that_a_bigger_gap_favours_home(data: Dataset) -> None:
    model = EloModel()
    model.fit(data)

    assert model.slope > 0
    assert model.cutpoints[0] < model.cutpoints[1]


def test_historical_baseline_matches_training_frequencies(data: Dataset) -> None:
    model = HistoricalBaseline(smoothing=0.0).fit(data)
    (prediction,) = model.predict(data.where(lambda e: e.match_id == 0))

    outcomes = data.outcomes
    assert prediction.home == pytest.approx(np.mean(outcomes == 0))
    assert prediction.draw == pytest.approx(np.mean(outcomes == 1))
    assert prediction.expected_home_goals == pytest.approx(np.minimum(data.home_goals, 10).mean())


def test_poisson_model_fits_a_dixon_coles_correction(data: Dataset) -> None:
    with_dc, without = PoissonModel(CORE_FEATURES), PoissonModel(CORE_FEATURES, dixon_coles=False)
    with_dc.fit(data)
    without.fit(data)

    assert -0.3 <= with_dc.rho <= 0.3
    assert without.rho == 0.0


def test_features_never_observed_in_training_are_dropped(data: Dataset) -> None:
    model = LogisticModel(CORE_FEATURES).fit(data)

    assert "elo_diff" in model.used_features
    assert "odds_bookmakers" not in model.used_features  # not in the core set at all
    assert "h2h_weighted_points" not in model.used_features  # always None in this data


def test_predicting_before_fitting_fails() -> None:
    with pytest.raises(NotFittedError):
        EloModel().predict(make_dataset(seasons=1, teams=4))


def test_fitting_on_nothing_fails() -> None:
    empty = make_dataset(seasons=1, teams=4).where(lambda e: False)

    with pytest.raises(ValueError, match="empty"):
        HistoricalBaseline().fit(empty)


def test_unbounded_history_counts_are_not_model_inputs() -> None:
    """Counts that only grow over time would make later seasons out-of-range."""
    for name in ("home_rated_matches", "away_rated_matches", "competition_matches"):
        assert name not in CORE_FEATURES
    assert "competition_id" not in CORE_FEATURES
    assert not any(name.startswith("odds_") for name in CORE_FEATURES)
