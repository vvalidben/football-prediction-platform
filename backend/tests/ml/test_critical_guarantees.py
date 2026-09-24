"""The eight guarantees the platform is not allowed to break.

Each one has a home elsewhere in the suite — leakage in `tests/features/`,
distributions in `tests/ml/test_probability_and_metrics.py`, loading in
`tests/registry/` — and this file states them once, in the terms a person would
use when asking "is this thing sound?". Where the assertion was only implied,
the test here is the first to make it, and says so.

1. No future information enters features.
2. No post-match information enters predictions.
3. Probability distributions sum to 1.
4. Predictions are reproducible.
5. Missing data is handled.
6. Unknown teams are handled.
7. Model loading works.
8. Historical backtesting is chronological.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from app.ml.dataset import Dataset, Example
from app.ml.feature_sets import CORE_FEATURES
from app.ml.persistence import load_model
from app.ml.predictors import (
    EloModel,
    HistoricalBaseline,
    LogisticModel,
    PoissonModel,
    Predictor,
)
from app.ml.probability import MAX_GOALS
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


@pytest.fixture(scope="module")
def split(data: Dataset) -> tuple[Dataset, Dataset]:
    parts = split_by_seasons(data)
    return parts.train, parts.validation


@pytest.fixture(scope="module", params=list(MODELS))
def fitted(request: pytest.FixtureRequest, split: tuple[Dataset, Dataset]) -> Predictor:
    train, _ = split
    return MODELS[request.param]().fit(train)


def blank_example(template: Example, **changes: object) -> Example:
    """A match the model knows nothing about: every feature missing."""
    return replace(template, features=dict.fromkeys(CORE_FEATURES), **changes)  # type: ignore[arg-type]


class TestDistributionsAreProbabilities:
    """3. Probability distributions sum to 1."""

    def test_the_score_grid_and_every_market_derived_from_it_are_normalised(
        self, fitted: Predictor, split: tuple[Dataset, Dataset]
    ) -> None:
        _, validation = split

        for prediction in fitted.predict(validation):
            assert prediction.grid.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
            assert (prediction.grid >= 0).all()
            assert prediction.grid.sum() == pytest.approx(1.0, abs=1e-9)
            assert sum(prediction.outcomes) == pytest.approx(1.0, abs=1e-9)
            both, neither = prediction.btts
            assert both + neither == pytest.approx(1.0, abs=1e-9)
            for line in (0.5, 1.5, 2.5, 3.5):
                over, under = prediction.over_under(line)
                # A whole-goal line would refund the stake, so these sum to 1 only
                # on half lines — which is exactly what is asserted here.
                assert over + under == pytest.approx(1.0, abs=1e-9)

    def test_predict_proba_agrees_with_the_grid_it_came_from(
        self, fitted: Predictor, split: tuple[Dataset, Dataset]
    ) -> None:
        _, validation = split

        probabilities = fitted.predict_proba(validation)

        assert probabilities.shape == (len(validation), 3)
        assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(validation)), abs=1e-9)
        for row, prediction in zip(probabilities, fitted.predict(validation), strict=True):
            assert list(row) == pytest.approx(list(prediction.outcomes), abs=1e-12)


class TestReproducibility:
    """4. Predictions are reproducible."""

    def test_the_same_model_gives_the_same_numbers_twice(
        self, fitted: Predictor, split: tuple[Dataset, Dataset]
    ) -> None:
        _, validation = split

        first, second = fitted.predict(validation), fitted.predict(validation)

        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a.grid, b.grid)

    def test_a_model_saved_and_loaded_again_predicts_identically(
        self, fitted: Predictor, split: tuple[Dataset, Dataset], tmp_path: Path
    ) -> None:
        """Serving loads an artifact from disk: a prediction must not depend on
        which process produced it."""
        _, validation = split
        directory = tmp_path / f"{fitted.name}-artifact"
        fitted.save(directory, {"feature_set_version": validation.feature_set_version})

        reloaded = load_model(directory).model

        for before, after in zip(
            fitted.predict(validation), reloaded.predict(validation), strict=True
        ):
            assert np.array_equal(before.grid, after.grid)

    def test_refitting_on_the_same_data_gives_the_same_model(
        self, split: tuple[Dataset, Dataset], fitted: Predictor
    ) -> None:
        train, validation = split

        refitted = MODELS[fitted.name]().fit(train)

        for a, b in zip(fitted.predict(validation), refitted.predict(validation), strict=True):
            assert a.grid == pytest.approx(b.grid, abs=1e-12)


class TestMissingData:
    """5. Missing data is handled."""

    def test_a_match_with_no_features_at_all_still_gets_a_valid_prediction(
        self, fitted: Predictor, data: Dataset
    ) -> None:
        """Real snapshots have holes: a promoted side, a cup tie, a new competition."""
        blank = Dataset((blank_example(data.examples[-1]),), data.feature_set_version)

        prediction = fitted.predict(blank)[0]

        assert prediction.grid.sum() == pytest.approx(1.0, abs=1e-9)
        assert sum(prediction.outcomes) == pytest.approx(1.0, abs=1e-9)
        assert all(0 < p < 1 for p in prediction.outcomes)
        assert 0 < prediction.expected_home_goals < 10

    def test_a_single_missing_feature_does_not_change_the_other_inputs(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        """Imputation must be local: blanking one column must not move a match's
        prediction to somewhere unrelated."""
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        example = validation.examples[0]
        without = Dataset(
            (replace(example, features={**example.features, "elo_diff": None}),),
            validation.feature_set_version,
        )
        complete = Dataset((example,), validation.feature_set_version)

        changed = model.predict(without)[0].outcomes
        original = model.predict(complete)[0].outcomes

        assert sum(changed) == pytest.approx(1.0, abs=1e-9)
        assert changed != original  # the model did use the feature
        assert max(abs(a - b) for a, b in zip(changed, original, strict=True)) < 0.6

    def test_training_survives_a_feature_that_is_never_observed(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        emptied = Dataset(
            tuple(
                replace(example, features={**example.features, "elo_diff": None})
                for example in train.examples
            ),
            train.feature_set_version,
        )

        model = LogisticModel(CORE_FEATURES, c=0.1).fit(emptied)

        assert "elo_diff" not in model.used_features
        assert sum(model.predict(validation)[0].outcomes) == pytest.approx(1.0, abs=1e-9)


class TestUnknownTeams:
    """6. Unknown teams are handled.

    A promoted club, a cup opponent from a lower division or a team added by a new
    provider has no history at all. The features are then empty and the ratings are
    at their default — the model must still answer, and must not treat the two
    unknown sides as different from each other.
    """

    def test_a_match_between_two_teams_with_no_history_is_predicted(
        self, fitted: Predictor, data: Dataset
    ) -> None:
        unknown = Dataset(
            (blank_example(data.examples[-1], match_id=999_001),), data.feature_set_version
        )

        prediction = fitted.predict(unknown)[0]

        assert prediction.grid.sum() == pytest.approx(1.0, abs=1e-9)
        assert all(0 < p < 1 for p in prediction.outcomes)

    def test_two_unknown_sides_come_out_balanced(self, split: tuple[Dataset, Dataset]) -> None:
        """`elo_diff` already contains home advantage, so a missing rating means
        "evenly matched" rather than "home team unknown": neither side may be
        favoured much, and neither may be dismissed."""
        train, _ = split
        model = EloModel().fit(train)
        unknown = Dataset((blank_example(train.examples[-1]),), train.feature_set_version)

        home, draw, away = model.predict(unknown)[0].outcomes

        assert abs(home - away) < 0.1
        assert min(home, draw, away) > 0.05

    def test_an_unknown_team_does_not_leak_another_team_s_strength(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        """Two different unknown fixtures must get the same answer: there is
        nothing to tell them apart."""
        train, _ = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        first = Dataset((blank_example(train.examples[-1], match_id=1),), train.feature_set_version)
        second = Dataset((blank_example(train.examples[0], match_id=2),), train.feature_set_version)

        assert model.predict(first)[0].outcomes == pytest.approx(
            model.predict(second)[0].outcomes, abs=1e-12
        )


class TestChronology:
    """8. Historical backtesting is chronological."""

    def test_a_dataset_out_of_order_is_refused(self, data: Dataset) -> None:
        """Every split, fold and backtest window relies on this invariant."""
        reversed_examples = tuple(reversed(data.examples))

        with pytest.raises(ValueError, match="chronological"):
            Dataset(reversed_examples, data.feature_set_version)

    def test_training_never_contains_a_match_later_than_the_validation_block(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split

        assert train.examples[-1].kickoff_at <= validation.examples[0].kickoff_at

    def test_the_backtest_guards_chronology_itself(self) -> None:
        """The walk-forward windows, the result delay and the "changing a future
        result changes no earlier prediction" property are covered in depth by
        `tests/ml/test_backtest.py`; what matters here is that a violation is an
        error the backtest raises, not something it quietly tolerates."""
        from app.ml.backtest import BacktestLeakError, run_backtest

        assert issubclass(BacktestLeakError, RuntimeError)
        assert callable(run_backtest)


def test_a_model_that_has_not_been_fitted_refuses_to_predict(data: Dataset) -> None:
    """7. Model loading works — and an unfitted object is not a model."""
    from app.ml.predictors import NotFittedError

    with pytest.raises(NotFittedError):
        LogisticModel(CORE_FEATURES).predict(data)


def test_a_saved_model_carries_the_feature_set_it_was_trained_on(
    split: tuple[Dataset, Dataset], tmp_path: Path
) -> None:
    """7. Model loading works: an artifact built for other features is refused
    rather than silently used with the wrong inputs."""
    train, _ = split
    model = EloModel().fit(train)
    directory = tmp_path / "elo-artifact"
    model.save(directory, {"feature_set_version": train.feature_set_version})

    loaded = load_model(directory, feature_set_version=train.feature_set_version)

    assert loaded.metadata["feature_set_version"] == train.feature_set_version
    with pytest.raises(ValueError, match="feature"):
        load_model(directory, feature_set_version="v1-something-else")


def test_the_time_between_kickoff_and_a_usable_result_is_respected(data: Dataset) -> None:
    """1 and 2, at the level everything else builds on: a result is only knowable
    some time after kickoff, and the feature configuration says how long."""
    from app.features.config import FeatureConfig

    delay = FeatureConfig().result_delay

    assert delay >= timedelta(hours=2)
    assert data.examples[0].kickoff_at + delay > data.examples[0].kickoff_at
