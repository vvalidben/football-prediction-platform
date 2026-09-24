"""Explanations: exactly additive, faithful to the model, and refused when invalid."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from app.ml.dataset import HOME, Dataset, Example
from app.ml.ensemble import EnsembleModel, SimpleAverage
from app.ml.explain import (
    MISSING_SUFFIX,
    PROBABILITY,
    Contribution,
    ExplanationUnavailableError,
    TargetExplanation,
    _pipeline_parts,
    explain,
    importance,
    label_of,
    shap_available,
)
from app.ml.feature_sets import CORE_FEATURES
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

EXPLAINABLE: dict[str, Callable[[], Predictor]] = {
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


@pytest.fixture(scope="module", params=list(EXPLAINABLE))
def fitted(
    request: pytest.FixtureRequest, split: tuple[Dataset, Dataset]
) -> tuple[Predictor, Dataset]:
    train, validation = split
    return EXPLAINABLE[request.param]().fit(train), validation


def total(target: TargetExplanation) -> float:
    return target.baseline + sum(item.contribution for item in target.contributions)


class TestExactness:
    def test_the_contributions_add_up_to_the_model_output(
        self, fitted: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = fitted

        explanation = explain(model, validation, 3)

        assert explanation.targets
        for target in explanation.targets:
            assert total(target) == pytest.approx(target.predicted, abs=1e-9)

    def test_nothing_is_left_over_when_every_feature_is_listed(
        self, fitted: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = fitted

        explanation = explain(model, validation, 3)

        for target in explanation.targets:
            assert len(target.contributions) == explanation.features_used

    def test_the_explained_outcome_is_the_one_the_model_predicts(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)

        explanation = explain(model, validation, 7)

        predicted = model.predict_proba(validation)[7]
        explained = [target.predicted_output for target in explanation.targets]
        assert explained == pytest.approx(list(predicted), abs=1e-9)
        assert all(target.output_unit == PROBABILITY for target in explanation.targets)

    def test_the_poisson_explanation_reproduces_the_model_goal_rates(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = PoissonModel(CORE_FEATURES, alpha=1.0).fit(train)

        explanation = explain(model, validation, 2)

        home_rates, away_rates = model.goals.rates(validation.matrix(model.used_features))
        explained = [target.predicted_output for target in explanation.targets]
        assert explained == pytest.approx([home_rates[2], away_rates[2]], rel=1e-9)

    def test_the_elo_explanation_reproduces_the_model_probabilities(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = EloModel().fit(train)

        explanation = explain(model, validation, 4)

        explained = [target.predicted_output for target in explanation.targets]
        assert explained == pytest.approx(list(model.predict_proba(validation)[4]), abs=1e-9)
        assert all(len(target.contributions) == 1 for target in explanation.targets)

    def test_explaining_twice_gives_the_same_numbers(
        self, fitted: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = fitted

        first, second = explain(model, validation, 1), explain(model, validation, 1)

        assert [c.contribution for c in first.targets[0].contributions] == [
            c.contribution for c in second.targets[0].contributions
        ]


class TestDirection:
    """A contribution's sign follows the model's own arithmetic.

    It is not an intuition about football: `elo_diff` is highly correlated with the
    rating-based goal features, and a fitted model splits the credit between them,
    so a strong home side can still get a positive contribution on the away side's
    score. These tests compare against the model's coefficients instead.
    """

    def test_a_contribution_is_the_coefficient_times_the_standardised_value(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        """The guard against attaching a number to the wrong feature: every
        contribution is checked against the column sklearn itself names."""
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        imputer, scaler, estimator = _pipeline_parts(model.classifier)
        names = list(imputer.get_feature_names_out(model.used_features))
        row = validation.matrix(model.used_features)[6]
        z = scaler.transform(imputer.transform(row.reshape(1, -1)))[0]

        explanation = explain(model, validation, 6)

        for outcome, target in enumerate(explanation.targets):
            for item in target.contributions:
                column = names.index(item.feature)
                expected = estimator.coef_[outcome][column] * z[column]
                assert item.contribution == pytest.approx(expected, abs=1e-12), item.feature

    def test_a_value_above_the_training_average_moves_the_score_the_way_the_model_leans(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        imputer, _, estimator = _pipeline_parts(model.classifier)
        names = list(imputer.get_feature_names_out(model.used_features))
        coefficient = float(estimator.coef_[HOME][names.index("elo_diff")])
        values = [float(e.features["elo_diff"]) for e in validation.examples]

        high = _contribution(
            explain(model, validation, int(np.argmax(values))).targets[HOME], "elo_diff"
        )
        low = _contribution(
            explain(model, validation, int(np.argmin(values))).targets[HOME], "elo_diff"
        )

        assert np.sign(high) == np.sign(coefficient)
        assert np.sign(low) == -np.sign(coefficient)

    def test_changing_one_feature_changes_only_that_features_contribution(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        """Attribution has to be local: nudging one input must not silently move
        the credit given to the others."""
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        example = validation.examples[0]
        nudged = replace(
            example, features={**example.features, "elo_diff": example.features["elo_diff"] + 200}
        )
        nudged_data = Dataset((nudged,), validation.feature_set_version)

        before = _by_feature(explain(model, validation, 0).targets[HOME])
        after = _by_feature(explain(model, nudged_data, 0).targets[HOME])

        assert after["elo_diff"] != before["elo_diff"]
        for feature, contribution in before.items():
            if feature != "elo_diff":
                assert after[feature] == pytest.approx(contribution, abs=1e-12), feature

    def test_contributions_are_ordered_by_size(self, fitted: tuple[Predictor, Dataset]) -> None:
        model, validation = fitted

        contributions = explain(model, validation, 0).targets[0].contributions

        sizes = [abs(item.contribution) for item in contributions]
        assert sizes == sorted(sizes, reverse=True)


class TestMissingData:
    @pytest.fixture(scope="class")
    def patchy(self) -> tuple[Predictor, Dataset]:
        """Training data where one feature is sometimes missing, so the pipeline
        adds a missingness flag: real snapshots look like this."""
        data = make_dataset(seasons=4, teams=12)
        examples = tuple(
            replace(example, features={**example.features, "home_form5_goals_for": None})
            if index % 3 == 0
            else example
            for index, example in enumerate(data.examples)
        )
        parts = split_by_seasons(Dataset(examples, data.feature_set_version))
        return LogisticModel(CORE_FEATURES, c=0.1).fit(parts.train), parts.validation

    def test_a_missing_feature_is_reported_as_a_flag_the_model_actually_uses(
        self, patchy: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = patchy

        explanation = explain(model, validation, 0)

        flags = [
            c for c in explanation.targets[0].contributions if c.feature.endswith(MISSING_SUFFIX)
        ]
        assert [flag.feature for flag in flags] == [f"home_form5_goals_for{MISSING_SUFFIX}"]
        assert {flag.value for flag in flags} <= {0.0, 1.0}
        assert "not available" in label_of(flags[0].feature)

    def test_the_flag_is_set_exactly_when_the_feature_has_no_value(
        self, patchy: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = patchy
        flag = f"home_form5_goals_for{MISSING_SUFFIX}"

        for index in range(6):
            by_name = _by_contribution(explain(model, validation, index).targets[0])

            absent = validation.examples[index].features["home_form5_goals_for"] is None
            assert by_name[flag].value == (1.0 if absent else 0.0)
            assert (by_name["home_form5_goals_for"].value is None) == absent

    def test_a_feature_with_no_value_has_no_number_shown(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
        example = validation.examples[0]
        absent = [
            name
            for name in model.used_features
            if example.features.get(name) is None and not name.endswith(MISSING_SUFFIX)
        ]

        by_name = _by_contribution(explain(model, validation, 0).targets[0])

        for name in absent:
            assert by_name[name].value is None


class TestLabels:
    def test_features_are_named_as_the_registry_describes_them(self) -> None:
        assert label_of("elo_diff") == "Home Elo + home advantage - away Elo"
        assert label_of("home_form5_points").startswith("Home team")
        assert "not available" in label_of(f"home_form5_points{MISSING_SUFFIX}")

    def test_an_unknown_feature_still_gets_a_readable_name(self) -> None:
        assert label_of("some_new_feature") == "Some new feature"


class TestImportance:
    def test_importance_is_a_normalised_ranking(self, fitted: tuple[Predictor, Dataset]) -> None:
        model, _ = fitted

        report = importance(model)

        values = [entry.importance for entry in report.entries]
        assert values == sorted(values, reverse=True)
        assert sum(values) == pytest.approx(1.0)
        assert all(value >= 0 for value in values)

    def test_the_informative_feature_outranks_the_noise_feature(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, _ = split
        model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)

        ranking = [entry.feature for entry in importance(model).entries]

        # `home_form5_points` is generated as pure noise by the synthetic data.
        assert ranking.index("elo_diff") < ranking.index("home_form5_points")


class TestRefusals:
    """An unexplainable model says so; it never returns a plausible-looking guess."""

    def test_the_historical_baseline_is_refused_with_its_reason(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        model = HistoricalBaseline().fit(train)

        with pytest.raises(ExplanationUnavailableError, match="reads no features"):
            explain(model, validation, 0)

    def test_an_ensemble_is_refused_because_the_mixture_is_not_additive(
        self, split: tuple[Dataset, Dataset]
    ) -> None:
        train, validation = split
        members = {"elo": EloModel().fit(train), "poisson": PoissonModel(CORE_FEATURES).fit(train)}
        combiner = SimpleAverage(tuple(members)).fit(
            {name: member.predict(validation) for name, member in members.items()}, validation
        )
        model = EnsembleModel(members, combiner)

        with pytest.raises(ExplanationUnavailableError, match="ensemble"):
            explain(model, validation, 0)

    def test_an_unfitted_model_cannot_be_explained(self, split: tuple[Dataset, Dataset]) -> None:
        _, validation = split

        with pytest.raises(NotFittedError):
            explain(LogisticModel(CORE_FEATURES), validation, 0)
        with pytest.raises(NotFittedError):
            importance(LogisticModel(CORE_FEATURES))

    def test_an_index_outside_the_dataset_is_rejected(
        self, fitted: tuple[Predictor, Dataset]
    ) -> None:
        model, validation = fitted

        with pytest.raises(IndexError):
            explain(model, validation, len(validation))


@pytest.mark.skipif(not shap_available(), reason="TreeSHAP needs the shap package")
class TestTreeShap:
    @pytest.fixture(scope="class")
    def tree_models(self, split: tuple[Dataset, Dataset]) -> dict[str, Predictor]:
        from app.ml.trees import LightGBMModel, RandomForestModel, XGBoostModel

        train, _ = split
        return {
            "random_forest": RandomForestModel(CORE_FEATURES, n_estimators=40).fit(train),
            "xgboost": XGBoostModel(CORE_FEATURES, n_estimators=40).fit(train),
            "lightgbm": LightGBMModel(CORE_FEATURES, n_estimators=40).fit(train),
        }

    def test_tree_contributions_add_up_to_the_classifier_output(
        self, tree_models: dict[str, Predictor], split: tuple[Dataset, Dataset]
    ) -> None:
        _, validation = split

        for name, model in tree_models.items():
            explanation = explain(model, validation, 5)

            assert explanation.method == "TreeSHAP", name
            for target in explanation.targets:
                assert total(target) == pytest.approx(target.predicted, abs=1e-6), name

    def test_tree_explanations_reproduce_the_predicted_probabilities(
        self, tree_models: dict[str, Predictor], split: tuple[Dataset, Dataset]
    ) -> None:
        _, validation = split

        for name, model in tree_models.items():
            explained = [t.predicted_output for t in explain(model, validation, 5).targets]

            assert explained == pytest.approx(list(model.predict_proba(validation)[5]), abs=1e-6), (
                name
            )

    def test_tree_importance_is_the_library_measure(
        self, tree_models: dict[str, Predictor]
    ) -> None:
        report = importance(tree_models["random_forest"])

        assert report.method == "impurity importance"
        assert sum(entry.importance for entry in report.entries) == pytest.approx(1.0)


def test_a_tree_model_says_so_when_shap_is_missing(
    split: tuple[Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serving image has no shap: the API must say that, not approximate."""
    from app.ml import explain as explain_module
    from app.ml.trees import RandomForestModel

    train, validation = split
    model = RandomForestModel(CORE_FEATURES, n_estimators=10).fit(train)
    monkeypatch.setattr(explain_module, "shap_available", lambda: False)

    with pytest.raises(ExplanationUnavailableError, match="shap package is not installed"):
        explain(model, validation, 0)


def test_an_all_missing_feature_is_dropped_before_it_reaches_the_explanation() -> None:
    """Features never observed in training are not in `used_features`, so they
    cannot appear as contributions."""
    data = make_dataset(seasons=3, teams=10)
    train = Dataset(data.examples[:180], data.feature_set_version)
    model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)

    explanation = explain(model, data, 200)

    named = {c.feature for c in explanation.targets[0].contributions}
    assert "odds_latest_home_prob" not in named  # never set by the synthetic data
    assert named <= set(model.used_features) | {
        f"{name}{MISSING_SUFFIX}" for name in model.used_features
    }


def test_an_example_with_no_features_at_all_is_still_explained() -> None:
    """A snapshot where everything is missing is a real case; the model falls back
    on the training medians and the explanation has to hold together."""
    data = make_dataset(seasons=3, teams=10)
    train = Dataset(data.examples[:180], data.feature_set_version)
    model = LogisticModel(CORE_FEATURES, c=0.1).fit(train)
    blank = Example(
        match_id=99_999,
        kickoff_at=data.examples[-1].kickoff_at,
        season_id=1,
        competition_id=1,
        features=dict.fromkeys(CORE_FEATURES),
        home_goals=0,
        away_goals=0,
    )
    blank_data = Dataset((blank,), data.feature_set_version)

    explanation = explain(model, blank_data, 0)

    assert all(
        total(target) == pytest.approx(target.predicted, abs=1e-9) for target in explanation.targets
    )
    assert np.isclose(
        sum(float(target.predicted_output or 0.0) for target in explanation.targets), 1.0
    )


def _by_feature(target: TargetExplanation) -> dict[str, float]:
    return {item.feature: item.contribution for item in target.contributions}


def _by_contribution(target: TargetExplanation) -> dict[str, Contribution]:
    return {item.feature: item for item in target.contributions}


def _contribution(target: TargetExplanation, feature: str) -> float:
    for item in target.contributions:
        if item.feature == feature:
            return item.contribution
    raise AssertionError(f"{feature} is not among the contributions")
