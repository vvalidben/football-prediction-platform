"""Mathematical consistency of the goal prediction engine.

`assert_consistent` recomputes every field of a prediction with plain loops over the
121 exact scores, independently of the vectorised code under test, and checks the
identities that must hold between fields. It is applied to property-based grids
(hypothesis) of every kind the models produce, to hand-built edge cases and to the
predictions of every model.
"""

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import poisson, skellam

from app.ml.dataset import Dataset
from app.ml.engine import GoalPredictionEngine, ScorePrediction
from app.ml.feature_sets import CORE_FEATURES
from app.ml.metrics import evaluate
from app.ml.predictors import (
    EloModel,
    HistoricalBaseline,
    LogisticModel,
    PoissonModel,
    Predictor,
)
from app.ml.probability import (
    DEFAULT_LINES,
    MAX_GOALS,
    MAX_TOTAL,
    Grid,
    MatchPrediction,
    over_under,
    poisson_grid,
    reconcile,
    total_goals_distribution,
)
from app.ml.splits import ChronologicalSplit, split_by_seasons
from app.ml.trees import LightGBMModel, RandomForestModel, XGBoostModel
from tests.ml.synthetic import make_dataset

SIZE = MAX_GOALS + 1
TOL = 1e-9
ALL_LINES = tuple(k + 0.5 for k in range(MAX_TOTAL))  # 0.5 .. 19.5
SCORES = [(h, a) for h in range(SIZE) for a in range(SIZE)]

# Strategies ----------------------------------------------------------------------------

rates = st.floats(min_value=0.05, max_value=4.0, allow_nan=False)
rhos = st.floats(min_value=-0.2, max_value=0.2, allow_nan=False)


@st.composite
def poisson_grids(draw: st.DrawFn) -> Grid:
    return poisson_grid(draw(rates), draw(rates), draw(rhos))


@st.composite
def reconciled_grids(draw: st.DrawFn) -> Grid:
    """What classifier-based models produce: a Poisson grid rescaled to a 1X2."""
    home, drawn, away = (draw(st.floats(0.01, 1.0)) for _ in range(3))
    total = home + drawn + away
    target = (home / total, drawn / total, away / total)
    return reconcile(poisson_grid(draw(rates), draw(rates)), target)


@st.composite
def arbitrary_grids(draw: st.DrawFn) -> Grid:
    """Any non-negative matrix with mass, normalised: no structure assumed."""
    cells = draw(
        arrays(np.float64, (SIZE, SIZE), elements=st.floats(0.0, 1.0, allow_subnormal=False))
    )
    if cells.sum() <= 1e-6:
        cells[draw(st.integers(0, MAX_GOALS)), draw(st.integers(0, MAX_GOALS))] = 1.0
    normalised: Grid = cells / cells.sum()
    return normalised


@st.composite
def sparse_grids(draw: st.DrawFn) -> Grid:
    """A few non-zero cells: stresses boundaries (lines, zero rows/columns)."""
    grid = np.zeros((SIZE, SIZE))
    cells = draw(
        st.lists(
            st.tuples(st.integers(0, MAX_GOALS), st.integers(0, MAX_GOALS), st.floats(0.01, 1)),
            min_size=1,
            max_size=5,
        )
    )
    for h, a, w in cells:
        grid[h, a] += w
    return grid / grid.sum()


any_grid = st.one_of(poisson_grids(), reconciled_grids(), arbitrary_grids(), sparse_grids())

# The independent checker ---------------------------------------------------------------


def assert_consistent(prediction: ScorePrediction, tol: float = TOL) -> None:
    m = prediction.score_probability_matrix
    p = {(h, a): float(m[h, a]) for h, a in SCORES}

    # The matrix is a probability distribution over 0-0 .. 10-10.
    assert m.shape == (SIZE, SIZE) and prediction.max_goals == MAX_GOALS
    assert all(v >= 0 for v in p.values())
    assert math.isclose(sum(p.values()), 1.0, abs_tol=tol)

    # 1X2: the three regions of the matrix, summing to 1.
    home = sum(v for (h, a), v in p.items() if h > a)
    draw = sum(v for (h, a), v in p.items() if h == a)
    away = sum(v for (h, a), v in p.items() if h < a)
    assert prediction.home_win_probability == pytest.approx(home, abs=tol)
    assert prediction.draw_probability == pytest.approx(draw, abs=tol)
    assert prediction.away_win_probability == pytest.approx(away, abs=tol)
    total_1x2 = (
        prediction.home_win_probability
        + prediction.draw_probability
        + prediction.away_win_probability
    )
    assert total_1x2 == pytest.approx(1.0, abs=tol)
    assert prediction.draw_probability >= p[0, 0] - tol  # 0-0 is a draw

    # Expected goals: the marginal means, inside the representable range.
    assert prediction.expected_home_goals == pytest.approx(
        sum(h * v for (h, _), v in p.items()), abs=tol
    )
    assert prediction.expected_away_goals == pytest.approx(
        sum(a * v for (_, a), v in p.items()), abs=tol
    )
    assert 0 <= prediction.expected_home_goals <= MAX_GOALS
    assert 0 <= prediction.expected_away_goals <= MAX_GOALS

    # Most probable score: a maximal cell, with its own probability.
    best = prediction.most_probable_score
    assert best.probability == p[best.home_goals, best.away_goals]
    assert best.probability >= max(p.values())
    assert best.probability > 0

    # Over/under: complementary, from the same cells, monotone in the line.
    previous_over = 1.0 + tol
    for entry in prediction.over_under_probabilities:
        over = sum(v for (h, a), v in p.items() if h + a > entry.line)
        under = sum(v for (h, a), v in p.items() if h + a < entry.line)
        assert entry.over == pytest.approx(over, abs=tol)
        assert entry.under == pytest.approx(under, abs=tol)
        assert entry.over + entry.under == pytest.approx(1.0, abs=tol)
        assert 0 - tol <= entry.over <= 1 + tol
        assert entry.over <= previous_over + tol  # a higher line is never more likely
        previous_over = entry.over
    if 0.5 in prediction.lines:
        assert prediction.over_under(0.5).under == pytest.approx(p[0, 0], abs=tol)
        # A home or away win needs at least one goal.
        assert prediction.over_under(0.5).over >= max(home, away) - tol
    if 1.5 in prediction.lines:
        assert prediction.over_under(1.5).under == pytest.approx(
            p[0, 0] + p[1, 0] + p[0, 1], abs=tol
        )

    # Both teams to score: complementary; both scoring means at least 2 goals.
    yes = sum(v for (h, a), v in p.items() if h > 0 and a > 0)
    no = sum(v for (h, a), v in p.items() if h == 0 or a == 0)
    assert prediction.btts_probability.yes == pytest.approx(yes, abs=tol)
    assert prediction.btts_probability.no == pytest.approx(no, abs=tol)
    assert prediction.btts_probability.yes + prediction.btts_probability.no == pytest.approx(
        1.0, abs=tol
    )
    assert prediction.btts_probability.no >= p[0, 0] - tol
    if 1.5 in prediction.lines:
        assert prediction.btts_probability.yes <= prediction.over_under(1.5).over + tol


# Property-based consistency ------------------------------------------------------------

PROPERTY = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@PROPERTY
@given(any_grid)
def test_every_derived_probability_is_consistent_with_the_matrix(grid: Grid) -> None:
    assert_consistent(ScorePrediction.from_grid(grid, ALL_LINES))


@PROPERTY
@given(any_grid)
def test_total_goals_distribution_matches_the_matrix(grid: Grid) -> None:
    totals = total_goals_distribution(grid)

    assert totals.shape == (MAX_TOTAL + 1,)
    assert totals.sum() == pytest.approx(1.0, abs=TOL)
    for k in range(MAX_TOTAL + 1):
        cells = sum(grid[h, a] for h, a in SCORES if h + a == k)
        assert totals[k] == pytest.approx(cells, abs=TOL)
    for line in ALL_LINES:
        # total > line  <=>  total >= line + 0.5 for a half-goal line
        assert over_under(grid, line)[0] == pytest.approx(totals[int(line + 0.5) :].sum(), abs=TOL)
    expected_total = float((np.arange(MAX_TOTAL + 1) * totals).sum())
    prediction = ScorePrediction.from_grid(grid)
    assert expected_total == pytest.approx(
        prediction.expected_home_goals + prediction.expected_away_goals, abs=TOL
    )


@PROPERTY
@given(any_grid)
def test_swapping_home_and_away_mirrors_the_prediction(grid: Grid) -> None:
    original = ScorePrediction.from_grid(grid)
    mirrored = ScorePrediction.from_grid(grid.T)

    assert mirrored.home_win_probability == pytest.approx(original.away_win_probability, abs=TOL)
    assert mirrored.away_win_probability == pytest.approx(original.home_win_probability, abs=TOL)
    assert mirrored.draw_probability == pytest.approx(original.draw_probability, abs=TOL)
    assert mirrored.expected_home_goals == pytest.approx(original.expected_away_goals, abs=TOL)
    for a, b in zip(
        original.over_under_probabilities, mirrored.over_under_probabilities, strict=True
    ):
        assert (a.over, a.under) == pytest.approx((b.over, b.under), abs=TOL)
    assert mirrored.btts_probability.yes == pytest.approx(original.btts_probability.yes, abs=TOL)
    best, swapped = original.most_probable_score, mirrored.most_probable_score
    assert swapped.probability == best.probability


@PROPERTY
@given(reconciled_grids())
def test_reconciled_grids_stay_consistent(grid: Grid) -> None:
    assert_consistent(ScorePrediction.from_grid(grid))


# Closed forms for independent Poisson goals -------------------------------------------

POISSON = settings(max_examples=150, deadline=None)
# Truncation at 10 goals per side moves probabilities by less than this for rates <= 3.
TRUNCATION = 2e-3


@POISSON
@given(st.floats(0.1, 3.0), st.floats(0.1, 3.0))
def test_poisson_predictions_match_closed_forms(home_rate: float, away_rate: float) -> None:
    prediction = ScorePrediction.from_rates(home_rate, away_rate)
    total = home_rate + away_rate

    assert prediction.expected_home_goals == pytest.approx(home_rate, abs=5 * TRUNCATION)
    assert prediction.expected_away_goals == pytest.approx(away_rate, abs=5 * TRUNCATION)
    # Home minus away goals is Skellam-distributed.
    assert prediction.draw_probability == pytest.approx(
        skellam.pmf(0, home_rate, away_rate), abs=TRUNCATION
    )
    assert prediction.home_win_probability == pytest.approx(
        skellam.sf(0, home_rate, away_rate), abs=TRUNCATION
    )
    # Total goals is Poisson(home + away).
    for line in DEFAULT_LINES:
        assert prediction.over_under(line).over == pytest.approx(
            poisson.sf(math.floor(line), total), abs=TRUNCATION
        )
    btts = (1 - math.exp(-home_rate)) * (1 - math.exp(-away_rate))
    assert prediction.btts_probability.yes == pytest.approx(btts, abs=TRUNCATION)


def test_truncation_at_ten_goals_is_negligible_for_realistic_rates() -> None:
    """Probability of scores beyond MAX_GOALS before renormalisation (documented)."""
    for rate, bound in ((1.5, 1e-6), (2.5, 1e-4), (3.0, 5e-4), (3.5, 2e-3)):
        assert poisson.sf(MAX_GOALS, rate) < bound


@POISSON
@given(rates, rates, st.floats(-0.2, -0.01))
def test_dixon_coles_moves_mass_to_low_draws_and_stays_consistent(
    home_rate: float, away_rate: float, rho: float
) -> None:
    corrected = ScorePrediction.from_rates(home_rate, away_rate, rho)
    independent = ScorePrediction.from_rates(home_rate, away_rate)

    assert_consistent(corrected)
    m, base = corrected.score_probability_matrix, independent.score_probability_matrix
    assert m[1, 1] / m[2, 2] > base[1, 1] / base[2, 2]  # 1-1 up, relative to untouched cells
    assert m[1, 0] / m[2, 0] < base[1, 0] / base[2, 0]  # 1-0 down


# Hand-built cases -----------------------------------------------------------------------


def point_mass(home: int, away: int) -> Grid:
    grid = np.zeros((SIZE, SIZE))
    grid[home, away] = 1.0
    return grid


@pytest.mark.parametrize(
    ("score", "outcome", "over", "btts"),
    [
        ((0, 0), "draw", {0.5: 0, 1.5: 0, 2.5: 0, 3.5: 0}, 0),
        ((1, 0), "home", {0.5: 1, 1.5: 0, 2.5: 0, 3.5: 0}, 0),
        ((0, 2), "away", {0.5: 1, 1.5: 1, 2.5: 0, 3.5: 0}, 0),
        ((1, 1), "draw", {0.5: 1, 1.5: 1, 2.5: 0, 3.5: 0}, 1),
        ((2, 1), "home", {0.5: 1, 1.5: 1, 2.5: 1, 3.5: 0}, 1),
        ((2, 2), "draw", {0.5: 1, 1.5: 1, 2.5: 1, 3.5: 1}, 1),
        ((10, 10), "draw", {0.5: 1, 1.5: 1, 2.5: 1, 3.5: 1}, 1),
        ((10, 0), "home", {0.5: 1, 1.5: 1, 2.5: 1, 3.5: 1}, 0),
    ],
)
def test_a_certain_score_gives_certain_markets(
    score: tuple[int, int], outcome: str, over: dict[float, int], btts: int
) -> None:
    prediction = ScorePrediction.from_grid(point_mass(*score))

    assert (
        prediction.home_win_probability,
        prediction.draw_probability,
        prediction.away_win_probability,
    ) == (float(outcome == "home"), float(outcome == "draw"), float(outcome == "away"))
    assert (prediction.expected_home_goals, prediction.expected_away_goals) == score
    assert (
        prediction.most_probable_score.home_goals,
        prediction.most_probable_score.away_goals,
    ) == score
    for line, value in over.items():
        assert prediction.over_under(line).over == value
        assert prediction.over_under(line).under == 1 - value
    assert prediction.btts_probability.yes == btts
    assert prediction.btts_probability.no == 1 - btts
    assert_consistent(prediction)


def test_uniform_grid_ties_resolve_to_the_fewest_goals() -> None:
    prediction = ScorePrediction.from_grid(np.full((SIZE, SIZE), 1 / SIZE**2))

    best = prediction.most_probable_score
    assert (best.home_goals, best.away_goals) == (0, 0)
    assert prediction.draw_probability == pytest.approx(SIZE / SIZE**2)
    assert prediction.home_win_probability == pytest.approx(prediction.away_win_probability)
    assert prediction.expected_home_goals == pytest.approx(MAX_GOALS / 2)


def test_most_probable_score_follows_the_favourite() -> None:
    assert ScorePrediction.from_rates(0.3, 0.3).most_probable_score.home_goals == 0
    strong_home = ScorePrediction.from_rates(3.0, 0.4).most_probable_score
    assert strong_home.home_goals > strong_home.away_goals
    strong_away = ScorePrediction.from_rates(0.4, 3.0).most_probable_score
    assert strong_away.away_goals > strong_away.home_goals


def test_top_scores_are_ordered_and_start_with_the_most_probable() -> None:
    prediction = ScorePrediction.from_rates(1.6, 1.1)

    top = prediction.top_scores(5)

    assert [(s.home_goals, s.away_goals) for s in top[:1]] == [
        (prediction.most_probable_score.home_goals, prediction.most_probable_score.away_goals)
    ]
    assert [s.probability for s in top] == sorted((s.probability for s in top), reverse=True)


# Validation ------------------------------------------------------------------------------


def negative_cell() -> Grid:
    """Sums to 1 but has a negative probability."""
    grid = poisson_grid(1.4, 1.1)
    grid[3, 3] -= 0.05
    grid[0, 0] += 0.05
    return grid


@pytest.mark.parametrize(
    "grid",
    [
        np.full((SIZE - 1, SIZE - 1), 1 / (SIZE - 1) ** 2),  # wrong shape
        np.full((SIZE, SIZE), 2 / SIZE**2),  # sums to 2
        negative_cell(),
        np.full((SIZE, SIZE), np.nan),
        np.zeros((SIZE, SIZE)),
    ],
    ids=["shape", "sum", "negative", "nan", "empty"],
)
def test_invalid_matrices_are_rejected(grid: Grid) -> None:
    with pytest.raises(ValueError, match="score grid"):
        ScorePrediction.from_grid(grid)


@pytest.mark.parametrize("line", [0.0, 2.0, -0.5, 20.5, 1.25, math.nan, math.inf])
def test_only_half_goal_lines_inside_the_range_are_accepted(line: float) -> None:
    with pytest.raises(ValueError, match="half-goal line"):
        ScorePrediction.from_rates(1.4, 1.1, lines=(line,))


def test_rates_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScorePrediction.from_rates(0.0, 1.0)


def test_uncomputed_lines_are_not_invented() -> None:
    prediction = ScorePrediction.from_rates(1.4, 1.1)

    assert prediction.lines == DEFAULT_LINES
    with pytest.raises(KeyError, match=r"4\.5"):
        prediction.over_under(4.5)


def test_the_matrix_is_a_read_only_private_copy() -> None:
    grid = poisson_grid(1.4, 1.1)
    prediction = ScorePrediction.from_grid(grid)
    home = prediction.home_win_probability

    grid[:] = 0  # the caller's array changes afterwards
    with pytest.raises(ValueError, match="read-only"):
        prediction.score_probability_matrix[0, 0] = 1.0

    assert prediction.score_probability_matrix.sum() == pytest.approx(1.0)
    assert prediction.home_win_probability == home


# Output object ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "expected_home_goals",
    "expected_away_goals",
    "most_probable_score",
    "score_probability_matrix",
    "home_win_probability",
    "draw_probability",
    "away_win_probability",
    "over_under_probabilities",
    "btts_probability",
}


def test_output_has_every_required_field_and_is_json() -> None:
    prediction = ScorePrediction.from_rates(1.7, 0.9, -0.08)

    data = json.loads(json.dumps(prediction.to_dict()))

    assert set(data) >= REQUIRED_FIELDS
    assert data["max_goals"] == MAX_GOALS
    matrix = np.array(data["score_probability_matrix"])
    assert matrix.shape == (SIZE, SIZE)
    assert matrix.sum() == pytest.approx(1.0, abs=TOL)
    assert set(data["over_under_probabilities"]) == {"0.5", "1.5", "2.5", "3.5"}
    for entry in data["over_under_probabilities"].values():
        assert entry["over"] + entry["under"] == pytest.approx(1.0, abs=TOL)
    assert data["btts_probability"]["yes"] + data["btts_probability"]["no"] == pytest.approx(1.0)
    best = data["most_probable_score"]
    assert matrix[best["home_goals"], best["away_goals"]] == best["probability"]
    rebuilt = ScorePrediction.from_grid(matrix)
    assert rebuilt.to_dict() == prediction.to_dict()


# The engine on every model ---------------------------------------------------------------

MODELS: dict[str, Callable[[], Predictor]] = {
    "historical_baseline": HistoricalBaseline,
    "elo": EloModel,
    "logistic_regression": lambda: LogisticModel(CORE_FEATURES, c=0.1),
    "poisson": lambda: PoissonModel(CORE_FEATURES),
    "random_forest": lambda: RandomForestModel(CORE_FEATURES, n_estimators=50),
    "xgboost": lambda: XGBoostModel(CORE_FEATURES, n_estimators=80),
    "lightgbm": lambda: LightGBMModel(CORE_FEATURES, n_estimators=80),
}


@pytest.fixture(scope="module")
def split() -> ChronologicalSplit:
    return split_by_seasons(make_dataset(seasons=4, teams=10))


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_yields_consistent_predictions(name: str, split: ChronologicalSplit) -> None:
    model = MODELS[name]().train(split.train)
    engine = GoalPredictionEngine(model, ALL_LINES)

    predictions = engine.predict(split.test)

    assert len(predictions) == len(split.test)
    for prediction, base in zip(predictions, model.predict(split.test), strict=True):
        assert_consistent(prediction)
        # The engine re-expresses the model's grid; it does not change it.
        np.testing.assert_array_equal(prediction.score_probability_matrix, base.grid)
        assert prediction.home_win_probability == base.home


def test_engine_matches_the_evaluated_market_metrics(split: ChronologicalSplit) -> None:
    model = PoissonModel(CORE_FEATURES).train(split.train)
    predictions = GoalPredictionEngine(model).predict(split.test)
    total = split.test.home_goals + split.test.away_goals
    both = (split.test.home_goals > 0) & (split.test.away_goals > 0)

    metrics = evaluate(model.predict(split.test), split.test)

    over = np.array([p.over_under(2.5).over for p in predictions])
    yes = np.array([p.btts_probability.yes for p in predictions])
    assert metrics["over_2_5_brier"] == pytest.approx(np.mean((over - (total > 2.5)) ** 2))
    assert metrics["btts_brier"] == pytest.approx(np.mean((yes - both) ** 2))
    actual = np.where(total > 2.5, over, 1 - over)
    assert metrics["over_2_5_log_loss"] == pytest.approx(-np.mean(np.log(actual)))
    assert {"over_1_5_brier", "over_3_5_log_loss", "btts_log_loss"} <= set(metrics)


def test_perfect_market_forecasts_score_zero() -> None:
    data = make_dataset(seasons=1, teams=4)
    certain = [
        ScorePrediction.from_grid(point_mass(min(e.home_goals, 10), min(e.away_goals, 10)))
        for e in data.examples
    ]
    metrics = evaluate([MatchPrediction(p.score_probability_matrix) for p in certain], data)

    for name in ("over_1_5", "over_2_5", "over_3_5", "btts"):
        assert metrics[f"{name}_brier"] == 0.0
        assert metrics[f"{name}_log_loss"] == pytest.approx(0.0, abs=1e-12)


def test_engine_needs_a_fitted_model_and_a_matching_feature_version(
    split: ChronologicalSplit, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="fitted"):
        GoalPredictionEngine(EloModel())
    model = EloModel().train(split.train)
    model.save(tmp_path / "elo", {"feature_set_version": split.train.feature_set_version})

    engine = GoalPredictionEngine.load(
        tmp_path / "elo", feature_set_version=split.train.feature_set_version
    )

    assert [p.to_dict() for p in engine.predict(split.test)] == [
        p.to_dict() for p in GoalPredictionEngine(model).predict(split.test)
    ]
    with pytest.raises(ValueError, match="trained on features"):
        GoalPredictionEngine.load(tmp_path / "elo", feature_set_version="v1-other")


def test_empty_dataset_gives_no_predictions(split: ChronologicalSplit) -> None:
    engine = GoalPredictionEngine(EloModel().train(split.train))

    assert engine.predict(Dataset((), split.test.feature_set_version)) == []
