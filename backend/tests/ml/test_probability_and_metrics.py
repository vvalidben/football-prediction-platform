import math
from datetime import timedelta

import numpy as np
import pytest

from app.ml.dataset import Dataset, Example, FloatArray
from app.ml.metrics import evaluate, metrics_by_competition
from app.ml.probability import (
    MAX_GOALS,
    MatchPrediction,
    outcome_probabilities,
    poisson_grid,
    reconcile,
)
from tests.ml.synthetic import START


class TestScoreGrids:
    def test_poisson_grid_is_a_distribution_with_the_right_means(self) -> None:
        grid = poisson_grid(1.6, 1.1)

        prediction = MatchPrediction(grid)
        assert grid.sum() == pytest.approx(1.0)
        assert prediction.expected_home_goals == pytest.approx(1.6, abs=1e-3)
        assert prediction.expected_away_goals == pytest.approx(1.1, abs=1e-3)
        assert sum(prediction.outcomes) == pytest.approx(1.0)

    def test_dixon_coles_negative_rho_raises_low_draws(self) -> None:
        plain, corrected = poisson_grid(1.3, 1.1), poisson_grid(1.3, 1.1, rho=-0.1)

        assert corrected[0, 0] > plain[0, 0]
        assert corrected[1, 1] > plain[1, 1]
        assert corrected[1, 0] < plain[1, 0]
        assert corrected.sum() == pytest.approx(1.0)

    def test_reconcile_hits_the_target_and_keeps_within_outcome_shape(self) -> None:
        grid = poisson_grid(1.5, 1.0)
        target = (0.40, 0.35, 0.25)

        adjusted = reconcile(grid, target)

        assert outcome_probabilities(adjusted) == pytest.approx(target)
        # Relative likelihood of 1-0 vs 2-0 (both home wins) is unchanged.
        assert adjusted[1, 0] / adjusted[2, 0] == pytest.approx(grid[1, 0] / grid[2, 0])

    @pytest.mark.parametrize(
        "grid",
        [np.full((MAX_GOALS + 1, MAX_GOALS + 1), 0.5), np.zeros((3, 3)), -poisson_grid(1, 1)],
        ids=["sums-to-60", "wrong-shape", "negative"],
    )
    def test_invalid_grids_are_rejected(self, grid: FloatArray) -> None:
        with pytest.raises(ValueError, match="score grid"):
            MatchPrediction(grid)

    def test_top_scores(self) -> None:
        prediction = MatchPrediction(poisson_grid(1.2, 0.9))

        top = prediction.top_scores(3)

        assert top[0][:2] == (1, 0)
        assert [p for *_, p in top] == sorted((p for *_, p in top), reverse=True)

    def test_scores_beyond_the_grid_have_zero_probability(self) -> None:
        assert MatchPrediction(poisson_grid(1.2, 0.9)).score_probability(MAX_GOALS + 1, 0) == 0.0


def dataset(scores: list[tuple[int, int]]) -> Dataset:
    return Dataset(
        tuple(
            Example(i, START.replace(day=1 + i), 1, 1, {}, h, a) for i, (h, a) in enumerate(scores)
        ),
        "v",
    )


class TestMetricsByCompetition:
    def _data(self, per_competition: dict[int, int]) -> Dataset:
        examples = []
        for competition, count in per_competition.items():
            examples += [
                Example(
                    competition * 1000 + i,
                    START + timedelta(days=i),
                    1,
                    competition,
                    {},
                    1 if i % 2 else 2,
                    0,
                )
                for i in range(count)
            ]
        return Dataset(tuple(sorted(examples, key=lambda e: e.kickoff_at)), "v")

    def test_each_competition_is_measured_on_its_own_matches(self) -> None:
        data = self._data({1: 40, 2: 30})
        prediction = MatchPrediction(poisson_grid(1.4, 1.0))

        measured = metrics_by_competition([prediction] * len(data), data)

        assert set(measured) == {"1", "2"}
        assert measured["1"]["matches"] == 40
        assert measured["2"]["matches"] == 30
        # All home wins, and the grid favours the home side.
        assert measured["1"]["accuracy"] == pytest.approx(1.0)
        assert measured["1"]["home_win_rate"] == pytest.approx(1.0)
        assert 0 <= measured["1"]["exact_score_accuracy"] <= 1
        assert measured["1"]["log_loss"] > 0
        assert measured["1"]["start"] < measured["1"]["end"]

    def test_too_few_matches_say_nothing_about_a_competition(self) -> None:
        data = self._data({1: 40, 2: 29})
        prediction = MatchPrediction(poisson_grid(1.4, 1.0))

        measured = metrics_by_competition([prediction] * len(data), data)

        assert set(measured) == {"1"}

    def test_it_needs_one_prediction_per_match(self) -> None:
        with pytest.raises(ValueError, match="one prediction per match"):
            metrics_by_competition([], self._data({1: 40}))


class TestMetrics:
    def test_hand_computed_values(self) -> None:
        # Two matches: a 1-0 home win and a 1-1 draw, both predicted by the same grid.
        grid = poisson_grid(1.4, 1.0)
        prediction = MatchPrediction(grid)
        home, draw, away = prediction.outcomes
        data = dataset([(1, 0), (1, 1)])

        metrics = evaluate([prediction, prediction], data)

        assert metrics["accuracy"] == pytest.approx(0.5 if home > max(draw, away) else 0.0)
        assert metrics["log_loss"] == pytest.approx(-(math.log(home) + math.log(draw)) / 2)
        brier_home = (home - 1) ** 2 + draw**2 + away**2
        brier_draw = home**2 + (draw - 1) ** 2 + away**2
        assert metrics["brier"] == pytest.approx((brier_home + brier_draw) / 2)
        rps_home = ((home - 1) ** 2 + (home + draw - 1) ** 2) / 2
        rps_draw = (home**2 + (home + draw - 1) ** 2) / 2
        assert metrics["rps"] == pytest.approx((rps_home + rps_draw) / 2)
        eh, ea = prediction.expected_home_goals, prediction.expected_away_goals
        assert metrics["home_goals_mae"] == pytest.approx(abs(eh - 1))
        assert metrics["total_goals_mae"] == pytest.approx(
            (abs(eh + ea - 1) + abs(eh + ea - 2)) / 2
        )
        assert metrics["actual_score_log_prob"] == pytest.approx(
            (math.log(grid[1, 0]) + math.log(grid[1, 1])) / 2
        )
        # 1-0 is the most likely score here; 1-1 is in the top 3.
        assert metrics["exact_score_accuracy"] == pytest.approx(0.5)
        assert metrics["top3_score_coverage"] == pytest.approx(1.0)

    def test_confident_correct_predictions_score_well(self) -> None:
        sure_home = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        sure_home[2, 0] = 1.0
        metrics = evaluate([MatchPrediction(sure_home)], dataset([(2, 0)]))

        assert metrics["accuracy"] == 1.0
        assert metrics["log_loss"] == pytest.approx(0.0, abs=1e-12)
        assert metrics["brier"] == pytest.approx(0.0)
        assert metrics["exact_score_accuracy"] == 1.0
        assert metrics["actual_score_log_prob"] == pytest.approx(0.0)

    def test_impossible_actual_outcome_is_finite(self) -> None:
        sure_home = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        sure_home[2, 0] = 1.0
        metrics = evaluate([MatchPrediction(sure_home)], dataset([(0, 3)]))

        assert math.isfinite(metrics["log_loss"])
        assert metrics["log_loss"] > 30

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="one prediction per match"):
            evaluate([], dataset([(1, 0)]))
