"""Goal prediction engine: from a trained model to a complete score prediction.

A fitted model (any `Predictor`) supplies one score grid per match: the joint
probability of every exact score from 0-0 to MAX_GOALS-MAX_GOALS (10-10). The engine
turns that grid into a `ScorePrediction`. **Every field is computed from the one
grid**; nothing is estimated separately, so the fields cannot contradict each other:

- expected home / away goals: the grid's marginal means;
- most probable score: the grid's largest cell;
- home / draw / away: sums over the cells below, on and above the diagonal;
- over / under a line: sums over the cells whose total is above / below it;
- both teams to score: the sum over cells where both sides have at least one goal.

    engine = GoalPredictionEngine.load(model_dir, feature_set_version="v1-...")
    prediction = engine.predict(dataset)[0]
    prediction.to_dict()
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.dataset import Dataset
from app.ml.persistence import load_model
from app.ml.predictors import Predictor
from app.ml.probability import (
    DEFAULT_LINES,
    MAX_GOALS,
    Grid,
    MatchPrediction,
    check_line,
    poisson_grid,
)


@dataclass(frozen=True)
class ScoreProbability:
    home_goals: int
    away_goals: int
    probability: float


@dataclass(frozen=True)
class OverUnder:
    line: float
    over: float
    under: float


@dataclass(frozen=True)
class BothTeamsToScore:
    yes: float
    no: float


@dataclass(frozen=True, eq=False)  # holds an array: compare with to_dict() instead
class ScorePrediction:
    """A complete, internally consistent score prediction for one match."""

    expected_home_goals: float
    expected_away_goals: float
    most_probable_score: ScoreProbability
    #: matrix[h, a] = P(home scores h, away scores a), h and a in 0..max_goals.
    #: Read-only; sums to 1.
    score_probability_matrix: Grid
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    over_under_probabilities: tuple[OverUnder, ...]
    btts_probability: BothTeamsToScore
    max_goals: int = MAX_GOALS

    @classmethod
    def from_grid(cls, grid: Grid, lines: Sequence[float] = DEFAULT_LINES) -> "ScorePrediction":
        """Derive every field from `grid`, which must be a normalised
        (MAX_GOALS + 1) x (MAX_GOALS + 1) probability matrix."""
        matrix = np.array(grid, dtype=np.float64)  # a private copy
        base = MatchPrediction(matrix)  # validates shape, finiteness, sign and sum
        matrix.flags.writeable = False
        checked = sorted({check_line(line) for line in lines})
        home, away, probability = base.most_probable_score
        yes, no = base.btts
        return cls(
            expected_home_goals=base.expected_home_goals,
            expected_away_goals=base.expected_away_goals,
            most_probable_score=ScoreProbability(home, away, probability),
            score_probability_matrix=matrix,
            home_win_probability=base.home,
            draw_probability=base.draw,
            away_win_probability=base.away,
            over_under_probabilities=tuple(
                OverUnder(line, *base.over_under(line)) for line in checked
            ),
            btts_probability=BothTeamsToScore(yes, no),
        )

    @classmethod
    def from_rates(
        cls,
        home_rate: float,
        away_rate: float,
        rho: float = 0.0,
        lines: Sequence[float] = DEFAULT_LINES,
    ) -> "ScorePrediction":
        """Independent Poisson goals with the given expected goals (and an optional
        Dixon-Coles low-score correction), truncated at MAX_GOALS and renormalised."""
        if not (home_rate > 0 and away_rate > 0):
            raise ValueError("scoring rates must be positive")
        return cls.from_grid(poisson_grid(home_rate, away_rate, rho), lines)

    def over_under(self, line: float) -> OverUnder:
        line = check_line(line)
        for entry in self.over_under_probabilities:
            if entry.line == line:
                return entry
        raise KeyError(f"line {line} was not computed; available: {self.lines}")

    @property
    def lines(self) -> tuple[float, ...]:
        return tuple(entry.line for entry in self.over_under_probabilities)

    def top_scores(self, k: int) -> list[ScoreProbability]:
        """The k most likely exact scores, most likely first."""
        return [
            ScoreProbability(h, a, p)
            for h, a, p in MatchPrediction(self.score_probability_matrix).top_scores(k)
        ]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: plain floats, ints, lists and string keys."""
        return {
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "most_probable_score": {
                "home_goals": self.most_probable_score.home_goals,
                "away_goals": self.most_probable_score.away_goals,
                "probability": self.most_probable_score.probability,
            },
            "max_goals": self.max_goals,
            "score_probability_matrix": self.score_probability_matrix.tolist(),
            "home_win_probability": self.home_win_probability,
            "draw_probability": self.draw_probability,
            "away_win_probability": self.away_win_probability,
            "over_under_probabilities": {
                f"{entry.line:.1f}": {"over": entry.over, "under": entry.under}
                for entry in self.over_under_probabilities
            },
            "btts_probability": {
                "yes": self.btts_probability.yes,
                "no": self.btts_probability.no,
            },
        }


class GoalPredictionEngine:
    """Wraps a fitted model and produces `ScorePrediction`s."""

    def __init__(self, model: Predictor, lines: Sequence[float] = DEFAULT_LINES) -> None:
        if not model.fitted:
            raise ValueError("the engine needs a fitted model")
        self.model = model
        self.lines = tuple(sorted({check_line(line) for line in lines}))

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        feature_set_version: str,
        lines: Sequence[float] = DEFAULT_LINES,
    ) -> "GoalPredictionEngine":
        """Load a saved model. It must have been trained on `feature_set_version`,
        the version of the features it will be given."""
        saved = load_model(directory, feature_set_version=feature_set_version)
        return cls(saved.model, lines)

    def predict(self, data: Dataset) -> list[ScorePrediction]:
        return [ScorePrediction.from_grid(p.grid, self.lines) for p in self.model.predict(data)]
