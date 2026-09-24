"""Score-grid maths shared by every model.

Every model outputs a joint distribution over exact scores 0 to MAX_GOALS per side.
Outcome probabilities, expected goals, over/under and both-teams-to-score are always
derived from that grid, so a model's markets can never contradict each other.

MAX_GOALS = 10 (121 scores, 0-0 to 10-10). Scores beyond it are not represented, and
a model's Poisson grid is renormalised over the 121 cells. The mass lost is tiny for
realistic scoring rates. For a team expected to score λ goals, P(more than 10):
λ = 1.5: 5.5e-7; λ = 2.5: 6e-5; λ = 3.0: 3e-4; λ = 3.5: 1e-3. The highest
rates actually predicted are reported in the real-data verification (docs/models.md).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

MAX_GOALS = 10
GOALS = np.arange(MAX_GOALS + 1)
Grid = NDArray[np.float64]

# Boolean masks over the grid: [home goals, away goals].
HOME_WIN = GOALS[:, None] > GOALS[None, :]
DRAW = GOALS[:, None] == GOALS[None, :]
AWAY_WIN = GOALS[:, None] < GOALS[None, :]
#: Total goals of each cell.
TOTALS = GOALS[:, None] + GOALS[None, :]
MAX_TOTAL = 2 * MAX_GOALS

#: Default over/under total-goals lines. Only half-goal lines are allowed, so every
#: score is either over or under and there are no pushes.
DEFAULT_LINES = (0.5, 1.5, 2.5, 3.5)


def normalise(grid: Grid) -> Grid:
    total = grid.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("score grid has no probability mass")
    return grid / total


def poisson_grid(home_rate: float, away_rate: float, rho: float = 0.0) -> Grid:
    """Independent Poisson scores, optionally with the Dixon-Coles low-score
    correction `rho` (negative rho raises 0-0 and 1-1, lowers 1-0 and 0-1)."""
    grid = np.outer(poisson.pmf(GOALS, home_rate), poisson.pmf(GOALS, away_rate))
    if rho:
        grid[0, 0] *= max(1e-6, 1 - home_rate * away_rate * rho)
        grid[0, 1] *= max(1e-6, 1 + home_rate * rho)
        grid[1, 0] *= max(1e-6, 1 + away_rate * rho)
        grid[1, 1] *= max(1e-6, 1 - rho)
    return normalise(grid)


def outcome_probabilities(grid: Grid) -> tuple[float, float, float]:
    return (float(grid[HOME_WIN].sum()), float(grid[DRAW].sum()), float(grid[AWAY_WIN].sum()))


def check_line(line: float) -> float:
    """A half-goal total-goals line strictly inside 0 .. MAX_TOTAL."""
    if not (math.isfinite(line) and 0 < line < MAX_TOTAL and math.isclose(line % 1, 0.5)):
        raise ValueError(f"over/under line must be a half-goal line in 0.5..{MAX_TOTAL - 0.5}")
    return float(line)


def total_goals_distribution(grid: Grid) -> NDArray[np.float64]:
    """P(total goals = k) for k = 0 .. MAX_TOTAL."""
    return np.bincount(TOTALS.ravel(), weights=grid.ravel(), minlength=MAX_TOTAL + 1)


def over_under(grid: Grid, line: float) -> tuple[float, float]:
    """(P(total > line), P(total < line)). Both are summed from the grid; for a
    normalised grid they add up to 1."""
    line = check_line(line)
    return float(grid[line < TOTALS].sum()), float(grid[line > TOTALS].sum())


def both_teams_score(grid: Grid) -> tuple[float, float]:
    """(P(both teams score), P(at least one team does not)), both summed from the grid."""
    return float(grid[1:, 1:].sum()), float(grid[0, :].sum() + grid[1:, 0].sum())


def most_probable_score(grid: Grid) -> tuple[int, int, float]:
    """The most likely exact score. Ties go to the fewest home goals, then the fewest
    away goals (the first maximum in row-major order)."""
    home, away = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return int(home), int(away), float(grid[home, away])


def reconcile(grid: Grid, target: tuple[float, float, float]) -> Grid:
    """Rescale a score grid so its home/draw/away mass equals `target`, keeping the
    relative likelihood of scores within each outcome."""
    result = grid.copy()
    for mask, wanted in zip((HOME_WIN, DRAW, AWAY_WIN), target, strict=True):
        mass = result[mask].sum()
        if mass > 0:
            result[mask] *= wanted / mass
    return normalise(result)


@dataclass(frozen=True)
class MatchPrediction:
    """One model's forecast for one match, derived from a normalised score grid."""

    grid: Grid

    def __post_init__(self) -> None:
        if self.grid.shape != (MAX_GOALS + 1, MAX_GOALS + 1):
            raise ValueError("score grid has the wrong shape")
        if (
            not np.isfinite(self.grid).all()
            or not np.isclose(self.grid.sum(), 1.0, atol=1e-9)
            or (self.grid < 0).any()
        ):
            raise ValueError("score grid is not a probability distribution")

    @property
    def home(self) -> float:
        return float(self.grid[HOME_WIN].sum())

    @property
    def draw(self) -> float:
        return float(self.grid[DRAW].sum())

    @property
    def away(self) -> float:
        return float(self.grid[AWAY_WIN].sum())

    @property
    def outcomes(self) -> tuple[float, float, float]:
        return self.home, self.draw, self.away

    @property
    def expected_home_goals(self) -> float:
        return float((self.grid.sum(axis=1) * GOALS).sum())

    @property
    def expected_away_goals(self) -> float:
        return float((self.grid.sum(axis=0) * GOALS).sum())

    def over_under(self, line: float) -> tuple[float, float]:
        return over_under(self.grid, line)

    @property
    def btts(self) -> tuple[float, float]:
        return both_teams_score(self.grid)

    @property
    def most_probable_score(self) -> tuple[int, int, float]:
        return most_probable_score(self.grid)

    def score_probability(self, home_goals: int, away_goals: int) -> float:
        if home_goals > MAX_GOALS or away_goals > MAX_GOALS:
            return 0.0
        return float(self.grid[home_goals, away_goals])

    def top_scores(self, k: int) -> list[tuple[int, int, float]]:
        flat = np.argsort(self.grid, axis=None)[::-1][:k]
        return [
            (int(h), int(a), float(self.grid[h, a]))
            for h, a in zip(*np.unravel_index(flat, self.grid.shape), strict=True)
        ]
