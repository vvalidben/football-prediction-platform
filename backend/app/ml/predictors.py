"""Baseline prediction models.

Every model maps a `Dataset` to one `MatchPrediction` (a full score grid) per match:

- `HistoricalBaseline`: the same smoothed empirical score distribution for every
  match. The floor any real model must beat.
- `EloModel`: an ordered-logit model on the Elo difference (three parameters) for
  home/draw/away. Goals come from the Elo system's attack/defence ratings, with the
  score grid rescaled to the ordered-logit 1X2.
- `LogisticModel`: multinomial logistic regression on the feature vector for 1X2.
  Poisson regressions shape the score grid, which is rescaled to the logistic 1X2.
- `PoissonModel`: two Poisson regressions (home and away goals) with a fitted
  Dixon-Coles low-score correction; 1X2 is derived from the goals.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.dataset import AWAY, DRAW, HOME, Dataset, FloatArray, IntArray
from app.ml.probability import MAX_GOALS, MatchPrediction, poisson_grid, reconcile

EPS = 1e-12


class NotFittedError(RuntimeError):
    pass


class Predictor(ABC):
    """A model producing full score-grid predictions.

    Every model, baseline or machine-learning, has the same interface: `train` (or
    `fit`), `predict` (score grids), `predict_proba` (home/draw/away), `save`, `load`.
    """

    name: ClassVar[str]

    def __init__(self) -> None:
        self.fitted = False
        #: Features actually used after dropping those never observed in training.
        self.used_features: tuple[str, ...] = ()

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Hyperparameters (not learned weights), for tracking."""

    @abstractmethod
    def _fit(self, train: Dataset) -> None: ...

    @abstractmethod
    def _predict(self, data: Dataset) -> list[MatchPrediction]: ...

    def fit(self, train: Dataset) -> Self:
        if len(train) == 0:
            raise ValueError("cannot fit on an empty dataset")
        self._fit(train)
        self.fitted = True
        return self

    def predict(self, data: Dataset) -> list[MatchPrediction]:
        """One full score-grid prediction per match."""
        if not self.fitted:
            raise NotFittedError(f"{self.name} has not been fitted")
        return self._predict(data)

    # Common model interface: train / predict / predict_proba / save / load.

    def train(self, data: Dataset) -> Self:
        """Same as `fit`."""
        self.fit(data)
        return self

    def predict_proba(self, data: Dataset) -> FloatArray:
        """Home/draw/away probabilities, shape (matches, 3), columns HOME, DRAW, AWAY."""
        return np.array([p.outcomes for p in self.predict(data)], dtype=float).reshape(-1, 3)

    def save(self, directory: Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write a new model directory. Refuses to overwrite an existing model."""
        from app.ml.persistence import save_model

        return save_model(self, directory, dict(metadata or {}))

    @classmethod
    def load(cls, directory: Path, *, feature_set_version: str | None = None) -> Self:
        """Load a saved model of this class (only files this project produced)."""
        from app.ml.persistence import load_model

        model = load_model(directory, feature_set_version=feature_set_version).model
        if not isinstance(model, cls):
            raise TypeError(f"{directory} holds a {type(model).__name__}, not a {cls.__name__}")
        return model


def _observed(train: Dataset, names: Sequence[str]) -> tuple[str, ...]:
    """Features with at least one value in the training data."""
    matrix = train.matrix(names)
    return tuple(
        name for name, column in zip(names, matrix.T, strict=True) if not np.isnan(column).all()
    )


class HistoricalBaseline(Predictor):
    name = "historical_baseline"

    def __init__(self, smoothing: float = 1.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.grid: FloatArray | None = None

    def params(self) -> dict[str, Any]:
        return {"smoothing": self.smoothing}

    def _fit(self, train: Dataset) -> None:
        counts = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for h, a in zip(train.home_goals, train.away_goals, strict=True):
            counts[min(h, MAX_GOALS), min(a, MAX_GOALS)] += 1
        # Smooth with an independent-Poisson prior so unseen scores keep some mass.
        prior = poisson_grid(float(train.home_goals.mean()), float(train.away_goals.mean()))
        self.grid = (counts + self.smoothing * prior * counts.size) / (
            counts.sum() + self.smoothing * counts.size
        )

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        assert self.grid is not None  # noqa: S101 - set by fit
        return [MatchPrediction(self.grid.copy()) for _ in data.examples]


class EloModel(Predictor):
    """Ordered logit on the Elo difference.

    P(away) = sigmoid(c1 - s*x),  P(home) = 1 - sigmoid(c2 - s*x),  P(draw) = rest,
    with x = elo_diff / 400 (home advantage already included) and c1 < c2.
    """

    name = "elo"
    FEATURE = "elo_diff"
    GOAL_FEATURES = ("rating_expected_home_goals", "rating_expected_away_goals")

    def __init__(self) -> None:
        super().__init__()
        self.slope = 1.0
        self.cutpoints = (-0.5, 0.5)
        self.goal_scale = (1.0, 1.0)

    def params(self) -> dict[str, Any]:
        return {"feature": self.FEATURE}

    def _probabilities(self, x: FloatArray, slope: float, c1: float, c2: float) -> FloatArray:
        away = expit(c1 - slope * x)
        not_home = expit(c2 - slope * x)
        return np.column_stack([1 - not_home, not_home - away, away])

    def _fit(self, train: Dataset) -> None:
        x = np.nan_to_num(train.matrix([self.FEATURE])[:, 0]) / 400.0
        y = train.outcomes

        def nll(theta: FloatArray) -> float:
            slope, c1, gap = theta
            probs = self._probabilities(x, slope, c1, c1 + math.exp(gap))
            return float(-np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1)).sum())

        fitted = minimize(nll, x0=np.array([1.0, -0.5, 0.0]), method="Nelder-Mead")
        slope, c1, gap = fitted.x
        self.slope, self.cutpoints = float(slope), (float(c1), float(c1 + math.exp(gap)))
        rates = np.nan_to_num(train.matrix(self.GOAL_FEATURES), nan=1.35)
        self.goal_scale = (
            float(train.home_goals.mean() / max(rates[:, 0].mean(), EPS)),
            float(train.away_goals.mean() / max(rates[:, 1].mean(), EPS)),
        )
        self.used_features = (self.FEATURE, *self.GOAL_FEATURES)

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        x = np.nan_to_num(data.matrix([self.FEATURE])[:, 0]) / 400.0
        probs = self._probabilities(x, self.slope, *self.cutpoints)
        rates = np.nan_to_num(data.matrix(self.GOAL_FEATURES), nan=1.35)
        return [
            MatchPrediction(
                reconcile(
                    poisson_grid(rh * self.goal_scale[0], ra * self.goal_scale[1]),
                    (float(p[HOME]), float(p[DRAW]), float(p[AWAY])),
                )
            )
            for p, (rh, ra) in zip(probs, rates, strict=True)
        ]


def _regression_pipeline(estimator: Any) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        estimator,
    )


class _GoalRegressions:
    """Two Poisson GLMs, one per side, on the same feature vector."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.home = _regression_pipeline(PoissonRegressor(alpha=alpha, max_iter=1000))
        self.away = _regression_pipeline(PoissonRegressor(alpha=alpha, max_iter=1000))

    def fit(self, x: FloatArray, home: IntArray, away: IntArray) -> None:
        self.home.fit(x, home)
        self.away.fit(x, away)

    def rates(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        clip = (0.05, 8.0)
        return (
            np.clip(self.home.predict(x), *clip),
            np.clip(self.away.predict(x), *clip),
        )


class LogisticModel(Predictor):
    name = "logistic_regression"

    def __init__(self, features: Sequence[str], c: float = 1.0, goal_alpha: float = 1.0) -> None:
        super().__init__()
        self.features = tuple(features)
        self.c = c
        self.goal_alpha = goal_alpha
        self.classifier = _regression_pipeline(LogisticRegression(C=c, max_iter=5000))
        self.goals = _GoalRegressions(goal_alpha)

    def params(self) -> dict[str, Any]:
        return {"C": self.c, "goal_alpha": self.goal_alpha, "features": len(self.features)}

    def _fit(self, train: Dataset) -> None:
        self.used_features = _observed(train, self.features)
        x = train.matrix(self.used_features)
        self.classifier.fit(x, train.outcomes)
        if list(self.classifier.classes_) != [HOME, DRAW, AWAY]:
            raise ValueError("training data must contain home wins, draws and away wins")
        self.goals.fit(x, train.home_goals, train.away_goals)

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        x = data.matrix(self.used_features)
        probs = self.classifier.predict_proba(x)
        home_rates, away_rates = self.goals.rates(x)
        return [
            MatchPrediction(
                reconcile(poisson_grid(rh, ra), (float(p[0]), float(p[1]), float(p[2])))
            )
            for p, rh, ra in zip(probs, home_rates, away_rates, strict=True)
        ]


class PoissonModel(Predictor):
    name = "poisson"

    def __init__(
        self, features: Sequence[str], alpha: float = 1.0, dixon_coles: bool = True
    ) -> None:
        super().__init__()
        self.features = tuple(features)
        self.alpha = alpha
        self.dixon_coles = dixon_coles
        self.goals = _GoalRegressions(alpha)
        self.rho = 0.0

    def params(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "dixon_coles": self.dixon_coles,
            "features": len(self.features),
        }

    def _fit(self, train: Dataset) -> None:
        self.used_features = _observed(train, self.features)
        x = train.matrix(self.used_features)
        self.goals.fit(x, train.home_goals, train.away_goals)
        if not self.dixon_coles:
            return
        home_rates, away_rates = self.goals.rates(x)
        hg, ag = train.home_goals, train.away_goals

        def nll(rho: float) -> float:
            tau = np.ones(len(hg))
            tau[(hg == 0) & (ag == 0)] = (
                1 - home_rates[(hg == 0) & (ag == 0)] * away_rates[(hg == 0) & (ag == 0)] * rho
            )
            tau[(hg == 0) & (ag == 1)] = 1 + home_rates[(hg == 0) & (ag == 1)] * rho
            tau[(hg == 1) & (ag == 0)] = 1 + away_rates[(hg == 1) & (ag == 0)] * rho
            tau[(hg == 1) & (ag == 1)] = 1 - rho
            if (tau <= 0).any():
                return math.inf
            ll = np.log(tau) + poisson.logpmf(hg, home_rates) + poisson.logpmf(ag, away_rates)
            return float(-ll.sum())

        self.rho = float(minimize_scalar(nll, bounds=(-0.3, 0.3), method="bounded").x)

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        home_rates, away_rates = self.goals.rates(data.matrix(self.used_features))
        return [
            MatchPrediction(poisson_grid(float(rh), float(ra), self.rho))
            for rh, ra in zip(home_rates, away_rates, strict=True)
        ]
