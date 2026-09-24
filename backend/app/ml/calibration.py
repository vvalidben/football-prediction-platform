"""Probability calibration: measurement (reliability, ECE) and calibrators.

Calibration asks whether events given probability p happen a fraction p of the
time. Measured with:

- `reliability`: bins of predicted probability, each with its mean prediction and
  observed frequency;
- `expected_calibration_error` (ECE): the match-weighted mean gap between the two
  over the bins. For 1X2, `classwise_ece` averages the home, draw and away ECE.

Calibrators map a model's home/draw/away probabilities to calibrated ones:

- `PlattCalibrator`: for each outcome, a logistic regression on the logit of its
  probability (one-vs-rest); the three results are renormalised to sum to 1.
- `IsotonicCalibrator`: for each outcome, a monotone (isotonic) regression of the
  outcome on its probability; renormalised, with a floor so no outcome gets 0.

A calibrator must be fitted on predictions the model made **out of sample**, whose
results are known at the time the calibrator is used. The backtest enforces this
(see `backtest.py`); fitting on in-sample predictions would learn nothing true.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.ml.dataset import FloatArray, IntArray

EPS = 1e-6
#: Calibrated probabilities are floored here before renormalising (isotonic can
#: output 0, which would make the log loss infinite).
FLOOR = 1e-4
DEFAULT_BINS = 10


def reliability(
    probability: FloatArray, happened: np.ndarray, bins: int = DEFAULT_BINS
) -> list[dict[str, float]]:
    """Equal-width bins over [0, 1]; empty bins are omitted."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(happened, dtype=float)
    index = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append(
                {
                    "low": b / bins,
                    "high": (b + 1) / bins,
                    "count": float(mask.sum()),
                    "mean_predicted": float(probability[mask].mean()),
                    "observed": float(outcome[mask].mean()),
                }
            )
    return rows


def expected_calibration_error(
    probability: FloatArray, happened: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    rows = reliability(probability, happened, bins)
    total = sum(r["count"] for r in rows)
    if total == 0:
        raise ValueError("no predictions to measure")
    return float(sum(r["count"] * abs(r["mean_predicted"] - r["observed"]) for r in rows) / total)


def classwise_ece(probs: FloatArray, outcomes: IntArray, bins: int = DEFAULT_BINS) -> float:
    """Mean of the home, draw and away ECE (each outcome one-vs-rest)."""
    return float(
        np.mean([expected_calibration_error(probs[:, k], outcomes == k, bins) for k in range(3)])
    )


class Calibrator(ABC):
    method: ClassVar[str]

    def __init__(self) -> None:
        self.fitted = False
        self.fitted_on = 0

    def fit(self, probs: FloatArray, outcomes: IntArray) -> Self:
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes)
        if probs.ndim != 2 or probs.shape[1] != 3 or len(probs) != len(outcomes):
            raise ValueError("expected (matches, 3) probabilities and one outcome per match")
        if set(np.unique(outcomes)) != {0, 1, 2}:
            raise ValueError("calibration needs home wins, draws and away wins")
        self._fit(probs, outcomes)
        self.fitted = True
        self.fitted_on = len(outcomes)
        return self

    def transform(self, probs: FloatArray) -> FloatArray:
        if not self.fitted:
            raise RuntimeError(f"{self.method} calibrator has not been fitted")
        raw = np.clip(self._transform(np.asarray(probs, dtype=float)), FLOOR, None)
        calibrated: FloatArray = raw / raw.sum(axis=1, keepdims=True)
        return calibrated

    @abstractmethod
    def _fit(self, probs: FloatArray, outcomes: IntArray) -> None: ...

    @abstractmethod
    def _transform(self, probs: FloatArray) -> FloatArray: ...

    def params(self) -> dict[str, Any]:
        return {"method": self.method, "fitted_on": self.fitted_on}


def _logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, EPS, 1 - EPS)
    result: FloatArray = np.log(p / (1 - p))
    return result


class PlattCalibrator(Calibrator):
    method = "platt"

    def _fit(self, probs: FloatArray, outcomes: IntArray) -> None:
        self.models = [
            LogisticRegression(C=1e6, max_iter=1000).fit(
                _logit(probs[:, k]).reshape(-1, 1), outcomes == k
            )
            for k in range(3)
        ]

    def _transform(self, probs: FloatArray) -> FloatArray:
        return np.column_stack(
            [
                m.predict_proba(_logit(probs[:, k]).reshape(-1, 1))[:, 1]
                for k, m in enumerate(self.models)
            ]
        )

    def params(self) -> dict[str, Any]:
        base = super().params()
        if self.fitted:
            base["slopes"] = [round(float(m.coef_[0, 0]), 4) for m in self.models]
            base["intercepts"] = [round(float(m.intercept_[0]), 4) for m in self.models]
        return base


class IsotonicCalibrator(Calibrator):
    method = "isotonic"

    def _fit(self, probs: FloatArray, outcomes: IntArray) -> None:
        self.models = [
            IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                probs[:, k], (outcomes == k).astype(float)
            )
            for k in range(3)
        ]

    def _transform(self, probs: FloatArray) -> FloatArray:
        return np.column_stack([m.predict(probs[:, k]) for k, m in enumerate(self.models)])


CALIBRATORS: dict[str, type[Calibrator]] = {
    PlattCalibrator.method: PlattCalibrator,
    IsotonicCalibrator.method: IsotonicCalibrator,
}
#: "none" means the model's own probabilities.
CALIBRATION_METHODS = ("none", *CALIBRATORS)
