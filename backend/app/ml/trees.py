"""Tree-ensemble models: random forest, XGBoost and LightGBM.

They take the same inputs and give the same outputs as the baselines. Input is the
stored feature vector (the Sprint 4 snapshots, e.g. `CORE_FEATURES`); there is no
model-specific feature engineering. Output is a full score grid per match. Each model
has:

- a classifier for home / draw / away;
- two regressors with a Poisson objective, for home and away goals. Their rates
  shape a Poisson score grid, which is rescaled to the classifier's 1X2 (as in
  `LogisticModel`).

The classifier and both goal regressors share one set of hyperparameters, so a
grid point is a single configuration. Missing feature values are handled natively
by all three libraries: no imputation. Random seeds are fixed, and predictions are
computed in a fixed order, so refits and reloaded models reproduce predictions
exactly.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.ml.dataset import AWAY, DRAW, HOME, Dataset
from app.ml.predictors import Predictor, _observed
from app.ml.probability import MatchPrediction, poisson_grid, reconcile

SEED = 0
#: Goal rates are clipped to this range, as for the Poisson GLMs.
RATE_RANGE = (0.05, 8.0)


class TreeEnsembleModel(Predictor):
    #: Hyperparameters used unless overridden; every value is tracked.
    DEFAULTS: ClassVar[dict[str, Any]]

    def __init__(self, features: Sequence[str], **hyperparameters: Any) -> None:
        super().__init__()
        unknown = set(hyperparameters) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown {self.name} hyperparameters: {sorted(unknown)}")
        self.features = tuple(features)
        self.hyperparameters = {**self.DEFAULTS, **hyperparameters}
        self.classifier: Any = None
        self.home_goals: Any = None
        self.away_goals: Any = None

    def params(self) -> dict[str, Any]:
        return {**self.hyperparameters, "features": len(self.features)}

    @abstractmethod
    def _classifier(self) -> Any: ...

    @abstractmethod
    def _goal_regressor(self) -> Any: ...

    def _fit(self, train: Dataset) -> None:
        if set(np.unique(train.outcomes)) != {HOME, DRAW, AWAY}:
            raise ValueError("training data must contain home wins, draws and away wins")
        self.used_features = _observed(train, self.features)
        x = train.matrix(self.used_features)
        self.classifier = self._classifier().fit(x, train.outcomes)
        self.home_goals = self._goal_regressor().fit(x, train.home_goals)
        self.away_goals = self._goal_regressor().fit(x, train.away_goals)

    def _predict(self, data: Dataset) -> list[MatchPrediction]:
        x = data.matrix(self.used_features)
        probs = self.classifier.predict_proba(x)
        home_rates = np.clip(self.home_goals.predict(x), *RATE_RANGE)
        away_rates = np.clip(self.away_goals.predict(x), *RATE_RANGE)
        return [
            MatchPrediction(
                reconcile(
                    poisson_grid(float(rh), float(ra)),
                    (float(p[HOME]), float(p[DRAW]), float(p[AWAY])),
                )
            )
            for p, rh, ra in zip(probs, home_rates, away_rates, strict=True)
        ]

    def feature_importances(self) -> dict[str, float]:
        """The 1X2 classifier's feature importances (library default measure),
        normalised to sum to 1, largest first."""
        if not self.fitted:
            return {}
        values = np.asarray(self.classifier.feature_importances_, dtype=float)
        total = values.sum()
        if total > 0:
            values = values / total
        ranked = sorted(zip(self.used_features, values, strict=True), key=lambda kv: -kv[1])
        return {name: float(value) for name, value in ranked}


class RandomForestModel(TreeEnsembleModel):
    """scikit-learn random forests: Gini classifier, Poisson-criterion regressors."""

    name = "random_forest"
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_estimators": 500,
        "max_depth": 6,
        "min_samples_leaf": 30,
        "max_features": 0.33,
    }

    def _classifier(self) -> Any:
        return RandomForestClassifier(**self.hyperparameters, random_state=SEED, n_jobs=-1)

    def _goal_regressor(self) -> Any:
        return RandomForestRegressor(
            **self.hyperparameters, criterion="poisson", random_state=SEED, n_jobs=-1
        )

    def _fit(self, train: Dataset) -> None:
        super()._fit(train)
        # Trees are built in parallel (each has its own seed), but parallel prediction
        # sums tree outputs in a varying order, changing the last bit of probabilities
        # between calls. Predict single-threaded so every call gives identical output.
        for forest in (self.classifier, self.home_goals, self.away_goals):
            forest.set_params(n_jobs=1)


class XGBoostModel(TreeEnsembleModel):
    """Gradient-boosted trees (XGBoost): softprob classifier, `count:poisson` regressors."""

    name = "xgboost"
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 10.0,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_lambda": 10.0,
    }

    def _common(self) -> dict[str, Any]:
        return {**self.hyperparameters, "tree_method": "hist", "random_state": SEED, "n_jobs": 4}

    def _classifier(self) -> Any:
        return XGBClassifier(**self._common(), objective="multi:softprob")

    def _goal_regressor(self) -> Any:
        return XGBRegressor(**self._common(), objective="count:poisson")


class LightGBMModel(TreeEnsembleModel):
    """Gradient-boosted trees (LightGBM): multiclass classifier, Poisson regressors."""

    name = "lightgbm"
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "learning_rate": 0.02,
        "num_leaves": 7,
        "min_child_samples": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_lambda": 10.0,
    }

    def _common(self) -> dict[str, Any]:
        return {
            **self.hyperparameters,
            "subsample_freq": 1,
            "random_state": SEED,
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": 4,
            "verbose": -1,
        }

    def _classifier(self) -> Any:
        return LGBMClassifier(**self._common(), objective="multiclass")

    def _goal_regressor(self) -> Any:
        return LGBMRegressor(**self._common(), objective="poisson")
