"""Saving and loading fitted models.

A saved model is a directory with `model.joblib` (the fitted object) and
`metadata.json` (name, hyperparameters, features, feature-set version, training
period, library versions). Loading can require a feature-set version, so a model is
never silently applied to features it was not trained on. Saving never overwrites:
a directory that already holds a model is refused, so every saved model is a new,
separate version.

joblib files are pickles: only load models this project produced.
"""

import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

from app.ml.predictors import Predictor

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
LIBRARIES = ("scikit-learn", "numpy", "scipy", "xgboost", "lightgbm")


@dataclass(frozen=True)
class SavedModel:
    model: Predictor
    metadata: dict[str, Any]


def save_model(model: Predictor, directory: Path, metadata: dict[str, Any]) -> Path:
    if not model.fitted:
        raise ValueError("only fitted models can be saved")
    existing = [f for f in (MODEL_FILE, METADATA_FILE) if (directory / f).exists()]
    if existing:
        raise FileExistsError(
            f"{directory} already holds a saved model ({', '.join(existing)}); "
            "save each model version to a new directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / MODEL_FILE)
    full = {
        "model": model.name,
        "class": type(model).__name__,
        "params": model.params(),
        "used_features": list(model.used_features),
        "python": platform.python_version(),
        "libraries": _library_versions(),
        **metadata,
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(full, indent=2, default=str), encoding="utf-8"
    )
    return directory


def _library_versions() -> dict[str, str]:
    versions = {}
    for lib in LIBRARIES:
        try:
            versions[lib] = version(lib)
        except PackageNotFoundError:
            continue
    return versions


def load_model(directory: Path, *, feature_set_version: str | None = None) -> SavedModel:
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    if (
        feature_set_version is not None
        and metadata.get("feature_set_version") != feature_set_version
    ):
        raise ValueError(
            f"model was trained on features {metadata.get('feature_set_version')!r}, "
            f"not {feature_set_version!r}"
        )
    model = joblib.load(directory / MODEL_FILE)
    if not isinstance(model, Predictor):
        raise TypeError(f"{directory / MODEL_FILE} does not contain a Predictor")
    return SavedModel(model, metadata)
