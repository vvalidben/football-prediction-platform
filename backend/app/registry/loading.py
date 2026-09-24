"""Load a registered model's artifact without MLflow.

Training writes each model to an MLflow run's artifact directory, and the registry
stores that location in `model_versions.artifact_uri`. When the artifact is on a
local filesystem (`file://…` or a path, which is how this project stores them), the
API can load it directly: it needs joblib and the libraries the model was built
with, but not MLflow.

Remote artifacts (http, s3, …) need the ML environment; `load_artifact` says so
instead of failing obscurely. Loaded models are cached per model version, so a
prediction request does not re-read the file.
"""

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from app.models import ModelVersion

if TYPE_CHECKING:  # the serving image imports the model libraries lazily
    from app.ml.predictors import Predictor

logger = logging.getLogger(__name__)
LOCAL_SCHEMES = ("", "file")


class ArtifactUnavailableError(RuntimeError):
    """The model artifact cannot be loaded here."""


def local_path(artifact_uri: str) -> Path:
    parsed = urlparse(artifact_uri)
    if parsed.scheme not in LOCAL_SCHEMES:
        raise ArtifactUnavailableError(
            f"artifact {artifact_uri} is not on a local filesystem; "
            "load it in the ML environment (app.ml.registry.load_model_version)"
        )
    path = unquote(parsed.path)
    # file:///C:/... parses to /C:/...
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return Path(path if parsed.scheme else artifact_uri)


class ModelCache:
    """Loaded models, keyed by model version id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[int, tuple[str, Predictor, dict[str, Any]]] = {}

    def get(self, row: ModelVersion) -> tuple["Predictor", dict[str, Any]]:
        with self._lock:
            cached = self._models.get(row.id)
            if cached is not None and cached[0] == row.version:
                return cached[1], cached[2]
        model, metadata = load_artifact(row)
        with self._lock:
            self._models[row.id] = (row.version, model, metadata)
        return model, metadata

    def clear(self) -> None:
        with self._lock:
            self._models.clear()


def load_artifact(row: ModelVersion) -> tuple["Predictor", dict[str, Any]]:
    """The fitted model and its saved metadata."""
    if not row.artifact_uri:
        raise ArtifactUnavailableError(f"{row.name} {row.version} has no model artifact")
    directory = local_path(row.artifact_uri)
    if not directory.exists():
        raise ArtifactUnavailableError(f"model artifact {directory} does not exist")
    try:
        from app.ml.persistence import load_model

        saved = load_model(directory, feature_set_version=row.feature_set_version)
    except ImportError as exc:
        # The serving image has scikit-learn but not XGBoost or LightGBM: a tree model
        # cannot be unpickled here, and that is a service problem, not a bad request.
        raise ArtifactUnavailableError(
            f"this environment cannot load {row.algorithm} models ({exc}); "
            "install the libraries it needs, or activate a model this service supports"
        ) from exc
    logger.info("Loaded model %s %s (%s) from %s", row.name, row.version, row.algorithm, directory)
    return saved.model, saved.metadata
