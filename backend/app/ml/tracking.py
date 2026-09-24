"""Experiment tracking with MLflow.

Each model run records its hyperparameters, the feature-set version and feature
list, the dataset periods and sizes, metrics per data split, and the saved model
directory as an artifact. The tracking store is configured by `MLFLOW_TRACKING_URI`
(default: a SQLite file under `backend/mlruns/`).

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # from backend/
"""

import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from app.ml.persistence import save_model
from app.ml.predictors import Predictor

BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_DIR = BACKEND_DIR / "mlruns"


def default_tracking_uri() -> str:
    DEFAULT_TRACKING_DIR.mkdir(exist_ok=True)
    return f"sqlite:///{(DEFAULT_TRACKING_DIR / 'mlflow.db').as_posix()}"


class Tracker:
    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        self.tracking_uri = (
            tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_tracking_uri()
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        existing = mlflow.get_experiment_by_name(experiment)
        if existing is None:
            artifacts = Path(self.tracking_uri.removeprefix("sqlite:///")).parent / "artifacts"
            self.experiment_id = mlflow.create_experiment(
                experiment,
                artifact_location=artifacts.as_uri()
                if self.tracking_uri.startswith("sqlite")
                else None,
            )
        else:
            self.experiment_id = existing.experiment_id

    @contextmanager
    def run(self, name: str, tags: Mapping[str, str] | None = None) -> Iterator[str]:
        with mlflow.start_run(
            experiment_id=self.experiment_id, run_name=name, tags=dict(tags or {})
        ) as run:
            yield str(run.info.run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], prefix: str) -> None:
        mlflow.log_metrics({f"{prefix}_{key}": value for key, value in metrics.items()})

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        mlflow.log_dict(dict(data), name)

    def log_model(self, model: Predictor, metadata: Mapping[str, Any]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, Path(tmp) / "model", dict(metadata))
            mlflow.log_artifacts(str(Path(tmp) / "model"), artifact_path="model")

    @staticmethod
    def artifact_uri(run_id: str) -> str:
        return str(mlflow.get_run(run_id).info.artifact_uri) + "/model"
