"""Mirror the registry into the MLflow Model Registry.

The database is the source of truth. MLflow gets, for each registered version, a
registered-model version pointing at the same artifact (tagged with the database
version, algorithm, dataset and feature versions), and the alias `active` follows
activations and rollbacks, so the MLflow UI shows the same state.

MLflow is imported lazily: without it (e.g. in the API image) the registry works
and mirroring is reported as skipped.
"""

import os
from pathlib import Path
from typing import Any

ACTIVE_ALIAS = "active"
DEFAULT_TRACKING = Path(__file__).resolve().parents[2] / "mlruns" / "mlflow.db"


def tracking_uri(explicit: str | None = None) -> str:
    return (
        explicit
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{DEFAULT_TRACKING.as_posix()}"
    )


class MlflowRegistry:
    def __init__(self, uri: str | None = None) -> None:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        from mlflow import MlflowClient

        self.uri = tracking_uri(uri)
        self.client = MlflowClient(tracking_uri=self.uri, registry_uri=self.uri)

    def register(self, name: str, source: str, run_id: str | None, tags: dict[str, Any]) -> str:
        """Create the registered model if needed and add a version; returns its number."""
        from mlflow.exceptions import MlflowException

        try:
            self.client.get_registered_model(name)
        except MlflowException:
            self.client.create_registered_model(
                name, description="Mirrored from the platform's model_versions table."
            )
        version = self.client.create_model_version(
            name, source=source, run_id=run_id, tags={k: str(v) for k, v in tags.items()}
        )
        return str(version.version)

    def set_active(self, name: str, version: str | None) -> None:
        """Point the `active` alias at `version`, or remove it when None."""
        if version is None:
            self.client.delete_registered_model_alias(name, ACTIVE_ALIAS)
        else:
            self.client.set_registered_model_alias(name, ACTIVE_ALIAS, version)

    def active_version(self, name: str) -> str | None:
        from mlflow.exceptions import MlflowException

        try:
            return str(self.client.get_model_version_by_alias(name, ACTIVE_ALIAS).version)
        except MlflowException:
            return None


def sync_active(mlflow_name: str | None, mlflow_version: str | None, uri: str | None = None) -> str:
    """Best effort: returns what happened, never raises (the database already changed)."""
    if mlflow_name is None:
        return "not mirrored in MLflow (no MLflow model version recorded)"
    try:
        MlflowRegistry(uri).set_active(mlflow_name, mlflow_version)
    except ImportError:
        return "MLflow is not installed; alias not updated"
    except Exception as exc:
        return f"MLflow alias not updated: {exc}"
    target = f"version {mlflow_version}" if mlflow_version else "nothing"
    return f"MLflow alias '{ACTIVE_ALIAS}' of {mlflow_name} now points to {target}"
