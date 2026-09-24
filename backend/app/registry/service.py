"""Model version management: registration, the active model, history and rollback.

The `model_versions` table is the source of truth; `model_activations` records every
period during which a version was active. Rules:

- **Registering** only inserts a new, inactive version, with every required field:
  algorithm, feature version, dataset version, training period, competitions,
  hyperparameters, metrics. Nothing existing is modified.
- **At most one active version per model name** (enforced by partial unique indexes
  on both tables). Activating a version deactivates the current one in the same
  transaction and records which version it replaced.
- **Only loadable versions can be activated**: a version without an artifact (e.g.
  a backtest record) cannot become active.
- **Rollback** re-activates the version that was active before the current
  activation (or an explicitly named earlier version) and is recorded as a rollback.
- Activation changes for one model name are serialised with a transaction-level
  PostgreSQL advisory lock, so concurrent activations cannot interleave.

This module has no ML dependencies: the API can use it to expose model metadata.
Loading a model's artifact is `app.ml.registry.load_model_version`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ModelActivation, ModelVersion
from app.schemas.model_registry import (
    ActivationRecord,
    MlflowLink,
    ModelVersionMetadata,
    TrainingPeriod,
)


class RegistryError(ValueError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class NewModelVersion:
    """Everything a model version must record."""

    name: str
    version: str
    algorithm: str
    feature_set_version: str
    dataset_version: str
    training_data_start: date
    training_data_end: date
    competitions: Sequence[dict[str, Any]]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    artifact_uri: str | None
    trained_at: datetime
    description: str | None = None
    mlflow_model_name: str | None = None
    mlflow_model_version: str | None = None

    def validate(self) -> None:
        for field_name in (
            "name",
            "version",
            "algorithm",
            "feature_set_version",
            "dataset_version",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise RegistryError(f"a model version needs a {field_name.replace('_', ' ')}")
        if self.training_data_end < self.training_data_start:
            raise RegistryError("the training period ends before it starts")
        if not self.competitions:
            raise RegistryError("a model version needs the competitions it was trained on")
        if any("id" not in c for c in self.competitions):
            raise RegistryError("every competition needs an id")


class ModelRegistry:
    def __init__(self, session: Session, *, clock: Callable[[], datetime] = utc_now) -> None:
        self.session = session
        self.clock = clock

    # Reading -------------------------------------------------------------------------

    def names(self) -> list[str]:
        return list(
            self.session.scalars(select(ModelVersion.name).distinct().order_by(ModelVersion.name))
        )

    def versions(self, name: str | None = None) -> list[ModelVersion]:
        """Newest first."""
        query = select(ModelVersion).order_by(
            ModelVersion.created_at.desc(), ModelVersion.id.desc()
        )
        if name is not None:
            query = query.where(ModelVersion.name == name)
        return list(self.session.scalars(query))

    def find(self, name: str, version: str) -> ModelVersion | None:
        return self.session.scalars(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.version == version)
        ).one_or_none()

    def active(self, name: str) -> ModelVersion | None:
        return self.session.scalars(
            select(ModelVersion).where(ModelVersion.name == name, ModelVersion.is_active)
        ).one_or_none()

    def get(self, name: str, version: str | None = None) -> ModelVersion:
        """A specific version, or the active one when `version` is None."""
        row = self.find(name, version) if version is not None else self.active(name)
        if row is None:
            raise RegistryError(
                f"no version {version!r} of model {name!r}"
                if version is not None
                else f"model {name!r} has no active version"
            )
        return row

    def history(self, name: str) -> list[ModelActivation]:
        """Every activation of the name, newest first."""
        return list(
            self.session.scalars(
                select(ModelActivation)
                .where(ModelActivation.model_name == name)
                .order_by(ModelActivation.activated_at.desc(), ModelActivation.id.desc())
            )
        )

    def previous(self, name: str) -> list[ModelVersion]:
        """Versions that were active before, most recently active first, excluding the
        current one."""
        current = self.active(name)
        seen: dict[int, ModelVersion] = {}
        for activation in self.history(name):
            version = activation.model_version
            if (current is None or version.id != current.id) and version.id not in seen:
                seen[version.id] = version
        return list(seen.values())

    # Changing --------------------------------------------------------------------------

    def register(self, new: NewModelVersion) -> ModelVersion:
        new.validate()
        if self.find(new.name, new.version) is not None:
            raise RegistryError(f"{new.name} {new.version} is already registered")
        row = ModelVersion(
            name=new.name,
            version=new.version,
            description=new.description,
            algorithm=new.algorithm,
            feature_set_version=new.feature_set_version,
            dataset_version=new.dataset_version,
            competitions=[dict(c) for c in new.competitions],
            hyperparameters=new.hyperparameters,
            training_metrics=new.metrics,
            training_data_start=new.training_data_start,
            training_data_end=new.training_data_end,
            trained_at=new.trained_at,
            artifact_uri=new.artifact_uri,
            mlflow_model_name=new.mlflow_model_name,
            mlflow_model_version=new.mlflow_model_version,
            is_active=False,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def activate(self, name: str, version: str, *, reason: str | None = None) -> ModelActivation:
        return self._activate(name, version, reason=reason, rollback=False)

    def rollback(
        self, name: str, *, to_version: str | None = None, reason: str | None = None
    ) -> ModelActivation:
        """Re-activate the version active before the current one, or `to_version`."""
        self._lock(name)
        active = self.active(name)
        if active is None:
            raise RegistryError(f"model {name!r} has no active version to roll back")
        current = self._open_activation(name) or self._record_legacy(active)
        if to_version is None:
            if current.previous_version is None:
                raise RegistryError(f"model {name!r} has no previous version to roll back to")
            target = current.previous_version.version
        else:
            target = to_version
            row = self.get(name, target)
            if not any(a.model_version_id == row.id for a in self.history(name)):
                raise RegistryError(f"{name} {target} was never active; activate it instead")
        if target == current.model_version.version:
            raise RegistryError(f"{name} {target} is already active")
        text = reason or f"rollback from {current.model_version.version} to {target}"
        return self._activate(name, target, reason=text, rollback=True)

    def deactivate(self, name: str, *, reason: str | None = None) -> None:
        """Leave the name without an active version."""
        self._lock(name)
        active = self.active(name)
        if active is None:
            raise RegistryError(f"model {name!r} has no active version")
        current = self._open_activation(name) or self._record_legacy(active)
        current.deactivated_at = self.clock()
        if reason:
            current.reason = f"{current.reason or ''}; deactivated: {reason}".lstrip("; ")
        active.is_active = False
        self.session.flush()

    def _activate(
        self, name: str, version: str, *, reason: str | None, rollback: bool
    ) -> ModelActivation:
        self._lock(name)
        target = self.get(name, version)
        if not target.artifact_uri:
            raise RegistryError(f"{name} {version} has no model artifact and cannot be activated")
        current = self._open_activation(name)
        # The active flag is authoritative: a version activated before activations
        # were recorded is active without an open activation row.
        active = self.active(name)
        if active is not None and active.id == target.id:
            return current if current is not None else self._record_legacy(active)
        if active is not None and current is None:
            current = self._record_legacy(active)  # so it appears in the history
        now = self.clock()
        if current is not None:
            current.deactivated_at = now
        if active is not None:
            active.is_active = False
        self.session.flush()  # free the one-active slot before taking it
        target.is_active = True
        activation = ModelActivation(
            model_name=name,
            model_version=target,
            previous_version_id=active.id if active is not None else None,
            activated_at=now,
            reason=reason,
            is_rollback=rollback,
        )
        self.session.add(activation)
        self.session.flush()
        return activation

    def _record_legacy(self, row: ModelVersion) -> ModelActivation:
        """Give an active version without history its open activation row."""
        now = self.clock()
        activation = ModelActivation(
            model_name=row.name,
            model_version=row,
            # When it became active is unknown; its last update is the best guess, and
            # never later than now (the row may have been updated since).
            activated_at=min(row.updated_at or now, now),
            reason="active before activations were recorded",
        )
        self.session.add(activation)
        self.session.flush()
        return activation

    def _open_activation(self, name: str) -> ModelActivation | None:
        return self.session.scalars(
            select(ModelActivation).where(
                ModelActivation.model_name == name, ModelActivation.deactivated_at.is_(None)
            )
        ).one_or_none()

    def _lock(self, name: str) -> None:
        self.session.execute(
            select(func.pg_advisory_xact_lock(func.hashtext(f"model_registry:{name}")))
        )


def metadata(row: ModelVersion) -> ModelVersionMetadata:
    """Everything recorded about a version (for the CLI and, later, the API)."""
    return ModelVersionMetadata(
        id=row.id,
        name=row.name,
        version=row.version,
        algorithm=row.algorithm,
        feature_set_version=row.feature_set_version,
        dataset_version=row.dataset_version,
        training_period=TrainingPeriod(start=row.training_data_start, end=row.training_data_end),
        competitions=row.competitions,
        hyperparameters=row.hyperparameters,
        metrics=row.training_metrics,
        created_at=row.created_at,
        trained_at=row.trained_at,
        artifact_uri=row.artifact_uri,
        is_active=row.is_active,
        mlflow=MlflowLink(name=row.mlflow_model_name, version=row.mlflow_model_version),
        description=row.description,
    )


def activation_record(activation: ModelActivation) -> ActivationRecord:
    return ActivationRecord(
        model_name=activation.model_name,
        version=activation.model_version.version,
        previous_version=(
            activation.previous_version.version if activation.previous_version else None
        ),
        activated_at=activation.activated_at,
        deactivated_at=activation.deactivated_at,
        reason=activation.reason,
        is_rollback=activation.is_rollback,
    )
