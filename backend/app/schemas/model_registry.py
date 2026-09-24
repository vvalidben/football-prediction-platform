"""Model metadata as the API will expose it (the registry CLI uses the same shape)."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class TrainingPeriod(BaseModel):
    start: date | None
    end: date | None


class MlflowLink(BaseModel):
    name: str | None
    version: str | None


class ModelVersionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    version: str
    algorithm: str
    feature_set_version: str | None
    dataset_version: str
    training_period: TrainingPeriod
    competitions: list[dict[str, Any]]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime
    trained_at: datetime | None
    artifact_uri: str | None
    is_active: bool
    mlflow: MlflowLink
    description: str | None


class ActivationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    version: str
    previous_version: str | None
    activated_at: datetime
    deactivated_at: datetime | None
    reason: str | None
    is_rollback: bool
