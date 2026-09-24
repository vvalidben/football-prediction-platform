"""Registered models and their evaluation metrics."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query, Request, status

from app.api.dependencies import DbSessionDep
from app.models import ModelVersion
from app.registry.service import ModelRegistry, activation_record, metadata
from app.schemas.explanations import ImportanceRead
from app.schemas.model_registry import ActivationRecord, ModelVersionMetadata
from app.services import analytics
from app.services.explanations import DEFAULT_TOP, model_importance
from app.services.predictions import PredictionError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"], prefix="/models")


@router.get("", response_model=list[ModelVersionMetadata], summary="Registered model versions")
def list_models(
    session: DbSessionDep,
    name: Annotated[str | None, Query(description="Only this model name")] = None,
    active_only: Annotated[bool, Query(description="Only the active version of each name")] = False,
) -> list[ModelVersionMetadata]:
    """Every registered version, newest first: algorithm, feature and dataset versions,
    training period, competitions, hyperparameters and metrics."""
    rows = ModelRegistry(session).versions(name)
    if active_only:
        rows = [row for row in rows if row.is_active]
    return [metadata(row) for row in rows]


@router.get(
    "/{model_id}",
    response_model=ModelVersionMetadata,
    summary="One model version",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such model version"}},
)
def read_model(session: DbSessionDep, model_id: Annotated[int, Path(ge=1)]) -> ModelVersionMetadata:
    return metadata(_version(session, model_id))


@router.get(
    "/{model_id}/metrics",
    summary="Evaluation metrics of a model version",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such model version"}},
)
def read_metrics(session: DbSessionDep, model_id: Annotated[int, Path(ge=1)]) -> dict[str, Any]:
    """The metrics recorded when the version was registered: validation,
    cross-validation and test figures, as produced by the training run."""
    row = _version(session, model_id)
    return {
        "model": {
            "id": row.id,
            "name": row.name,
            "version": row.version,
            "algorithm": row.algorithm,
            "feature_set_version": row.feature_set_version,
            "dataset_version": row.dataset_version,
            "is_active": row.is_active,
        },
        "metrics": row.training_metrics,
    }


@router.get(
    "/{model_id}/history",
    response_model=list[ActivationRecord],
    summary="When this model's name was served by which version",
)
def read_history(
    session: DbSessionDep, model_id: Annotated[int, Path(ge=1)]
) -> list[ActivationRecord]:
    row = _version(session, model_id)
    return [activation_record(a) for a in ModelRegistry(session).history(row.name)]


def _version(session: DbSessionDep, model_id: int) -> ModelVersion:
    row = session.get(ModelVersion, model_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"model version {model_id} was not found"
        )
    return row


@router.get(
    "/{model_id}/performance",
    summary="How this version has actually done",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such model version"}},
)
def read_performance(session: DbSessionDep, model_id: Annotated[int, Path(ge=1)]) -> dict[str, Any]:
    """Accuracy, scores, calibration bins and a monthly series, computed from this
    version's stored predictions whose match has been played.

    Everything is empty until predictions made by this version have been evaluated;
    nothing here is carried over from training.
    """
    row = _version(session, model_id)
    return {
        "model": {
            "id": row.id,
            "name": row.name,
            "version": row.version,
            "algorithm": row.algorithm,
            "is_active": row.is_active,
        },
        **analytics.model_performance(session, row.id),
    }


@router.get(
    "/{model_id}/importance",
    response_model=ImportanceRead,
    summary="What this model relies on in general",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No such model version"},
        status.HTTP_409_CONFLICT: {"description": "This model has no importance measure"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The model artifact is unavailable"},
    },
)
def read_importance(
    request: Request,
    session: DbSessionDep,
    model_id: Annotated[int, Path(ge=1)],
    top: Annotated[int, Query(ge=1, le=100, description="Features to list")] = DEFAULT_TOP,
) -> ImportanceRead:
    """How much this version's fitted model relies on each feature, across all
    matches.

    This is model behaviour, not a ranking of what matters in football: a feature
    can rank high because the model leans on it, including where it stands in for
    other features it moves with. For one match's figures use
    `/predictions/{prediction_id}/explanation`.
    """
    row = _version(session, model_id)
    predictions: Any = request.app.state.predictions
    try:
        return model_importance(session, row, cache=predictions.cache, top=top)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
