"""Register trained models and load registered ones.

Registration goes through the model registry service (`app.registry.service`): a
new, inactive `model_versions` row with every required field (algorithm, feature
version, dataset version, training period, competitions, hyperparameters, metrics),
mirrored as a version in the MLflow Model Registry. It never modifies an existing
version or changes which version is active; activation is a deliberate, separate
step (`python -m app.ml registry activate`).

`load_model_version` loads the active (or a named) version from its artifact and
refuses one whose feature-set version differs from the features it will be given.
"""

import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.ensemble import EnsembleResult
from app.ml.experiment import CandidateResult, ExperimentResult
from app.ml.metrics import SELECTION_METRIC
from app.ml.persistence import load_model
from app.ml.predictors import Predictor
from app.ml.tracking import Tracker
from app.models import Competition, ModelVersion
from app.registry.mlflow_sync import MlflowRegistry
from app.registry.service import ModelRegistry, NewModelVersion


def competitions(session: Session, ids: list[int]) -> list[dict[str, Any]]:
    names = dict(
        session.execute(select(Competition.id, Competition.name).where(Competition.id.in_(ids)))
        .tuples()
        .all()
    )
    return [{"id": i, "name": names.get(i)} for i in ids]


def _version(run_id: str) -> str:
    return f"{datetime.now(UTC):%Y%m%d%H%M%S}-{run_id[:8]}"


def _register(
    session: Session, new: NewModelVersion, tracker: Tracker, run_id: str, *, mirror: bool
) -> ModelVersion:
    registry = ModelRegistry(session)
    row = registry.register(new)
    if mirror and new.artifact_uri:
        mlflow_version = MlflowRegistry(tracker.tracking_uri).register(
            new.name,
            source=new.artifact_uri,
            run_id=run_id,
            tags={
                "db_version": new.version,
                "algorithm": new.algorithm,
                "dataset_version": new.dataset_version,
                "feature_set_version": new.feature_set_version,
            },
        )
        row.mlflow_model_name, row.mlflow_model_version = new.name, mlflow_version
        session.flush()
    return row


def _fit(candidate: CandidateResult) -> tuple[dict[str, Any], str]:
    if candidate.fit_period is None or candidate.fit_dataset_version is None:
        raise ValueError(f"{candidate.name} has no fitted final model")
    return candidate.fit_period, candidate.fit_dataset_version


def register_selected(
    session: Session, result: ExperimentResult, tracker: Tracker, *, mirror: bool = True
) -> int:
    selected = result.candidate(result.selected)
    if selected.run_id is None or selected.final_model is None:
        raise ValueError("the selected model has no tracked final run")
    fit, dataset_version = _fit(selected)
    new = NewModelVersion(
        name=selected.name,
        version=_version(selected.run_id),
        algorithm=selected.model_type,
        description=(
            f"{selected.model_type} selected on {result.selection_metric}; "
            f"split {result.split['split']}"
        ),
        feature_set_version=result.feature_set_version,
        dataset_version=dataset_version,
        training_data_start=datetime.fromisoformat(fit["start"]).date(),
        training_data_end=datetime.fromisoformat(fit["end"]).date(),
        competitions=competitions(session, selected.fit_competitions),
        hyperparameters={
            **selected.final_model.params(),
            **selected.params,
            "model_type": selected.model_type,
        },
        metrics={
            "validation": selected.validation_metrics,
            "cross_validation": selected.cv_metrics or {},
            "selection_metric": result.selection_metric,
            "test": selected.test_metrics or {},
            "mlflow_run_id": selected.run_id,
        },
        trained_at=result.finished_at,
        artifact_uri=tracker.artifact_uri(selected.run_id),
    )
    return _register(session, new, tracker, selected.run_id, mirror=mirror).id


def register_ensemble(
    session: Session,
    ensemble: EnsembleResult,
    experiment: ExperimentResult,
    tracker: Tracker,
    method: str | None = None,
    *,
    mirror: bool = True,
) -> int:
    """Register an ensemble (default: the best ensemble method on forward validation)
    as its own inactive model version, named `ensemble_<method>`."""
    entry = ensemble.candidate(method or ensemble.best_ensemble)
    if entry.run_id is None:
        raise ValueError("the ensemble has no tracked run")
    fit, dataset_version = _fit(experiment.candidate(ensemble.members[0]))
    new = NewModelVersion(
        name=f"ensemble_{entry.method}",
        version=_version(entry.run_id),
        algorithm="EnsembleModel",
        description=(
            f"ensemble ({entry.method}) of {', '.join(ensemble.members)}; combiner fitted on "
            f"out-of-fold validation predictions; recommended on forward validation: "
            f"{ensemble.recommendation['recommended']}; split {experiment.split['split']}"
        ),
        feature_set_version=experiment.feature_set_version,
        dataset_version=dataset_version,
        training_data_start=datetime.fromisoformat(fit["start"]).date(),
        training_data_end=datetime.fromisoformat(fit["end"]).date(),
        competitions=competitions(
            session, experiment.candidate(ensemble.members[0]).fit_competitions
        ),
        hyperparameters={
            **entry.combiner.params(),
            "model_type": "EnsembleModel",
            "members": {
                name: {
                    "params": experiment.candidate(name).params,
                    "algorithm": experiment.candidate(name).model_type,
                    "mlflow_run_id": experiment.candidate(name).run_id,
                }
                for name in ensemble.members
            },
        },
        metrics={
            "validation": entry.validation_metrics,
            "selection_metric": f"forward_validation_{SELECTION_METRIC}",
            "forward_validation_ranking": ensemble.ranking,
            "recommendation": ensemble.recommendation,
            "test": entry.test_metrics,
            "mlflow_run_id": entry.run_id,
        },
        trained_at=datetime.now(UTC),
        artifact_uri=tracker.artifact_uri(entry.run_id),
    )
    return _register(session, new, tracker, entry.run_id, mirror=mirror).id


@dataclass(frozen=True)
class LoadedModel:
    model: Predictor
    version: ModelVersion
    metadata: dict[str, Any]


def load_model_version(
    session: Session,
    name: str,
    version: str | None = None,
    *,
    feature_set_version: str | None = None,
    tracking_uri: str | None = None,
) -> LoadedModel:
    """Load the active version of `name` (or `version`) from its artifact.

    `feature_set_version`, when given, must equal the version's own: a model is never
    applied to features it was not trained on."""
    row = ModelRegistry(session).get(name, version)
    if not row.artifact_uri:
        raise ValueError(f"{name} {row.version} has no model artifact")
    if feature_set_version is not None and feature_set_version != row.feature_set_version:
        raise ValueError(
            f"{name} {row.version} was trained on features {row.feature_set_version!r}, "
            f"not {feature_set_version!r}"
        )
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=row.artifact_uri, dst_path=tmp)
        saved = load_model(Path(local), feature_set_version=row.feature_set_version)
    return LoadedModel(saved.model, row, saved.metadata)
