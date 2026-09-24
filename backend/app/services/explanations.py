"""Explanations of stored predictions, and what a model relies on in general.

A stored prediction keeps the feature snapshot it was made from, so an explanation
is reproducible: the same snapshot and the same model version give the same
contributions, whatever has happened since. Nothing is recomputed from today's data.
"""

import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.models import FeatureSnapshot, Match, ModelVersion, Prediction, Team
from app.registry.loading import ArtifactUnavailableError, ModelCache
from app.schemas.explanations import (
    ContributionRead,
    ExplanationRead,
    ImportanceEntryRead,
    ImportanceRead,
    TargetExplanationRead,
)
from app.services.predictions import PredictionError

if TYPE_CHECKING:  # the serving image imports the model libraries lazily
    from app.ml.explain import TargetExplanation
    from app.ml.predictors import Predictor

logger = logging.getLogger(__name__)

#: Contributions and importances are long tails; the rest is summed into "other".
DEFAULT_TOP = 12


def explain_prediction(
    session: Session, prediction_id: int, *, cache: ModelCache, top: int = DEFAULT_TOP
) -> ExplanationRead:
    """Feature contributions behind one stored prediction."""
    # Imported lazily, as in the prediction service: the API image installs the
    # serving extra, and these modules pull in numpy and scikit-learn.
    from app.ml.dataset import Dataset, Example
    from app.ml.explain import CAVEAT, ExplanationUnavailableError, explain

    row = session.get(Prediction, prediction_id)
    if row is None:
        raise PredictionError(f"prediction {prediction_id} was not found", status_code=404)
    match = session.get(Match, row.match_id)
    model_version = session.get(ModelVersion, row.model_version_id)
    if match is None or model_version is None:  # pragma: no cover - foreign keys enforce this
        raise PredictionError(f"prediction {prediction_id} is incomplete", status_code=404)
    snapshot = (
        None
        if row.feature_snapshot_id is None
        else session.get(FeatureSnapshot, row.feature_snapshot_id)
    )
    if snapshot is None:
        raise PredictionError(
            f"prediction {prediction_id} was stored without its feature snapshot, so the "
            "features behind it cannot be recovered",
            status_code=409,
        )
    model = _load(model_version, cache)
    data = Dataset(
        (
            Example(
                match_id=match.id,
                kickoff_at=match.kickoff_at,
                season_id=match.season_id,
                competition_id=match.competition_id,
                features=snapshot.features,
                # Targets are unused when explaining a prediction.
                home_goals=match.home_score or 0,
                away_goals=match.away_score or 0,
            ),
        ),
        snapshot.feature_set_version,
    )
    try:
        explanation = explain(model, data, 0)
    except ExplanationUnavailableError as exc:
        raise PredictionError(str(exc), status_code=409) from exc

    logger.info(
        "Explained prediction %s (%s %s) with %s",
        prediction_id,
        model_version.name,
        model_version.version,
        explanation.method,
    )
    return ExplanationRead(
        prediction_id=row.id,
        match_id=match.id,
        kickoff_at=match.kickoff_at,
        home_team=_team_name(session, match.home_team_id),
        away_team=_team_name(session, match.away_team_id),
        model_id=model_version.id,
        model_name=model_version.name,
        model_version=model_version.version,
        algorithm=model_version.algorithm,
        method=explanation.method,
        method_detail=explanation.method_detail,
        baseline_description=explanation.baseline_description,
        features_used=explanation.features_used,
        targets=[_target(target, top) for target in explanation.targets],
        caveat=CAVEAT,
    )


def model_importance(
    session: Session, model_version: ModelVersion, *, cache: ModelCache, top: int = DEFAULT_TOP
) -> ImportanceRead:
    """What this model version relies on across all matches, not for one prediction."""
    from app.ml.explain import CAVEAT, ExplanationUnavailableError, importance

    model = _load(model_version, cache)
    try:
        report = importance(model)
    except ExplanationUnavailableError as exc:
        raise PredictionError(str(exc), status_code=409) from exc
    shown, rest = report.entries[:top], report.entries[top:]
    return ImportanceRead(
        model_id=model_version.id,
        model_name=model_version.name,
        model_version=model_version.version,
        algorithm=model_version.algorithm,
        method=report.method,
        method_detail=report.method_detail,
        entries=[
            ImportanceEntryRead(
                feature=entry.feature,
                label=entry.label,
                group=entry.group,
                importance=entry.importance,
            )
            for entry in shown
        ],
        other_features=len(rest),
        other_importance=sum(entry.importance for entry in rest),
        caveat=CAVEAT,
    )


def _load(model_version: ModelVersion, cache: ModelCache) -> "Predictor":
    try:
        model, _ = cache.get(model_version)
    except ArtifactUnavailableError as exc:
        raise PredictionError(str(exc), status_code=503) from exc
    return model


def _target(target: "TargetExplanation", top: int) -> TargetExplanationRead:
    shown, rest = target.contributions[:top], target.contributions[top:]
    return TargetExplanationRead(
        key=target.key,
        label=target.label,
        unit=target.unit,
        baseline=target.baseline,
        predicted=target.predicted,
        output_unit=target.output_unit,
        baseline_output=target.baseline_output,
        predicted_output=target.predicted_output,
        contributions=[
            ContributionRead(
                feature=item.feature,
                label=item.label,
                group=item.group,
                value=item.value,
                contribution=item.contribution,
                direction="positive" if item.contribution >= 0 else "negative",
            )
            for item in shown
        ],
        other_contribution=sum(item.contribution for item in rest),
        other_features=len(rest),
    )


def _team_name(session: Session, team_id: int) -> str:
    team = session.get(Team, team_id)
    return team.name if team else f"team {team_id}"
