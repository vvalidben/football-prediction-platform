"""Producing and reading predictions."""

import datetime as dt
import logging
import time
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Path, Query, Request, status

from app.api.dependencies import DbSessionDep
from app.schemas.explanations import ExplanationRead
from app.schemas.history import PredictionHistoryPage
from app.schemas.predictions import (
    BatchItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRead,
    PredictionRequest,
    StoredPredictionPage,
)
from app.services.explanations import DEFAULT_TOP, explain_prediction
from app.services.history import HistoryFilters, history, summary
from app.services.predictions import (
    PredictionError,
    PredictionService,
    list_stored,
    resolve_match,
    select_model,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"], prefix="/predictions")

ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "Match, team, competition or model not found"},
    status.HTTP_409_CONFLICT: {"description": "Ambiguous match, or no single active model"},
    # 422 by number: the Starlette constant was renamed between versions.
    422: {"description": "Features could not be computed"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The model artifact is unavailable"},
}


def service(request: Request) -> PredictionService:
    predictions: PredictionService = request.app.state.predictions
    return predictions


@router.post(
    "/match",
    response_model=PredictionRead,
    status_code=status.HTTP_200_OK,
    summary="Predict one match",
    responses=ERRORS,
)
def predict_match(
    request: Request, body: PredictionRequest, session: DbSessionDep
) -> PredictionRead:
    """Predict a match, identified by `match_id` or by team names (with an optional
    competition and date).

    Features are computed from information available now (or at kickoff for a match
    already played), the active model version is applied, and the prediction is
    stored and returned. Asking again within the same minute returns the stored
    prediction with `reused: true`; `refresh: true` forces a new one.
    """
    started = time.perf_counter()
    now = datetime.now(UTC)
    predictions = service(request)
    try:
        match = resolve_match(session, body, now=now)
        model_version = select_model(session, body.model_name)
        outcome = predictions.predict(
            session, [match], model_version, refresh=body.refresh, now=now
        )[0]
        session.commit()
    except PredictionError as exc:
        session.rollback()
        logger.info("Prediction refused: %s", exc.detail)
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    logger.info(
        "Predicted match %s with %s %s in %.0f ms (reused=%s)",
        match.id,
        model_version.name,
        model_version.version,
        (time.perf_counter() - started) * 1000,
        not outcome.created,
    )
    return outcome.prediction


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict several matches",
    responses=ERRORS,
)
def predict_batch(
    request: Request, body: BatchPredictionRequest, session: DbSessionDep
) -> BatchPredictionResponse:
    """Predict up to 50 matches in one call. Matches that cannot be resolved are
    reported per item; the rest are predicted. Features for every match are computed
    in one pass."""
    now = datetime.now(UTC)
    predictions = service(request)
    try:
        model_version = select_model(session, body.model_name)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    resolved: list[tuple[int, object]] = []
    items: dict[int, BatchItem] = {}
    for index, selector in enumerate(body.matches):
        try:
            resolved.append((index, resolve_match(session, selector, now=now)))
        except PredictionError as exc:
            items[index] = BatchItem(index=index, error=exc.detail)
    if resolved:
        try:
            outcomes = predictions.predict(
                session,
                [match for _, match in resolved],  # type: ignore[misc]
                model_version,
                refresh=body.refresh,
                now=now,
            )
            session.commit()
        except PredictionError as exc:
            session.rollback()
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        for (index, _), outcome in zip(resolved, outcomes, strict=True):
            items[index] = BatchItem(index=index, prediction=outcome.prediction)
    ordered = [items[index] for index in sorted(items)]
    failed = sum(1 for item in ordered if item.error)
    logger.info(
        "Batch prediction: %s requested, %s predicted, %s failed",
        len(body.matches),
        len(ordered) - failed,
        failed,
    )
    return BatchPredictionResponse(
        requested=len(body.matches),
        predicted=len(ordered) - failed,
        failed=failed,
        items=ordered,
    )


@router.get(
    "/history",
    response_model=PredictionHistoryPage,
    summary="Prediction history, with results and errors",
)
def prediction_history(
    session: DbSessionDep,
    competition_id: Annotated[int | None, Query(ge=1)] = None,
    season_id: Annotated[int | None, Query(ge=1)] = None,
    team_id: Annotated[int | None, Query(ge=1, description="Home or away")] = None,
    model_name: Annotated[str | None, Query(max_length=120)] = None,
    model_version_id: Annotated[int | None, Query(ge=1)] = None,
    from_date: Annotated[dt.date | None, Query(description="Earliest kickoff date")] = None,
    to_date: Annotated[dt.date | None, Query(description="Latest kickoff date (inclusive)")] = None,
    evaluated: Annotated[
        bool | None, Query(description="true: only scored predictions; false: only pending")
    ] = None,
    outcome_correct: Annotated[
        bool | None, Query(description="true: the most likely outcome happened")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PredictionHistoryPage:
    """Every stored prediction, newest kickoff first, with the result and the errors
    once the match has been played and evaluated.

    Filter by competition, season, team, model or kickoff date. The summary covers
    every prediction matching the filters, not just the page returned, and its
    averages are over evaluated rows only: a match that has not been played has no
    error to average.
    """
    filters = HistoryFilters(
        competition_id=competition_id,
        season_id=season_id,
        team_id=team_id,
        model_name=model_name,
        model_version_id=model_version_id,
        from_date=from_date,
        to_date=to_date,
        evaluated=evaluated,
        outcome_correct=outcome_correct,
    )
    total, items = history(session, filters, limit=limit, offset=offset)
    return PredictionHistoryPage(
        total=total, limit=limit, offset=offset, items=items, summary=summary(session, filters)
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionRead,
    summary="Read a stored prediction",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such prediction"}},
)
def read_prediction(
    request: Request, session: DbSessionDep, prediction_id: Annotated[int, Path(ge=1)]
) -> PredictionRead:
    """A prediction exactly as it was served, including the model version that made
    it and how fresh its data was."""
    try:
        return service(request).read(session, prediction_id)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.get("", response_model=StoredPredictionPage, summary="Recently made predictions")
def list_predictions(
    session: DbSessionDep,
    match_id: Annotated[int | None, Query(ge=1)] = None,
    model_name: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> StoredPredictionPage:
    """Stored predictions, newest first, with the result once the match is played."""
    total, items = list_stored(
        session, match_id=match_id, model_name=model_name, limit=limit, offset=offset
    )
    return StoredPredictionPage(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/{prediction_id}/explanation",
    response_model=ExplanationRead,
    summary="Which features moved this prediction",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "No such prediction"},
        status.HTTP_409_CONFLICT: {
            "description": "This model cannot be explained, or the features are gone"
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "The model artifact is unavailable"},
    },
)
def explain_stored_prediction(
    request: Request,
    session: DbSessionDep,
    prediction_id: Annotated[int, Path(ge=1)],
    top: Annotated[int, Query(ge=1, le=50, description="Contributions to list per target")] = (
        DEFAULT_TOP
    ),
) -> ExplanationRead:
    """The feature contributions behind a stored prediction, per outcome.

    Contributions are computed from the feature snapshot the prediction was made
    from, so they describe that prediction and not today's data. They describe the
    model's behaviour: a contribution is what a feature added to the model's own
    score, not a cause of the result. Models that have no valid method here say so
    (409) rather than return an approximation.
    """
    try:
        return explain_prediction(session, prediction_id, cache=service(request).cache, top=top)
    except PredictionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
