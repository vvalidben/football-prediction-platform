"""Data status: synchronisation, coverage, gaps and providers.

Both answers aggregate across several tables and describe imports that run every
few minutes, so they are cached briefly in Redis (`STATUS_CACHE_SECONDS`, 0 to
turn it off). The response says how old the figures are, and a cache that cannot
be reached is skipped rather than fatal.
"""

import logging
from typing import Any

from fastapi import APIRouter, Request, Response

from app.api.dependencies import DbSessionDep
from app.cache.responses import ResponseCache
from app.core.config import Settings
from app.services import analytics

logger = logging.getLogger(__name__)
router = APIRouter(tags=["status"])


def _cached(request: Request, key: str, build: Any) -> Any:
    settings: Settings = request.app.state.settings
    cache: ResponseCache | None = getattr(request.app.state, "response_cache", None)
    if cache is None:
        return build()
    return cache.get_or_set(key, settings.status_cache_seconds, build)


def _cache_header(request: Request, response: Response) -> None:
    settings: Settings = request.app.state.settings
    seconds = int(settings.status_cache_seconds) if settings.cache_enabled else 0
    response.headers["Cache-Control"] = f"public, max-age={seconds}" if seconds else "no-store"


@router.get("/data/status", summary="Synchronisation, coverage and data gaps")
def data_status(request: Request, response: Response, session: DbSessionDep) -> dict[str, Any]:
    """What the platform holds and how current it is.

    Coverage counts matches, statistics and feature snapshots per competition, so a
    gap is visible rather than implied. Open data-quality issues are listed by
    severity and type; providers show when each last ran. `generated_at` is when the
    figures were computed, which may be a few seconds before this request.
    """
    _cache_header(request, response)
    body: dict[str, Any] = _cached(request, "data:status", lambda: analytics.data_status(session))
    return body


@router.get("/data/freshness", summary="How current the data is")
def data_freshness(request: Request, response: Response, session: DbSessionDep) -> dict[str, Any]:
    """The small panel the dashboard shows: last successful import, last job, the
    most recent result, the newest feature snapshot and how many models are active."""
    _cache_header(request, response)
    body: dict[str, Any] = _cached(request, "data:freshness", lambda: analytics.freshness(session))
    return body
