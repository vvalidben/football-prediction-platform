"""Application factory. Run with `uvicorn app.main:create_app --factory`."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.middleware import RequestContextMiddleware, request_id_of
from app.api.rate_limit import (
    DEFAULT_BUCKET,
    PREDICT_BUCKET,
    RateLimiter,
    RateLimitMiddleware,
    Rule,
)
from app.api.routes import catalog, health, jobs, models, predictions
from app.api.routes import status as status_routes
from app.cache.redis import create_redis_client
from app.cache.responses import ResponseCache
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory
from app.registry.loading import ModelCache
from app.services.predictions import PredictionService

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Football match predictions from leak-free features and versioned models.

Every prediction is derived from one exact-score distribution, so the 1X2,
over/under and both-teams-to-score figures always agree with each other. Each
response names the model version that produced it and how fresh the underlying
data was.

Predictions are stored: `POST /predictions/match` returns an `id` that
`GET /predictions/{prediction_id}` returns unchanged.
"""

OPENAPI_TAGS = [
    {"name": "health", "description": "Service and dependency status."},
    {"name": "catalog", "description": "Competitions, teams and matches."},
    {"name": "predictions", "description": "Producing and reading predictions."},
    {"name": "models", "description": "Registered model versions and their metrics."},
    {"name": "jobs", "description": "Status of the scheduled background jobs."},
    {"name": "status", "description": "Data coverage, freshness and providers."},
]


def _rules(settings: Settings) -> dict[str, Rule]:
    return {
        DEFAULT_BUCKET: Rule(settings.rate_limit_requests, settings.rate_limit_window_seconds),
        PREDICT_BUCKET: Rule(
            settings.rate_limit_predict_requests, settings.rate_limit_window_seconds
        ),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_db_engine(settings)
        redis = create_redis_client(settings)
        app.state.settings = settings
        app.state.engine = engine
        session_factory = create_session_factory(engine)
        app.state.session_factory = session_factory
        app.state.redis = redis
        app.state.model_cache = ModelCache()
        app.state.predictions = PredictionService(session_factory, cache=app.state.model_cache)
        app.state.response_cache = ResponseCache(redis, enabled=settings.cache_enabled)
        app.state.rate_limiter = (
            RateLimiter(redis, _rules(settings)) if settings.rate_limit_enabled else None
        )
        logger.info(
            "Starting %s v%s (environment=%s)",
            settings.app_name,
            settings.app_version,
            settings.environment,
        )
        try:
            yield
        finally:
            redis.close()
            engine.dispose()
            logger.info("Shutdown complete")

    app = FastAPI(
        title="Football Prediction API",
        description=API_DESCRIPTION,
        version=settings.app_version,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        # Interactive docs are useful locally but should not be public in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        # The schema describes every endpoint and its parameters; it is published
        # with the docs, and withheld with them.
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    if settings.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware)
    # Added last, so it wraps the others: every response gets an id and one log line.
    app.add_middleware(RequestContextMiddleware)

    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["*"],
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request_id_of(request)
        logger.exception(
            "Unhandled error on %s %s [request_id=%s]",
            request.method,
            request.url.path,
            request_id,
        )
        # The message stays generic; the id is what connects it to the logs.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    app.include_router(health.router)
    app.include_router(catalog.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(jobs.router)
    app.include_router(status_routes.router)
    return app
