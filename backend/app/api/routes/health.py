from functools import partial

from fastapi import APIRouter, Response, status

from app.api.dependencies import EngineDep, RedisDep, SettingsDep
from app.schemas.health import HealthResponse
from app.services.health import check_database, check_redis, run_checks

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
    summary="Service and dependency health",
)
def health(
    response: Response, settings: SettingsDep, engine: EngineDep, redis: RedisDep
) -> HealthResponse:
    """Report API status plus PostgreSQL and Redis connectivity.

    Returns 200 when every dependency is reachable, otherwise 503 with the same body.
    """
    checks = run_checks(
        {
            "database": partial(check_database, engine),
            "redis": partial(check_redis, redis),
        },
        timeout_seconds=settings.health_check_timeout_seconds,
    )
    healthy = all(check.status == "ok" for check in checks.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if healthy else "degraded",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        checks=checks,
    )
