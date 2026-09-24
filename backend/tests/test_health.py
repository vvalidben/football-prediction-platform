import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.conftest import make_client


def test_health_ok_when_all_dependencies_reachable(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "football-prediction-api"
    assert body["environment"] == "test"
    assert set(body["checks"]) == {"database", "redis"}
    for check in body["checks"].values():
        assert check["status"] == "ok"
        assert check["latency_ms"] >= 0
        assert check["error"] is None


@pytest.mark.parametrize(
    ("engine_fixture", "redis_fixture", "failed", "error"),
    [
        ("failing_engine", "healthy_redis", "database", "OperationalError"),
        ("healthy_engine", "failing_redis", "redis", "ConnectionError"),
    ],
)
def test_health_degraded_when_a_dependency_fails(
    request: pytest.FixtureRequest,
    app: FastAPI,
    engine_fixture: str,
    redis_fixture: str,
    failed: str,
    error: str,
) -> None:
    engine: MagicMock = request.getfixturevalue(engine_fixture)
    redis: MagicMock = request.getfixturevalue(redis_fixture)

    for client in make_client(app, engine, redis):
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"][failed] == {"status": "error", "latency_ms": None, "error": error}


def test_health_does_not_leak_error_details(
    app: FastAPI, failing_engine: MagicMock, healthy_redis: MagicMock
) -> None:
    for client in make_client(app, failing_engine, healthy_redis):
        response = client.get("/health")

    assert "refused" not in response.text


def test_unhandled_errors_return_generic_500(
    app: FastAPI, healthy_engine: MagicMock, healthy_redis: MagicMock
) -> None:
    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    for client in make_client(app, healthy_engine, healthy_redis):
        response = client.get("/boom")

    assert response.status_code == 500
    body = response.json()
    # The message stays generic; the request id is what ties it to the logs.
    assert body["detail"] == "Internal server error"
    assert "secret internal detail" not in response.text
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert set(body) == {"detail", "request_id"}


def test_health_reports_timeout_for_a_hanging_dependency(
    settings: Settings, healthy_engine: MagicMock, healthy_redis: MagicMock
) -> None:
    release = threading.Event()
    healthy_redis.ping.side_effect = lambda: release.wait(5)
    app = create_app(settings.model_copy(update={"health_check_timeout_seconds": 0.2}))

    try:
        for client in make_client(app, healthy_engine, healthy_redis):
            started = time.perf_counter()
            response = client.get("/health")
            elapsed = time.perf_counter() - started
    finally:
        release.set()

    assert response.status_code == 503
    assert response.json()["checks"]["redis"] == {
        "status": "error",
        "latency_ms": None,
        "error": "Timeout",
    }
    assert response.json()["checks"]["database"]["status"] == "ok"
    assert elapsed < 2
