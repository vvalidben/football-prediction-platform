"""Checks against real PostgreSQL and Redis instances (see README: "Running tests")."""

import pytest
from fastapi.testclient import TestClient

from app.cache.redis import create_redis_client
from app.core.config import Settings
from app.db.session import create_db_engine
from app.main import create_app
from app.services.health import check_database, check_redis

pytestmark = pytest.mark.integration


@pytest.fixture
def live_settings() -> Settings:
    return Settings()  # values come from the environment / .env


def test_postgres_is_reachable(live_settings: Settings) -> None:
    engine = create_db_engine(live_settings)
    try:
        assert check_database(engine).status == "ok"
    finally:
        engine.dispose()


def test_redis_is_reachable(live_settings: Settings) -> None:
    client = create_redis_client(live_settings)
    try:
        assert check_redis(client).status == "ok"
        client.set("integration-test:key", "value", ex=10)
        assert client.get("integration-test:key") == "value"
    finally:
        client.close()


def test_health_endpoint_against_live_services(live_settings: Settings) -> None:
    with TestClient(create_app(live_settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
