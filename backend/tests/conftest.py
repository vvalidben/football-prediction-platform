import os
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from app.api.dependencies import get_engine, get_redis
from app.core.config import Settings
from app.main import create_app

INTEGRATION_ENABLED = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

pytest_plugins = ["tests.database"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if INTEGRATION_ENABLED:
        return
    skip = pytest.mark.skip(reason="set RUN_INTEGRATION_TESTS=1 with PostgreSQL and Redis running")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings() -> Settings:
    # _env_file=None keeps unit tests independent of a developer's local .env.
    return Settings(
        _env_file=None,
        environment="test",
        postgres_host="db.invalid",
        postgres_user="test_user",
        postgres_password=SecretStr("test-password"),
        postgres_db="test_db",
        redis_host="redis.invalid",
    )


@pytest.fixture
def healthy_engine() -> MagicMock:
    return MagicMock(spec=Engine)


@pytest.fixture
def failing_engine() -> MagicMock:
    engine = MagicMock(spec=Engine)
    engine.connect.side_effect = OperationalError("SELECT 1", {}, Exception("refused"))
    return engine


@pytest.fixture
def healthy_redis() -> MagicMock:
    client = MagicMock(spec=Redis)
    client.ping.return_value = True
    return client


@pytest.fixture
def failing_redis() -> MagicMock:
    client = MagicMock(spec=Redis)
    client.ping.side_effect = RedisConnectionError("refused")
    return client


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


def make_client(app: FastAPI, engine: MagicMock, redis: MagicMock) -> Iterator[TestClient]:
    """Yield a client whose engine and Redis dependencies are replaced by test doubles."""
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_redis] = lambda: redis
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client(
    app: FastAPI, healthy_engine: MagicMock, healthy_redis: MagicMock
) -> Iterator[TestClient]:
    yield from make_client(app, healthy_engine, healthy_redis)
