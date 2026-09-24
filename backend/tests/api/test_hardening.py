"""Production hardening: request ids, rate limits, caching, headers, exposure."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from redis import Redis
from redis.exceptions import RedisError

from app.api.middleware import REQUEST_ID_HEADER, SECURITY_HEADERS
from app.api.rate_limit import (
    DEFAULT_BUCKET,
    PREDICT_BUCKET,
    RateLimiter,
    Rule,
    bucket_for,
)
from app.cache.responses import ResponseCache
from app.core.config import Settings
from app.main import create_app


class FakeRedis:
    """Just enough Redis for the limiter and the cache, with a failure switch."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.fail = False
        self.sets = 0

    # Rate limiter ------------------------------------------------------------
    def pipeline(self) -> "FakeRedis":
        self._queued: list[tuple[str, tuple[Any, ...]]] = []
        return self

    def incr(self, key: str) -> None:
        self._queued.append(("incr", (key,)))

    def expire(self, key: str, seconds: int) -> None:
        self._queued.append(("expire", (key, seconds)))

    def execute(self) -> list[Any]:
        if self.fail:
            raise RedisError("unavailable")
        results = []
        for name, args in self._queued:
            if name == "incr":
                self.store[args[0]] = int(self.store.get(args[0], 0)) + 1
                results.append(self.store[args[0]])
            else:
                results.append(True)
        return results

    # Cache -------------------------------------------------------------------
    def get(self, key: str) -> Any:
        if self.fail:
            raise RedisError("unavailable")
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.fail:
            raise RedisError("unavailable")
        self.sets += 1
        self.store[key] = value

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


def settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "environment": "test",
        "postgres_host": "db.invalid",
        "postgres_user": "test_user",
        "postgres_password": "test-password",
        "postgres_db": "test_db",
        "redis_host": "redis.invalid",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})


@pytest.fixture
def app_and_redis() -> tuple[FastAPI, FakeRedis]:
    """The real app, with fake Redis and a health check that needs no services."""
    config = settings()
    app = create_app(config)
    fake = FakeRedis()
    app.state.settings = config
    app.state.engine = MagicMock()
    app.state.session_factory = MagicMock()
    app.state.redis = MagicMock(spec=Redis)
    app.state.rate_limiter = RateLimiter(
        fake,  # type: ignore[arg-type]
        {
            DEFAULT_BUCKET: Rule(config.rate_limit_requests, config.rate_limit_window_seconds),
            PREDICT_BUCKET: Rule(2, config.rate_limit_window_seconds),
        },
    )
    app.state.response_cache = ResponseCache(fake)  # type: ignore[arg-type]

    # A cheap stand-in for the real prediction endpoint: this exercises the
    # middleware, not the model stack behind it.
    @app.post("/predictions/echo")
    def echo() -> dict[str, bool]:
        return {"ok": True}

    return app, fake


@pytest.fixture
def client(app_and_redis: tuple[FastAPI, FakeRedis]) -> TestClient:
    app, _ = app_and_redis
    return TestClient(app, raise_server_exceptions=False)


class TestRequestId:
    def test_every_response_carries_an_id(self, client: TestClient) -> None:
        response = client.get("/openapi.json")

        assert response.headers[REQUEST_ID_HEADER]

    def test_a_callers_id_is_echoed_so_traces_line_up(self, client: TestClient) -> None:
        response = client.get("/openapi.json", headers={REQUEST_ID_HEADER: "abc-123"})

        assert response.headers[REQUEST_ID_HEADER] == "abc-123"

    def test_a_forged_id_is_replaced_rather_than_logged(self, client: TestClient) -> None:
        response = client.get("/openapi.json", headers={REQUEST_ID_HEADER: "a\nFAKE LOG LINE"})

        assert response.headers[REQUEST_ID_HEADER] != "a\nFAKE LOG LINE"
        assert "\n" not in response.headers[REQUEST_ID_HEADER]

    def test_an_overlong_id_is_truncated(self, client: TestClient) -> None:
        response = client.get("/openapi.json", headers={REQUEST_ID_HEADER: "x" * 500})

        assert len(response.headers[REQUEST_ID_HEADER]) == 64

    def test_the_request_is_logged_once_with_its_outcome(
        self, client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO", logger="app.access"):
            client.get("/openapi.json")

        lines = [record.getMessage() for record in caplog.records]
        assert sum("GET /openapi.json ->" in line for line in lines) == 1
        assert any("200" in line and "request_id=" in line for line in lines)

    def test_an_unhandled_error_returns_the_id_and_nothing_else(
        self, app_and_redis: tuple[FastAPI, FakeRedis]
    ) -> None:
        app, _ = app_and_redis

        @app.get("/boom")
        def boom() -> None:
            raise RuntimeError("database password is hunter2")

        response = TestClient(app, raise_server_exceptions=False).get("/boom")

        assert response.status_code == 500
        body = response.json()
        assert body["detail"] == "Internal server error"
        assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
        assert "hunter2" not in response.text


class TestSecurityHeaders:
    def test_responses_cannot_be_sniffed_or_framed(self, client: TestClient) -> None:
        headers = client.get("/openapi.json").headers

        for header, value in SECURITY_HEADERS.items():
            assert headers[header] == value

    def test_a_route_may_still_choose_its_own_caching(
        self, app_and_redis: tuple[FastAPI, FakeRedis]
    ) -> None:
        app, _ = app_and_redis

        @app.get("/cacheable")
        def cacheable(response: Response) -> dict[str, str]:
            response.headers["Cache-Control"] = "public, max-age=30"
            return {"ok": "yes"}

        result = TestClient(app).get("/cacheable")

        assert result.headers["Cache-Control"] == "public, max-age=30"


class TestRateLimiting:
    def test_a_client_over_the_limit_is_refused_with_retry_after(self, client: TestClient) -> None:
        for _ in range(2):
            assert client.post("/predictions/echo").status_code != 429

        response = client.post("/predictions/echo")

        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) >= 1
        assert "Too many requests" in response.json()["detail"]

    def test_cheap_reads_are_not_spent_by_expensive_predictions(self, client: TestClient) -> None:
        for _ in range(3):
            client.post("/predictions/echo")

        # The read bucket has its own, much larger allowance.
        assert client.get("/openapi.json").status_code == 200

    def test_the_remaining_allowance_is_reported(self, client: TestClient) -> None:
        first = client.post("/predictions/echo")

        assert first.headers["X-RateLimit-Limit"] == "2"
        assert first.headers["X-RateLimit-Remaining"] == "1"

    def test_health_is_never_rate_limited(self, client: TestClient) -> None:
        """The container's own health check uses it; throttling it kills the container."""
        assert bucket_for("GET", "/health") == DEFAULT_BUCKET  # sanity
        for _ in range(5):
            client.post("/predictions/echo")

        assert "X-RateLimit-Limit" not in client.get("/health").headers

    def test_the_api_stays_up_when_redis_is_down(
        self, client: TestClient, app_and_redis: tuple[FastAPI, FakeRedis]
    ) -> None:
        """A broken limiter must not become a broken API."""
        _, fake = app_and_redis
        fake.fail = True

        for _ in range(5):
            assert client.post("/predictions/echo").status_code != 429

    def test_predictions_are_the_expensive_bucket(self) -> None:
        assert bucket_for("POST", "/predictions/match") == PREDICT_BUCKET
        assert bucket_for("POST", "/predictions/batch") == PREDICT_BUCKET
        assert bucket_for("GET", "/predictions/history") == DEFAULT_BUCKET
        assert bucket_for("GET", "/matches") == DEFAULT_BUCKET


class TestResponseCache:
    def test_a_second_call_is_served_from_the_cache(self) -> None:
        fake = FakeRedis()
        cache = ResponseCache(fake)  # type: ignore[arg-type]
        calls = []

        def build() -> dict[str, int]:
            calls.append(1)
            return {"value": len(calls)}

        first = cache.get_or_set("k", 30, build)
        second = cache.get_or_set("k", 30, build)

        assert first == second == {"value": 1}
        assert len(calls) == 1

    def test_a_zero_ttl_turns_caching_off(self) -> None:
        fake = FakeRedis()
        cache = ResponseCache(fake)  # type: ignore[arg-type]

        cache.get_or_set("k", 0, lambda: {"value": 1})

        assert fake.sets == 0

    def test_a_broken_cache_still_answers(self) -> None:
        fake = FakeRedis()
        fake.fail = True
        cache = ResponseCache(fake)  # type: ignore[arg-type]

        assert cache.get_or_set("k", 30, lambda: {"value": 7}) == {"value": 7}

    def test_an_unreadable_entry_is_recomputed(self) -> None:
        fake = FakeRedis()
        fake.store["cache:k"] = "{not json"
        cache = ResponseCache(fake)  # type: ignore[arg-type]

        assert cache.get_or_set("k", 30, lambda: {"value": 7}) == {"value": 7}


class TestProductionExposure:
    def test_docs_and_schema_are_public_outside_production(self) -> None:
        app = create_app(settings(environment="staging"))

        assert app.docs_url == "/docs"
        assert app.openapi_url == "/openapi.json"

    def test_neither_docs_nor_schema_are_served_in_production(self) -> None:
        """The schema lists every endpoint and parameter: it goes with the docs."""
        app = create_app(settings(environment="production"))
        app.state.settings = settings(environment="production")
        app.state.redis = MagicMock(spec=Redis)

        client = TestClient(app, raise_server_exceptions=False)

        assert app.docs_url is None and app.openapi_url is None
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    def test_a_secret_never_reaches_a_response(self) -> None:
        secret = "super-secret-value"  # noqa: S105 - the point of the test
        config = settings(postgres_password=secret)

        assert secret not in str(config)
        assert secret not in repr(config)
        assert config.postgres_password.get_secret_value() == secret
