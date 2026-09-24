"""Retries, timeouts, rate limits and error translation in the shared HTTP client."""

import httpx2 as httpx
import pytest

from app.providers.errors import (
    AuthenticationError,
    MalformedResponseError,
    ProviderRequestError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.providers.http import JsonHttpClient, RateLimiter, RetryPolicy
from tests.providers.fake_api import Responder, body, connection_error, raw, status, timeout

OK = {"ok": True}


def make_client(
    *responders: Responder, max_attempts: int = 4
) -> tuple[JsonHttpClient, list[float], list[httpx.Request]]:
    queue = list(responders)
    seen: list[httpx.Request] = []
    sleeps: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        responder = queue.pop(0) if queue else body(OK)
        return responder(request)

    client = JsonHttpClient(
        provider="test",
        base_url="https://provider.test",
        retry_policy=RetryPolicy(max_attempts=max_attempts, base_delay_seconds=1.0),
        transport=httpx.MockTransport(handle),
        sleep=sleeps.append,
    )
    return client, sleeps, seen


@pytest.mark.parametrize(
    "failure",
    [timeout(), connection_error(), status(500), status(502), status(503), status(504)],
    ids=["timeout", "connection", "500", "502", "503", "504"],
)
def test_transient_failures_are_retried(failure: Responder) -> None:
    client, sleeps, seen = make_client(failure, failure)

    assert client.get_json("/x") == OK
    assert len(seen) == 3
    assert len(sleeps) == 2


def test_backoff_grows_exponentially_with_jitter() -> None:
    client, sleeps, _ = make_client(status(503), status(503), status(503))

    client.get_json("/x")

    # Attempt n waits base * 2^(n-1), scaled by jitter in [0.5, 1.0].
    for attempt, delay in enumerate(sleeps, start=1):
        ceiling = 2.0 ** (attempt - 1)
        assert ceiling * 0.5 <= delay <= ceiling


def test_gives_up_after_max_attempts() -> None:
    client, sleeps, seen = make_client(*[timeout()] * 5, max_attempts=3)

    with pytest.raises(ProviderUnavailableError, match="timed out"):
        client.get_json("/x")
    assert len(seen) == 3
    assert len(sleeps) == 2


def test_rate_limit_honours_retry_after() -> None:
    client, sleeps, _ = make_client(status(429, {"Retry-After": "7"}))

    assert client.get_json("/x") == OK
    assert sleeps[0] >= 7


def test_exhausted_rate_limit_raises_rate_limit_error() -> None:
    client, _, _ = make_client(*[status(429)] * 4)

    with pytest.raises(RateLimitError):
        client.get_json("/x")


@pytest.mark.parametrize(
    ("responder", "error"),
    [
        (status(401), AuthenticationError),
        (status(403), AuthenticationError),
        (status(404), ProviderRequestError),
        (status(422), ProviderRequestError),
        (raw(b"<html>maintenance</html>"), MalformedResponseError),
    ],
    ids=["401", "403", "404", "422", "not-json"],
)
def test_permanent_failures_are_not_retried(responder: Responder, error: type[Exception]) -> None:
    client, sleeps, seen = make_client(responder)

    with pytest.raises(error):
        client.get_json("/x")
    assert len(seen) == 1
    assert sleeps == []


def test_check_hook_errors_are_retried_when_retryable() -> None:
    calls = 0

    def check(_: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("in-body rate limit", retry_after=2)

    client, sleeps, _ = make_client()

    assert client.get_json("/x", check=check) == OK
    assert calls == 2
    assert sleeps[0] >= 2


def test_query_parameters_are_sent() -> None:
    client, _, seen = make_client()

    client.get_json("/fixtures", {"league": 39, "season": 2023})

    assert dict(seen[0].url.params) == {"league": "39", "season": "2023"}


class TestRateLimiter:
    def test_spaces_requests_evenly(self) -> None:
        now = [100.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        limiter = RateLimiter(30, clock=lambda: now[0], sleep=sleep)  # one per 2 s

        for _ in range(3):
            limiter.acquire()

        assert sleeps == [2.0, 2.0]

    def test_no_wait_when_requests_are_already_spaced(self) -> None:
        now = [0.0]
        sleeps: list[float] = []
        limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleeps.append)

        limiter.acquire()
        now[0] += 5
        limiter.acquire()

        assert sleeps == []

    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(0)


def test_retry_after_is_capped() -> None:
    policy = RetryPolicy(max_retry_after_seconds=10)

    assert policy.delay(1, retry_after=3600) == 10


def test_counts_every_request_sent_including_retries() -> None:
    client, _, _ = make_client(status(503), timeout())

    client.get_json("/x")

    assert client.requests_sent == 3
