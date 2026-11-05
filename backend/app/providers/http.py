"""HTTP plumbing shared by provider adapters: timeouts, retries and rate limiting.

Adapters describe *what* to fetch; this module decides *how hard to try*. Every
failure leaves here as a `ProviderError`, never as an httpx exception.
"""

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx2 as httpx

from app.providers.errors import (
    AuthenticationError,
    MalformedResponseError,
    ProviderError,
    ProviderRequestError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

Sleep = Callable[[float], None]
Clock = Callable[[], float]
ResponseCheck = Callable[[Any], None]


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for retryable errors."""

    max_attempts: int = 4
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    #: Upper bound for a server-requested wait (Retry-After) before giving up on it.
    max_retry_after_seconds: float = 120.0

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after failed attempt number `attempt` (1-based)."""
        backoff = min(self.max_delay_seconds, self.base_delay_seconds * 2.0 ** (attempt - 1))
        backoff *= random.uniform(0.5, 1.0)  # noqa: S311 - jitter, not cryptography
        if retry_after is not None:
            return max(backoff, min(retry_after, self.max_retry_after_seconds))
        return backoff


class RateLimiter:
    """Spaces requests evenly so a per-minute quota is never exceeded client-side."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            self._sleep(wait)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After") or response.headers.get("X-RequestCounter-Reset")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # HTTP-date form; fall back to backoff


class JsonHttpClient:
    """GETs JSON with timeouts, bounded retries and client-side rate limiting."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 10.0,
        retry_policy: RetryPolicy | None = None,
        rate_limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self._provider = provider
        self._retry = retry_policy or RetryPolicy()
        self._rate_limiter = rate_limiter
        self._sleep = sleep
        #: Requests actually sent, retries included (what counts against a quota).
        self.requests_sent = 0
        #: Headers of the most recent response, e.g. for quota headers.
        self.last_headers: Mapping[str, str] = {}
        self._client = httpx.Client(
            base_url=base_url,
            headers=dict(headers or {}),
            timeout=timeout_seconds,
            transport=transport,
        )

    def get_json(
        self,
        path: str,
        params: Mapping[str, str | int] | None = None,
        check: ResponseCheck | None = None,
    ) -> Any:
        """Return the decoded JSON body of `GET path`.

        `check` may inspect the body and raise a `ProviderError` for errors a provider
        reports inside a successful response; retryable ones are retried here too.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                body = self._attempt(path, params)
                if check is not None:
                    check(body)
                return body
            except ProviderError as error:
                if not error.retryable or attempt >= self._retry.max_attempts:
                    raise
                delay = self._retry.delay(attempt, getattr(error, "retry_after", None))
                logger.warning(
                    "%s GET %s failed (attempt %d/%d): %s; retrying in %.1fs",
                    self._provider,
                    path,
                    attempt,
                    self._retry.max_attempts,
                    error,
                    delay,
                )
                self._sleep(delay)

    def _attempt(self, path: str, params: Mapping[str, str | int] | None) -> Any:
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        self.requests_sent += 1
        try:
            response = self._client.get(path, params=dict(params or {}))
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"timed out calling {path}", provider=self._provider
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"network error calling {path}: {type(exc).__name__}", provider=self._provider
            ) from exc

        self.last_headers = dict(response.headers)
        self._raise_for_status(path, response)
        try:
            return response.json()
        except ValueError as exc:
            raise MalformedResponseError(
                f"{path} returned a body that is not valid JSON", provider=self._provider
            ) from exc

    def _raise_for_status(self, path: str, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        message = f"{path} returned HTTP {status}"
        if status == 429:
            raise RateLimitError(
                message, provider=self._provider, retry_after=_retry_after(response)
            )
        if status in (401, 403):
            raise AuthenticationError(message, provider=self._provider)
        if status >= 500:
            raise ProviderUnavailableError(message, provider=self._provider)
        raise ProviderRequestError(message, provider=self._provider)

    def close(self) -> None:
        self._client.close()
