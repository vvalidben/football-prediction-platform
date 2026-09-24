"""Per-client rate limiting, so one caller cannot exhaust the service.

The API is public and read-only, but asking for a prediction is not cheap: it
builds features and runs a model. A fixed window per client keeps that bounded
without turning the site into something people must log in to use.

**It fails open.** If Redis is unavailable the request is allowed and a warning is
logged: a rate limiter that is down must not take the API down with it.
"""

import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from starlette.types import ASGIApp

Handler = Callable[[Request], Awaitable[Response]]

logger = logging.getLogger(__name__)

KEY_PREFIX = "ratelimit"
#: Buckets are counted separately, so cheap reads and expensive predictions
#: cannot consume each other's allowance.
DEFAULT_BUCKET = "read"
PREDICT_BUCKET = "predict"
#: Never rate limited: the container's own health check uses it.
EXEMPT_PATHS = frozenset({"/health"})


@dataclass(frozen=True)
class Rule:
    requests: int
    window_seconds: float

    def window_of(self, now: float) -> int:
        return int(now // self.window_seconds)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int


class RateLimiter:
    """Fixed-window counters in Redis, one key per client, bucket and window."""

    def __init__(self, redis: Redis, rules: dict[str, Rule]) -> None:
        self._redis = redis
        self._rules = rules
        self._warned = False

    def rule_for(self, bucket: str) -> Rule:
        return self._rules[bucket]

    def check(self, *, client: str, bucket: str, now: float | None = None) -> Decision:
        rule = self.rule_for(bucket)
        now = time.time() if now is None else now
        window = rule.window_of(now)
        key = f"{KEY_PREFIX}:{bucket}:{client}:{window}"
        try:
            pipeline = self._redis.pipeline()
            pipeline.incr(key)
            # Expiry is set every time: cheap, and it cannot leave a key without one.
            pipeline.expire(key, math.ceil(rule.window_seconds) + 1)
            used = int(pipeline.execute()[0])
        except RedisError as exc:
            if not self._warned:
                logger.warning("Rate limiting is disabled while Redis is unavailable: %s", exc)
                self._warned = True
            return Decision(True, rule.requests, rule.requests, 0)
        self._warned = False
        remaining = max(rule.requests - used, 0)
        reset_in = math.ceil((window + 1) * rule.window_seconds - now)
        return Decision(
            allowed=used <= rule.requests,
            limit=rule.requests,
            remaining=remaining,
            retry_after_seconds=max(reset_in, 1),
        )


def bucket_for(method: str, path: str) -> str:
    """Predictions are the expensive path; everything else shares the read bucket."""
    expensive = method == "POST" and path.startswith("/predictions")
    return PREDICT_BUCKET if expensive else DEFAULT_BUCKET


def client_of(request: Request) -> str:
    """The caller's address.

    uvicorn runs with `--proxy-headers`, so behind the bundled nginx this is the
    browser's address rather than the proxy's. Callers that reach the backend port
    directly are identified by their socket address.
    """
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject a client that is over its allowance, with `Retry-After`.

    The limiter is read from application state at request time: it holds the Redis
    client created during startup, which does not exist yet when middleware is added.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None or request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        bucket = bucket_for(request.method, request.url.path)
        decision = limiter.check(client=client_of(request), bucket=bucket)
        if not decision.allowed:
            logger.warning(
                "Rate limit reached: %s %s from %s (%s bucket)",
                request.method,
                request.url.path,
                client_of(request),
                bucket,
            )
            response: Response = JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": (
                        f"Too many requests: at most {decision.limit} "
                        f"{'prediction' if bucket == PREDICT_BUCKET else 'request'}s per "
                        f"{int(limiter.rule_for(bucket).window_seconds)} seconds. "
                        f"Try again in {decision.retry_after_seconds}s."
                    )
                },
            )
            response.headers["Retry-After"] = str(decision.retry_after_seconds)
        else:
            response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        return response
