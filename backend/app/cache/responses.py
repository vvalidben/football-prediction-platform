"""A small Redis cache for endpoints that aggregate a lot and change slowly.

Only used where a slightly stale answer is honest: the data-status and freshness
panels describe imports that run every few minutes, and their queries scan several
tables. Predictions, history and model metrics are never cached — they are records,
and a caller asking for them should see what is stored now.

**It fails open.** A cache that cannot be reached is skipped, not fatal: the
endpoint computes its answer as it did before.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

KEY_PREFIX = "cache"


class ResponseCache:
    """Cache JSON-serialisable results under a key, for a few seconds."""

    def __init__(self, redis: Redis, *, enabled: bool = True) -> None:
        self._redis = redis
        self._enabled = enabled

    def get_or_set(self, key: str, ttl_seconds: float, build: Callable[[], Any]) -> Any:
        """The cached value for `key`, or `build()`'s result, stored for `ttl_seconds`."""
        if not self._enabled or ttl_seconds <= 0:
            return build()
        full_key = f"{KEY_PREFIX}:{key}"
        try:
            cached = self._redis.get(full_key)
        except RedisError as exc:
            logger.warning("Cache read failed for %s: %s", key, exc)
            return build()
        if isinstance(cached, str | bytes):
            try:
                return json.loads(cached)
            except json.JSONDecodeError:  # pragma: no cover - a corrupt entry
                logger.warning("Ignoring an unreadable cache entry for %s", key)
        value = build()
        try:
            self._redis.set(full_key, json.dumps(value, default=str), ex=int(max(ttl_seconds, 1)))
        except (RedisError, TypeError) as exc:
            logger.warning("Cache write failed for %s: %s", key, exc)
        return value

    def invalidate(self, key: str) -> None:
        try:
            self._redis.delete(f"{KEY_PREFIX}:{key}")
        except RedisError as exc:  # pragma: no cover - best effort
            logger.warning("Cache invalidation failed for %s: %s", key, exc)
