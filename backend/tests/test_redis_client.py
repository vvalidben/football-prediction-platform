import time

from app.cache.redis import REDIS_MAX_RETRIES, create_redis_client
from app.core.config import Settings
from app.services.health import check_redis


def test_client_uses_bounded_retries(settings: Settings) -> None:
    client = create_redis_client(settings)

    retry = client.get_retry()
    assert retry is not None
    assert retry.get_retries() == REDIS_MAX_RETRIES


def test_unreachable_redis_fails_fast(settings: Settings) -> None:
    # Nothing listens on port 1. Each attempt is capped by the socket timeout, so the
    # whole check must finish within (retries + 1) timeouts plus a little backoff.
    timeout = 0.2
    client = create_redis_client(
        settings.model_copy(
            update={
                "redis_host": "127.0.0.1",
                "redis_port": 1,
                "dependency_timeout_seconds": timeout,
            }
        )
    )

    started = time.perf_counter()
    result = check_redis(client)

    assert result.status == "error"
    assert time.perf_counter() - started < (REDIS_MAX_RETRIES + 1) * timeout + 2
