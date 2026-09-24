"""Connectivity checks for the services the API depends on."""

import logging
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from redis import Redis
from sqlalchemy import Engine, text

from app.schemas.health import DependencyCheck

logger = logging.getLogger(__name__)

# Checks run concurrently so /health answers within one deadline even when a
# dependency hangs (e.g. slow DNS for a stopped container). The small fixed pool
# bounds how many stuck probes can pile up; later checks queue and time out.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health-check")


def _run_check(name: str, probe: Callable[[], object]) -> DependencyCheck:
    started = time.perf_counter()
    try:
        probe()
    except Exception as exc:
        # Only the exception type is returned to clients; details may contain hostnames.
        logger.warning("Health check for %s failed: %s", name, exc)
        return DependencyCheck(status="error", error=type(exc).__name__)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return DependencyCheck(status="ok", latency_ms=latency_ms)


def check_database(engine: Engine) -> DependencyCheck:
    def probe() -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    return _run_check("database", probe)


def check_redis(client: Redis) -> DependencyCheck:
    return _run_check("redis", client.ping)


def run_checks(
    checks: Mapping[str, Callable[[], DependencyCheck]], timeout_seconds: float
) -> dict[str, DependencyCheck]:
    """Run checks in parallel; any not finished within the shared deadline is an error."""
    futures = {name: _executor.submit(check) for name, check in checks.items()}
    deadline = time.monotonic() + timeout_seconds

    results: dict[str, DependencyCheck] = {}
    for name, future in futures.items():
        try:
            results[name] = future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError:
            logger.warning("Health check for %s timed out after %.1fs", name, timeout_seconds)
            results[name] = DependencyCheck(status="error", error="Timeout")
    return results
