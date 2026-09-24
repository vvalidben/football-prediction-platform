"""Redis client construction."""

from redis import Redis
from redis.backoff import ExponentialWithJitterBackoff
from redis.retry import Retry

from app.core.config import Settings

# redis-py retries 10 times by default, which can stall a request for close to a
# minute when Redis is down. Fail fast instead and let callers decide what to do.
REDIS_MAX_RETRIES = 2


def create_redis_client(settings: Settings) -> Redis:
    """Create a pooled client. No connection is opened until first use."""
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password.get_secret_value() if settings.redis_password else None,
        socket_connect_timeout=settings.dependency_timeout_seconds,
        socket_timeout=settings.dependency_timeout_seconds,
        retry=Retry(ExponentialWithJitterBackoff(base=0.05, cap=0.5), retries=REDIS_MAX_RETRIES),
        decode_responses=True,
    )
