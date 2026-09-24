"""What a worker process shares between tasks: settings, database sessions, the
provider and the model cache.

Each is created once per process and can be replaced in tests, so the tasks
themselves take no infrastructure arguments (Celery arguments must be serialisable).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_db_engine, create_session_factory
from app.providers.base import FootballDataProvider
from app.providers.registry import create_provider

if TYPE_CHECKING:
    from app.registry.loading import ModelCache

_settings: Settings | None = None
_session_factory: sessionmaker[Session] | None = None
_provider_factory: Callable[[Settings], FootballDataProvider] | None = None
_model_cache: "ModelCache | None" = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        # Jobs run long queries on purpose (backfill, feature generation, retraining).
        _session_factory = create_session_factory(
            create_db_engine(settings(), statement_timeout=False)
        )
    return _session_factory


def provider() -> FootballDataProvider:
    """A configured provider; credentials come from settings, never from a task
    argument. Raises if the provider is not configured."""
    factory = _provider_factory or (lambda config: create_provider(config.job_provider, config))
    return factory(settings())


def model_cache() -> "ModelCache":
    global _model_cache
    if _model_cache is None:
        from app.registry.loading import ModelCache

        _model_cache = ModelCache()
    return _model_cache


def configure(
    *,
    new_settings: Settings | None = None,
    new_session_factory: sessionmaker[Session] | None = None,
    new_provider_factory: Callable[[Settings], FootballDataProvider] | None = None,
) -> None:
    """Replace process state (used by tests and by the worker's startup hook)."""
    global _settings, _session_factory, _provider_factory, _model_cache
    if new_settings is not None:
        _settings = new_settings
    if new_session_factory is not None:
        _session_factory = new_session_factory
    if new_provider_factory is not None:
        _provider_factory = new_provider_factory
    _model_cache = None


def reset() -> None:
    global _settings, _session_factory, _provider_factory, _model_cache
    _settings = _session_factory = _provider_factory = _model_cache = None
