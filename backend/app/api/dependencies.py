"""FastAPI dependencies exposing resources created during application startup."""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def get_redis(request: Request) -> Redis:
    client: Redis = request.app.state.redis
    return client


def get_db_session(request: Request) -> Iterator[Session]:
    """Yield a request-scoped session that is always closed afterwards."""
    with request.app.state.session_factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
EngineDep = Annotated[Engine, Depends(get_engine)]
RedisDep = Annotated[Redis, Depends(get_redis)]
DbSessionDep = Annotated[Session, Depends(get_db_session)]
