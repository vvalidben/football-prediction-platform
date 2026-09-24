"""Declarative base for ORM models. Alembic autogenerate reads `Base.metadata`."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from sqlalchemy import Date, DateTime, MetaData, Numeric, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase

# Deterministic constraint names keep Alembic migrations stable across environments.
# Multi-column constraints include every column so names never collide; SQLAlchemy
# shortens names beyond PostgreSQL's 63-character limit with a stable hash suffix.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Defaults for plain annotations: unbounded text, timezone-aware timestamps, JSONB.
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        str: Text(),
        datetime: DateTime(timezone=True),
        date: Date(),
        Decimal: Numeric(),
        dict[str, Any]: JSONB(),
    }
