"""Pytest plugin with fixtures for tests against a real PostgreSQL database.

A dedicated `<POSTGRES_DB>_test` database is created once per session and built by
running the Alembic migrations, so tests exercise the real schema. Each test runs
inside a transaction that is rolled back, keeping tests isolated and fast.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


def alembic_config(connection: object) -> Config:
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.attributes["connection"] = connection
    return config


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    settings = Settings()
    test_database = f"{settings.postgres_db}_test"
    admin_engine = create_engine(
        settings.database_url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    with admin_engine.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{test_database}" WITH (FORCE)'))
        connection.execute(text(f'CREATE DATABASE "{test_database}"'))

    engine = create_engine(settings.database_url.set(database=test_database))
    with engine.begin() as connection:
        command.upgrade(alembic_config(connection), "head")

    yield engine

    engine.dispose()
    with admin_engine.connect() as connection:
        connection.execute(text(f'DROP DATABASE IF EXISTS "{test_database}" WITH (FORCE)'))
    admin_engine.dispose()


@pytest.fixture
def session(db_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back after the test.

    `create_savepoint` lets tests call `session.commit()` or recover from an expected
    IntegrityError with `session.begin_nested()` without ending the outer transaction.
    """
    with db_engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as db_session:
            yield db_session
        transaction.rollback()


@pytest.fixture
def session_factory(db_engine: Engine) -> Iterator[sessionmaker[Session]]:
    """A session factory whose sessions all share one rolled-back transaction.

    For code that opens its own sessions (e.g. ingestion). Each `session.begin()`
    becomes a savepoint, so commits inside the code under test are undone at the end.
    """
    with db_engine.connect() as connection:
        transaction = connection.begin()
        yield sessionmaker(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        transaction.rollback()
