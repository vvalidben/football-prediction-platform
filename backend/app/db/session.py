"""Database engine and session construction."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings


def create_db_engine(settings: Settings, *, statement_timeout: bool = True) -> Engine:
    """Create the engine. No connection is opened until first use.

    `statement_timeout` asks PostgreSQL to cancel a query that overruns
    `DATABASE_STATEMENT_TIMEOUT_SECONDS`, so a single slow statement cannot hold a
    connection until the pool is exhausted and the whole API stalls. The background
    jobs pass `statement_timeout=False`: a backfill or a retrain runs long queries
    deliberately, and cancelling those would be the bug.
    """
    connect_args: dict[str, object] = {
        "connect_timeout": max(1, round(settings.dependency_timeout_seconds))
    }
    seconds = settings.database_statement_timeout_seconds
    if statement_timeout and seconds > 0:
        milliseconds = int(seconds * 1000)
        connect_args["options"] = (
            f"-c statement_timeout={milliseconds} "
            # A transaction left open by a crashed request holds its locks; this
            # gives it the same deadline.
            f"-c idle_in_transaction_session_timeout={milliseconds * 2}"
        )
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        connect_args=connect_args,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
