import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect, text

import app.models  # noqa: F401  - registers every table
from app.core.config import Settings
from app.db.base import Base
from app.db.session import create_db_engine
from tests.database import alembic_config

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "bookmakers",
    "competition_external_ids",
    "competitions",
    "data_ingestion_runs",
    "data_quality_issues",
    "data_sources",
    "feature_snapshots",
    "injuries",
    "job_runs",
    "lineup_players",
    "lineups",
    "match_events",
    "match_external_ids",
    "matches",
    "model_activations",
    "model_versions",
    "odds",
    "player_external_ids",
    "player_match_statistics",
    "players",
    "prediction_over_under",
    "prediction_results",
    "prediction_score_probabilities",
    "predictions",
    "season_external_ids",
    "seasons",
    "standings",
    "team_external_ids",
    "team_match_statistics",
    "teams",
}


@pytest.fixture
def api_settings_for_timeout() -> Settings:
    """The test database, with a deadline short enough to prove inside a test."""
    settings = Settings()
    return settings.model_copy(
        update={
            "postgres_db": f"{settings.postgres_db}_test",
            "database_statement_timeout_seconds": 2.0,
        }
    )


def test_migration_creates_every_table(db_engine: Engine) -> None:
    tables = set(inspect(db_engine).get_table_names()) - {"alembic_version"}

    assert tables == EXPECTED_TABLES


def test_migrated_schema_matches_models(db_engine: Engine) -> None:
    with db_engine.connect() as connection:
        diff = compare_metadata(
            MigrationContext.configure(connection, opts={"compare_type": True}), Base.metadata
        )

    assert diff == [], "models and migrations disagree; generate a new migration"


def test_downgrade_to_base_and_upgrade_again(db_engine: Engine) -> None:
    # PostgreSQL DDL is transactional, so the round trip is rolled back afterwards.
    with db_engine.connect() as connection, connection.begin() as transaction:
        config = alembic_config(connection)

        command.downgrade(config, "base")
        assert set(inspect(connection).get_table_names()) == {"alembic_version"}

        command.upgrade(config, "head")
        assert set(inspect(connection).get_table_names()) >= EXPECTED_TABLES
        transaction.rollback()


def test_every_foreign_key_is_indexed(db_engine: Engine) -> None:
    """Each FK's columns must be the leading columns of some index.

    Unindexed FKs make joins and ON DELETE checks scan the referencing table.
    """
    query = text(
        """
        SELECT c.conrelid::regclass::text AS table_name, c.conname
        FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.connamespace = 'public'::regnamespace
          AND NOT EXISTS (
              SELECT 1 FROM pg_index i
              WHERE i.indrelid = c.conrelid
                AND (i.indkey::int2[])[0:array_length(c.conkey, 1) - 1]
                    @> c.conkey
          )
        ORDER BY 1, 2
        """
    )
    with db_engine.connect() as connection:
        unindexed = connection.execute(query).all()

    assert unindexed == []


def test_the_hot_orderings_are_indexed(db_engine: Engine) -> None:
    """Both were sorts over the whole table before Sprint 16.

    `GET /predictions` orders by `predicted_at DESC, id DESC` on every dashboard
    load, and the reporting query filters `prediction_results.evaluated_at`.
    """
    query = text(
        """
        SELECT indexname, indexdef FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname IN ('ix_predictions_predicted_at', 'ix_prediction_results_evaluated_at')
        """
    )
    with db_engine.connect() as connection:
        found = {name: definition for name, definition in connection.execute(query).all()}

    assert set(found) == {"ix_predictions_predicted_at", "ix_prediction_results_evaluated_at"}
    # Descending, so the newest rows are the first the index returns.
    assert "predicted_at DESC" in found["ix_predictions_predicted_at"]


def test_the_planner_uses_the_prediction_index_for_the_dashboard_query(
    db_engine: Engine,
) -> None:
    """An index nothing plans to use is not an improvement."""
    with db_engine.connect() as connection, connection.begin() as transaction:
        # The planner prefers a sequential scan on a tiny table; this asks what it
        # would do with a realistic one, without inserting 60k rows into the test
        # database.
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            connection.execute(
                text(
                    "EXPLAIN SELECT id FROM predictions "
                    "ORDER BY predicted_at DESC, id DESC LIMIT 20"
                )
            )
            .scalars()
            .all()
        )
        transaction.rollback()

    assert any("ix_predictions_predicted_at" in line for line in plan), plan


def test_api_connections_carry_a_statement_timeout(api_settings_for_timeout: Settings) -> None:
    """A slow query must not be able to hold a connection until the pool empties."""
    engine = create_db_engine(api_settings_for_timeout)
    try:
        with engine.connect() as connection:
            timeout = connection.execute(text("SHOW statement_timeout")).scalar_one()
            idle = connection.execute(text("SHOW idle_in_transaction_session_timeout")).scalar_one()
    finally:
        engine.dispose()

    assert timeout == "2s"
    assert idle == "4s"


def test_a_query_over_the_deadline_is_cancelled(api_settings_for_timeout: Settings) -> None:
    """The timeout is enforced by PostgreSQL, not hoped for."""
    from sqlalchemy.exc import OperationalError

    engine = create_db_engine(api_settings_for_timeout)
    try:
        with engine.connect() as connection, pytest.raises(OperationalError, match="statement"):
            connection.execute(text("SELECT pg_sleep(5)"))
    finally:
        engine.dispose()


def test_the_background_jobs_opt_out(api_settings_for_timeout: Settings) -> None:
    """A backfill or a retrain runs long queries deliberately."""
    engine = create_db_engine(api_settings_for_timeout, statement_timeout=False)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("SHOW statement_timeout")).scalar_one() == "0"
    finally:
        engine.dispose()
