"""The registry migration: new columns, backfill, and the activation history table."""

import pytest
from alembic import command
from sqlalchemy import Engine, inspect, text

from tests.database import alembic_config

pytestmark = pytest.mark.integration

BEFORE = "f8d038477c6a"  # the revision before the model registry


def test_existing_versions_are_backfilled_and_keep_their_active_model(
    db_engine: Engine,
) -> None:
    """A database written before this sprint upgrades without losing anything."""
    with db_engine.connect() as connection, connection.begin() as transaction:
        config = alembic_config(connection)
        command.downgrade(config, BEFORE)
        columns = {c["name"] for c in inspect(connection).get_columns("model_versions")}
        assert "algorithm" not in columns and "dataset_version" not in columns
        connection.execute(
            text(
                "INSERT INTO model_versions "
                "(name, version, hyperparameters, training_metrics, is_active, updated_at) VALUES "
                "('elo', 'v1', '{\"model_type\": \"EloModel\"}'::jsonb, '{}'::jsonb, true, now()), "
                "('poisson', 'v1', '{}'::jsonb, '{}'::jsonb, false, now())"
            )
        )

        command.upgrade(config, "head")

        rows = connection.execute(
            text(
                "SELECT name, algorithm, dataset_version, competitions, is_active "
                "FROM model_versions ORDER BY name"
            )
        ).all()
        assert [(r.name, r.algorithm) for r in rows] == [
            ("elo", "EloModel"),
            ("poisson", "poisson"),
        ]
        assert all(r.dataset_version == "unrecorded" for r in rows)
        assert all(r.competitions == [] for r in rows)
        activations = connection.execute(
            text(
                "SELECT model_name, deactivated_at, reason FROM model_activations "
                "JOIN model_versions ON model_versions.id = model_activations.model_version_id"
            )
        ).all()
        # Only the active version gets an open activation row.
        assert len(activations) == 1
        assert activations[0].model_name == "elo"
        assert activations[0].deactivated_at is None
        assert "before the model registry migration" in activations[0].reason
        transaction.rollback()


def test_downgrade_removes_the_registry_additions(db_engine: Engine) -> None:
    with db_engine.connect() as connection, connection.begin() as transaction:
        config = alembic_config(connection)

        command.downgrade(config, BEFORE)

        assert "model_activations" not in set(inspect(connection).get_table_names())
        columns = {c["name"] for c in inspect(connection).get_columns("model_versions")}
        assert {
            "algorithm",
            "dataset_version",
            "competitions",
            "mlflow_model_name",
        } & columns == set()
        command.upgrade(config, "head")
        transaction.rollback()
