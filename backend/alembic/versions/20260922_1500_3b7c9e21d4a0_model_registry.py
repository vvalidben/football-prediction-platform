"""Model registry: algorithm, dataset version, competitions, MLflow link, activations.

Revision ID: 3b7c9e21d4a0
Revises: f8d038477c6a
Create Date: 2026-09-22 15:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3b7c9e21d4a0"
down_revision: str | Sequence[str] | None = "f8d038477c6a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # New required columns: added nullable, backfilled, then made NOT NULL.
    op.add_column("model_versions", sa.Column("algorithm", sa.Text(), nullable=True))
    op.add_column("model_versions", sa.Column("dataset_version", sa.Text(), nullable=True))
    op.execute(
        "UPDATE model_versions SET algorithm = coalesce(hyperparameters->>'model_type', name)"
    )
    op.execute("UPDATE model_versions SET dataset_version = 'unrecorded'")
    op.alter_column("model_versions", "algorithm", nullable=False)
    op.alter_column("model_versions", "dataset_version", nullable=False)
    op.add_column(
        "model_versions",
        sa.Column(
            "competitions",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column("model_versions", sa.Column("mlflow_model_name", sa.Text(), nullable=True))
    op.add_column("model_versions", sa.Column("mlflow_model_version", sa.Text(), nullable=True))

    op.create_table(
        "model_activations",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_version_id", sa.BigInteger(), nullable=False),
        sa.Column("previous_version_id", sa.BigInteger(), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("is_rollback", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "deactivated_at IS NULL OR deactivated_at >= activated_at",
            name=op.f("ck_model_activations_deactivation_after_activation"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_model_activations_model_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["previous_version_id"],
            ["model_versions.id"],
            name=op.f("fk_model_activations_previous_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_activations")),
    )
    op.create_index(
        op.f("ix_model_activations_model_version_id"),
        "model_activations",
        ["model_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_model_activations_previous_version_id"),
        "model_activations",
        ["previous_version_id"],
        unique=False,
    )
    op.create_index(
        "ix_model_activations_model_name_activated_at",
        "model_activations",
        ["model_name", "activated_at"],
        unique=False,
    )
    op.create_index(
        "uq_model_activations_one_open_per_name",
        "model_activations",
        ["model_name"],
        unique=True,
        postgresql_where=sa.text("deactivated_at IS NULL"),
    )
    # A version already active before this migration gets its open activation row.
    op.execute(
        "INSERT INTO model_activations (model_name, model_version_id, activated_at, reason) "
        "SELECT name, id, updated_at, 'active before the model registry migration' "
        "FROM model_versions WHERE is_active"
    )


def downgrade() -> None:
    op.drop_index(
        "uq_model_activations_one_open_per_name",
        table_name="model_activations",
        postgresql_where=sa.text("deactivated_at IS NULL"),
    )
    op.drop_index("ix_model_activations_model_name_activated_at", table_name="model_activations")
    op.drop_index(op.f("ix_model_activations_previous_version_id"), table_name="model_activations")
    op.drop_index(op.f("ix_model_activations_model_version_id"), table_name="model_activations")
    op.drop_table("model_activations")
    op.drop_column("model_versions", "mlflow_model_version")
    op.drop_column("model_versions", "mlflow_model_name")
    op.drop_column("model_versions", "competitions")
    op.drop_column("model_versions", "dataset_version")
    op.drop_column("model_versions", "algorithm")
