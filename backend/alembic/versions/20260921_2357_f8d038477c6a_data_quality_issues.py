"""Data-quality issues: questionable records found by ingestion or the audit.

Revision ID: f8d038477c6a
Revises: ea24dba4071d
Create Date: 2026-09-21 23:57:37.912445+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f8d038477c6a"
down_revision: str | Sequence[str] | None = "ea24dba4071d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "data_quality_issues",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=True),
        sa.Column("ingestion_run_id", sa.BigInteger(), nullable=True),
        sa.Column("detector", sa.Text(), nullable=False),
        sa.Column(
            "issue_type",
            sa.Enum(
                "malformed_record",
                "missing_team",
                "duplicate_match",
                "duplicate_provider_id",
                "possible_duplicate_team",
                "impossible_score",
                "invalid_date",
                "missing_statistics",
                "inconsistent_statistics",
                "inconsistent_competition",
                "incomplete_record",
                name="issue_type",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "severity",
            sa.Enum(
                "error",
                "warning",
                "info",
                name="issue_severity",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "entity_type",
            sa.Enum(
                "competition",
                "season",
                "team",
                "match",
                "team_match_statistics",
                name="issue_entity",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("entity_id", sa.BigInteger(), nullable=True),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("fingerprint", sa.Text(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("occurrences", sa.Integer(), server_default="1", nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "last_seen_at >= first_seen_at", name=op.f("ck_data_quality_issues_seen_order")
        ),
        sa.CheckConstraint(
            "occurrences >= 1", name=op.f("ck_data_quality_issues_occurrences_positive")
        ),
        sa.CheckConstraint(
            "resolved_at IS NULL OR resolved_at >= first_seen_at",
            name=op.f("ck_data_quality_issues_resolved_after_seen"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_data_quality_issues_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"],
            ["data_ingestion_runs.id"],
            name=op.f("fk_data_quality_issues_ingestion_run_id_data_ingestion_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_quality_issues")),
    )
    op.create_index(
        op.f("ix_data_quality_issues_data_source_id"),
        "data_quality_issues",
        ["data_source_id"],
        unique=False,
    )
    op.create_index(
        "ix_data_quality_issues_entity",
        "data_quality_issues",
        ["entity_type", "entity_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_data_quality_issues_ingestion_run_id"),
        "data_quality_issues",
        ["ingestion_run_id"],
        unique=False,
    )
    op.create_index(
        "ix_data_quality_issues_type_severity",
        "data_quality_issues",
        ["issue_type", "severity"],
        unique=False,
    )
    op.create_index(
        "uq_data_quality_issues_open_fingerprint",
        "data_quality_issues",
        ["fingerprint"],
        unique=True,
        postgresql_where=sa.text("resolved_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_data_quality_issues_open_fingerprint",
        table_name="data_quality_issues",
        postgresql_where=sa.text("resolved_at IS NULL"),
    )
    op.drop_index("ix_data_quality_issues_type_severity", table_name="data_quality_issues")
    op.drop_index(op.f("ix_data_quality_issues_ingestion_run_id"), table_name="data_quality_issues")
    op.drop_index("ix_data_quality_issues_entity", table_name="data_quality_issues")
    op.drop_index(op.f("ix_data_quality_issues_data_source_id"), table_name="data_quality_issues")
    op.drop_table("data_quality_issues")
