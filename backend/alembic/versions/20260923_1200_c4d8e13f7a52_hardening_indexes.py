"""Indexes for the two hot orderings: newest predictions, recently evaluated.

`GET /predictions` (the dashboard) orders by `predicted_at DESC, id DESC`, and the
reporting query filters `prediction_results.evaluated_at`. Neither had an index, so
both sorted or scanned the whole table.

Revision ID: c4d8e13f7a52
Revises: 9c4f1a7b52de
Create Date: 2026-09-23 12:00:00+00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c4d8e13f7a52"
down_revision: str | Sequence[str] | None = "9c4f1a7b52de"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_predictions_predicted_at ON predictions (predicted_at DESC, id DESC)"
    )
    op.create_index("ix_prediction_results_evaluated_at", "prediction_results", ["evaluated_at"])


def downgrade() -> None:
    op.drop_index("ix_prediction_results_evaluated_at", table_name="prediction_results")
    op.drop_index("ix_predictions_predicted_at", table_name="predictions")
