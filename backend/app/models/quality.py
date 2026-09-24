"""Data-quality issues found during ingestion or by the database audit."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import IssueEntity, IssueSeverity, IssueType
from app.models.mixins import CreatedAtMixin, IdMixin, str_enum


class DataQualityIssue(IdMixin, CreatedAtMixin, Base):
    """One questionable record or fact.

    Issues are never deleted. When the same problem is detected again, the open row
    is updated (`last_seen_at`, `occurrences`) rather than duplicated: `fingerprint`
    is unique among unresolved issues. When the problem disappears, the row is
    kept and `resolved_at` is set.

    `entity_id` points at the internal row when one exists (no FK: the target table
    depends on `entity_type`); `external_id` identifies the provider record, which
    matters when the record was never stored.
    """

    __tablename__ = "data_quality_issues"
    __table_args__ = (
        CheckConstraint("occurrences >= 1", name="occurrences_positive"),
        CheckConstraint("last_seen_at >= first_seen_at", name="seen_order"),
        CheckConstraint(
            "resolved_at IS NULL OR resolved_at >= first_seen_at", name="resolved_after_seen"
        ),
        Index(
            "uq_data_quality_issues_open_fingerprint",
            "fingerprint",
            unique=True,
            postgresql_where=text("resolved_at IS NULL"),
        ),
        Index("ix_data_quality_issues_type_severity", "issue_type", "severity"),
        Index("ix_data_quality_issues_entity", "entity_type", "entity_id"),
    )

    data_source_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("data_sources.id", ondelete="RESTRICT"), index=True
    )
    ingestion_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("data_ingestion_runs.id", ondelete="SET NULL"), index=True
    )
    #: Which check found it: "ingestion:<operation>" (e.g. "ingestion:matches") or
    #: "audit". Only that check resolves it.
    detector: Mapped[str]
    issue_type: Mapped[IssueType] = mapped_column(str_enum(IssueType, "issue_type"))
    severity: Mapped[IssueSeverity] = mapped_column(str_enum(IssueSeverity, "issue_severity"))
    entity_type: Mapped[IssueEntity] = mapped_column(str_enum(IssueEntity, "issue_entity"))
    entity_id: Mapped[int | None] = mapped_column(BigInteger)
    external_id: Mapped[str | None]
    message: Mapped[str]
    #: Evidence, including the offending provider record when it was not stored.
    details: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    fingerprint: Mapped[str]
    first_seen_at: Mapped[datetime]
    last_seen_at: Mapped[datetime]
    occurrences: Mapped[int] = mapped_column(server_default="1")
    resolved_at: Mapped[datetime | None]
