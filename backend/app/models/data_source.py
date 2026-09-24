"""External data providers and the log of ingestion runs against them."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import IngestionStatus
from app.models.mixins import CreatedAtMixin, IdMixin, TimestampMixin, str_enum


class DataSource(IdMixin, TimestampMixin, Base):
    """A provider of football data, e.g. an API or a CSV archive.

    Every provider-specific identifier in the database references this table, so the
    provider name is stored once rather than repeated on every row.
    """

    __tablename__ = "data_sources"
    __table_args__ = (CheckConstraint("code ~ '^[a-z0-9_]+$'", name="code_format"),)

    code: Mapped[str] = mapped_column(unique=True)  # stable slug, e.g. "football_data_org"
    name: Mapped[str]
    base_url: Mapped[str | None]
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))

    ingestion_runs: Mapped[list["DataIngestionRun"]] = relationship(back_populates="data_source")


class DataIngestionRun(IdMixin, CreatedAtMixin, Base):
    """One execution of an import job, kept for auditing and incremental updates."""

    __tablename__ = "data_ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at", name="finished_after_started"
        ),
        CheckConstraint(
            "(status = 'running') = (finished_at IS NULL)", name="finished_at_matches_status"
        ),
        CheckConstraint(
            "records_fetched >= 0 AND records_inserted >= 0"
            " AND records_updated >= 0 AND records_failed >= 0",
            name="record_counts_non_negative",
        ),
        Index("ix_data_ingestion_runs_source_job_started", "data_source_id", "job", "started_at"),
        Index(
            "ix_data_ingestion_runs_running",
            "data_source_id",
            postgresql_where=text("status = 'running'"),
        ),
    )

    data_source_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("data_sources.id", ondelete="RESTRICT")
    )
    job: Mapped[str]  # e.g. "fixtures", "results", "odds"
    parameters: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[IngestionStatus] = mapped_column(
        str_enum(IngestionStatus, "ingestion_status"),
        server_default=IngestionStatus.RUNNING.value,
    )
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime | None]
    records_fetched: Mapped[int] = mapped_column(server_default="0")
    records_inserted: Mapped[int] = mapped_column(server_default="0")
    records_updated: Mapped[int] = mapped_column(server_default="0")
    records_failed: Mapped[int] = mapped_column(server_default="0")
    error_message: Mapped[str | None]

    data_source: Mapped[DataSource] = relationship(back_populates="ingestion_runs")
