"""Status of scheduled background jobs (see docs/scheduled-jobs.md)."""

from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, Index, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.enums import JobStatus
from app.models.mixins import CreatedAtMixin, IdMixin, str_enum


class JobRun(IdMixin, CreatedAtMixin, Base):
    """One execution of one scheduled task, kept for monitoring and failure reports."""

    __tablename__ = "job_runs"
    __table_args__ = (
        CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at", name="finished_after_started"
        ),
        CheckConstraint("attempt >= 1", name="attempt_positive"),
        Index("ix_job_runs_task_started_at", "task", "started_at"),
        Index("ix_job_runs_status_started_at", "status", "started_at"),
    )

    task: Mapped[str]
    status: Mapped[JobStatus] = mapped_column(str_enum(JobStatus, "job_status"))
    #: 1 for the first try; higher after a retry.
    attempt: Mapped[int] = mapped_column(server_default=text("1"))
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime | None]
    duration_ms: Mapped[int | None]
    #: What the job did: counts, ids, the reason it was skipped.
    detail: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    error: Mapped[str | None]
    #: Celery task id, when the job ran through the worker.
    task_id: Mapped[str | None]
