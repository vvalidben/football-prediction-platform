"""Recording what each scheduled job did, so job status is queryable, not only logged.

Every task body runs inside `record(...)`, which writes a `job_runs` row when the
job starts and completes it afterwards:

- finished normally: `succeeded`, with whatever counts the job reported;
- raised `JobSkipped`: `skipped`, with the reason (nothing to do, not configured,
  not due yet). A skipped job is not a failure and is not retried;
- raised anything else: `failed`, with the error, and the exception is re-raised so
  Celery can retry it.

Rows are written in their own transaction, so a job that fails mid-way still leaves
a record.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import JobRun
from app.models.enums import JobStatus

logger = logging.getLogger(__name__)
MAX_ERROR = 2000


class JobSkipped(Exception):  # noqa: N818 - a skip is an outcome, not an error
    """The job had nothing to do; not an error, and not retried."""


@dataclass
class JobReport:
    """What a job did; becomes the `detail` of its `job_runs` row."""

    values: dict[str, Any] = field(default_factory=dict)

    def add(self, **counts: Any) -> None:
        self.values.update(counts)

    def increment(self, **counts: int) -> None:
        for key, value in counts.items():
            self.values[key] = self.values.get(key, 0) + value


def running_since(
    session_factory: sessionmaker[Session],
    task: str,
    *,
    stale_after: timedelta,
    now: datetime | None = None,
) -> datetime | None:
    """When a live run of this task started, or None if none is running.

    A schedule shorter than a job's runtime would otherwise start the same job
    again while the first is still working — two fixture fetches spend the
    provider quota twice for the same data.

    A run older than `stale_after` is not treated as live: Celery kills a task at
    its hard time limit, so beyond that the row belongs to a worker that died
    rather than to work in progress.
    """
    now = now or datetime.now(UTC)
    with session_factory() as session:
        started = session.scalars(
            select(JobRun.started_at)
            .where(
                JobRun.task == task,
                JobRun.status == JobStatus.RUNNING,
                JobRun.started_at >= now - stale_after,
            )
            .order_by(JobRun.started_at.desc())
            .limit(1)
        ).first()
    return started


def record_skip(
    session_factory: sessionmaker[Session],
    task: str,
    reason: str,
    *,
    attempt: int = 1,
    task_id: str | None = None,
) -> None:
    """Write a skipped run without running anything: the skip is still history."""
    now = datetime.now(UTC)
    with session_factory() as session, session.begin():
        session.add(
            JobRun(
                task=task,
                status=JobStatus.SKIPPED,
                attempt=attempt,
                started_at=now,
                finished_at=now,
                duration_ms=0,
                detail={"reason": reason},
                task_id=task_id,
            )
        )
    logger.info("Job %s skipped: %s", task, reason)


@contextmanager
def record(
    task: str,
    session_factory: sessionmaker[Session],
    *,
    attempt: int = 1,
    task_id: str | None = None,
) -> Iterator[JobReport]:
    report = JobReport()
    started = datetime.now(UTC)
    clock = time.perf_counter()
    with session_factory() as session, session.begin():
        row = JobRun(
            task=task,
            status=JobStatus.RUNNING,
            attempt=attempt,
            started_at=started,
            detail={},
            task_id=task_id,
        )
        session.add(row)
        session.flush()
        run_id = row.id
    logger.info("Job %s started (attempt %s, run %s)", task, attempt, run_id)
    try:
        yield report
    except JobSkipped as skipped:
        _finish(session_factory, run_id, JobStatus.SKIPPED, report, clock, reason=str(skipped))
        logger.info("Job %s skipped: %s", task, skipped)
    except Exception as exc:
        _finish(session_factory, run_id, JobStatus.FAILED, report, clock, error=repr(exc))
        logger.exception("Job %s failed on attempt %s", task, attempt)
        raise
    else:
        _finish(session_factory, run_id, JobStatus.SUCCEEDED, report, clock)
        logger.info("Job %s succeeded: %s", task, report.values)


def _finish(
    session_factory: sessionmaker[Session],
    run_id: int,
    status: JobStatus,
    report: JobReport,
    clock: float,
    *,
    error: str | None = None,
    reason: str | None = None,
) -> None:
    detail = dict(report.values)
    if reason is not None:
        detail["reason"] = reason
    with session_factory() as session, session.begin():
        row = session.get(JobRun, run_id)
        if row is None:  # pragma: no cover - the row was just written
            return
        row.status = status
        row.finished_at = datetime.now(UTC)
        row.duration_ms = int((time.perf_counter() - clock) * 1000)
        row.detail = detail
        row.error = error[:MAX_ERROR] if error else None
