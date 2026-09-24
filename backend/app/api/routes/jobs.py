"""Status of the scheduled background jobs."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status
from sqlalchemy import func, select

from app.api.dependencies import DbSessionDep
from app.models import JobRun
from app.models.enums import JobStatus
from app.schemas.jobs import JobRunPage, JobRunRead, JobSummary, TaskSummary

logger = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"], prefix="/jobs")


def _read(row: JobRun) -> JobRunRead:
    return JobRunRead(
        id=row.id,
        task=row.task,
        status=row.status.value,
        attempt=row.attempt,
        started_at=row.started_at,
        finished_at=row.finished_at,
        duration_ms=row.duration_ms,
        detail=row.detail,
        error=row.error,
        task_id=row.task_id,
    )


@router.get("", response_model=JobRunPage, summary="Recent job runs")
def list_job_runs(
    session: DbSessionDep,
    task: Annotated[
        str | None, Query(description="Only this task, e.g. jobs.fetch_results")
    ] = None,
    job_status: Annotated[
        JobStatus | None, Query(alias="status", description="Only runs with this status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobRunPage:
    """What the scheduler has been doing, newest first."""
    query = select(JobRun).order_by(JobRun.started_at.desc(), JobRun.id.desc())
    if task:
        query = query.where(JobRun.task == task)
    if job_status is not None:
        query = query.where(JobRun.status == job_status)
    total = session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = session.scalars(query.limit(limit).offset(offset))
    return JobRunPage(total=total, limit=limit, offset=offset, items=[_read(row) for row in rows])


@router.get("/summary", response_model=JobSummary, summary="Job health over a window")
def job_summary(
    session: DbSessionDep,
    hours: Annotated[float, Query(gt=0, le=720, description="Window to summarise")] = 24.0,
) -> JobSummary:
    """Per task: how many runs succeeded, were skipped or failed, and when it last ran.

    A task that should be running but has no runs here is the signal that the
    scheduler is not working.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = session.execute(
        select(
            JobRun.task,
            JobRun.status,
            func.count(),
            func.max(JobRun.started_at),
        )
        .where(JobRun.started_at >= since)
        .group_by(JobRun.task, JobRun.status)
    ).all()
    tasks: dict[str, TaskSummary] = {}
    for task, run_status, count, last in rows:
        summary = tasks.get(task) or TaskSummary(task=task)
        counts = dict(summary.counts)
        counts[str(run_status.value)] = int(count)
        last_run = summary.last_run_at
        tasks[task] = TaskSummary(
            task=task,
            counts=counts,
            last_run_at=max(last, last_run) if last_run else last,
        )
    failing = sorted(t for t, s in tasks.items() if s.counts.get("failed"))
    return JobSummary(
        window_hours=hours,
        tasks=sorted(tasks.values(), key=lambda s: s.task),
        failing_tasks=failing,
    )


@router.get(
    "/{run_id}",
    response_model=JobRunRead,
    summary="One job run",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such job run"}},
)
def read_job_run(session: DbSessionDep, run_id: Annotated[int, Path(ge=1)]) -> JobRunRead:
    row = session.get(JobRun, run_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"job run {run_id} was not found"
        )
    return _read(row)
