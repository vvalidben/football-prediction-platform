"""Job status as the API returns it."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.catalog import Page


class JobRunRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    task: str
    status: str
    attempt: int
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    #: What the job did: counts, or the reason it was skipped.
    detail: dict[str, Any]
    error: str | None
    task_id: str | None


class JobRunPage(Page):
    items: list[JobRunRead]


class TaskSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    #: Runs per status in the window, e.g. {"succeeded": 12, "failed": 1}.
    counts: dict[str, int] = Field(default_factory=dict)
    last_run_at: datetime | None = None


class JobSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_hours: float
    tasks: list[TaskSummary]
    failing_tasks: list[str]
