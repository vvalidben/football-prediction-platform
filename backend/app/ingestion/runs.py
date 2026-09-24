"""Audit log of ingestion operations (the `data_ingestion_runs` table)."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy.orm import Session, sessionmaker

from app.models import DataIngestionRun
from app.models.enums import IngestionStatus

logger = logging.getLogger(__name__)

Outcome = Literal["inserted", "updated", "unchanged"]
Clock = Callable[[], datetime]

#: How many individual record errors are spelled out in `error_message`.
MAX_REPORTED_ERRORS = 10


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunStats:
    """Counters an operation fills in while it runs, plus its final outcome."""

    job: str
    run_id: int = 0
    status: IngestionStatus = IngestionStatus.RUNNING
    error_message: str | None = None
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def record(self, outcome: Outcome) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)

    def fail(self, external_id: str | None, reason: str) -> None:
        self.failed += 1
        self.errors.append(f"[{external_id or '?'}] {reason}")

    def error_summary(self) -> str | None:
        if not self.errors:
            return None
        shown = "; ".join(self.errors[:MAX_REPORTED_ERRORS])
        hidden = len(self.errors) - MAX_REPORTED_ERRORS
        more = f"; … and {hidden} more" if hidden > 0 else ""
        return f"{self.failed} record(s) failed: {shown}{more}"


class IngestionRunLogger:
    """Writes one `data_ingestion_runs` row per operation.

    The row is written in its own session and committed immediately, both when the
    operation starts and when it ends. It therefore survives even if the
    operation's own transaction is rolled back, and a crashed run stays visible
    with status `running`.
    """

    def __init__(self, session_factory: sessionmaker[Session], clock: Clock = utc_now) -> None:
        self._session_factory = session_factory
        self._clock = clock

    @contextmanager
    def track(
        self, data_source_id: int, job: str, parameters: dict[str, Any]
    ) -> Iterator[RunStats]:
        stats = RunStats(job=job, run_id=self._start(data_source_id, job, parameters))
        try:
            yield stats
        except Exception as exc:
            self._finish(stats, IngestionStatus.FAILED, f"{type(exc).__name__}: {exc}")
            raise
        status = IngestionStatus.SUCCEEDED if stats.failed == 0 else IngestionStatus.PARTIAL
        self._finish(stats, status, stats.error_summary())

    def _start(self, data_source_id: int, job: str, parameters: dict[str, Any]) -> int:
        with self._session_factory() as session, session.begin():
            run = DataIngestionRun(
                data_source_id=data_source_id,
                job=job,
                parameters=parameters,
                status=IngestionStatus.RUNNING,
                started_at=self._clock(),
            )
            session.add(run)
            session.flush()
            logger.info("Ingestion run %d started: %s %s", run.id, job, parameters)
            return run.id

    def _finish(self, stats: RunStats, status: IngestionStatus, error: str | None) -> None:
        stats.status, stats.error_message = status, error
        with self._session_factory() as session, session.begin():
            run = session.get_one(DataIngestionRun, stats.run_id)
            run.status = status
            run.finished_at = max(self._clock(), run.started_at)
            run.records_fetched = stats.fetched
            run.records_inserted = stats.inserted
            run.records_updated = stats.updated
            run.records_failed = stats.failed
            run.error_message = error
        log = logger.info if status is IngestionStatus.SUCCEEDED else logger.warning
        log(
            "Ingestion run %d %s: fetched=%d inserted=%d updated=%d unchanged=%d failed=%d%s",
            stats.run_id,
            status.value,
            stats.fetched,
            stats.inserted,
            stats.updated,
            stats.unchanged,
            stats.failed,
            f" ({error})" if error else "",
        )
