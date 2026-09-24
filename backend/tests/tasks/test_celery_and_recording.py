"""The Celery application, the schedule, backoff, and job recording."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import JobRun
from app.models.enums import JobStatus
from app.providers.errors import ProviderUnavailableError, RateLimitError
from app.tasks.celery_app import SCHEDULES, beat_schedule, broker_url, create_celery, result_backend
from app.tasks.jobs import retry_delay
from app.tasks.recording import JobSkipped, record


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "postgres_user": "u",
        "postgres_password": SecretStr("p"),
        "postgres_db": "d",
        "redis_host": "cache",
        "redis_port": 6380,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class TestConfiguration:
    def test_broker_and_backend_default_to_redis(self) -> None:
        settings = make_settings(redis_db=3, redis_password=SecretStr("s3cret/pass"))

        assert broker_url(settings) == "redis://:s3cret%2Fpass@cache:6380/4"
        assert result_backend(settings) == "redis://:s3cret%2Fpass@cache:6380/5"

    def test_explicit_broker_wins(self) -> None:
        settings = make_settings(celery_broker_url="redis://elsewhere:6379/9")

        assert broker_url(settings) == "redis://elsewhere:6379/9"

    def test_every_job_is_scheduled_from_settings(self) -> None:
        settings = make_settings(schedule_results_minutes=15, schedule_retrain_minutes=60 * 24 * 7)

        entries = beat_schedule(settings)

        assert set(entries) == set(SCHEDULES)
        assert entries["jobs.fetch_results"]["schedule"].run_every == timedelta(minutes=15)
        assert entries["jobs.retrain_models"]["schedule"].run_every == timedelta(days=7)
        # A queued job expires rather than piling up if the worker is behind.
        assert entries["jobs.fetch_results"]["options"]["expires"] == 15 * 60

    def test_the_scheduler_can_be_switched_off(self) -> None:
        assert beat_schedule(make_settings(jobs_enabled=False)) == {}
        assert create_celery(make_settings(jobs_enabled=False)).conf.beat_schedule == {}

    def test_worker_settings_are_safe_for_retries(self) -> None:
        celery = create_celery(make_settings(task_time_limit_seconds=600))

        assert celery.conf.task_acks_late is True  # a dead worker's job is redelivered
        assert celery.conf.worker_prefetch_multiplier == 1
        assert celery.conf.task_time_limit == 600
        assert celery.conf.timezone == "UTC"

    def test_every_task_in_the_schedule_exists(self) -> None:
        celery = create_celery(make_settings())
        import app.tasks.jobs  # noqa: F401 - registers the tasks

        for task in SCHEDULES:
            assert task in celery.tasks, task


class TestBackoff:
    def test_delay_grows_exponentially_and_is_capped(self) -> None:
        settings = make_settings(task_retry_backoff_seconds=10, task_retry_backoff_max_seconds=100)

        delays = [retry_delay(settings, attempt) for attempt in range(1, 8)]

        assert 7 <= delays[0] <= 10  # 10s with jitter
        assert 14 <= delays[1] <= 20
        assert 28 <= delays[2] <= 40
        assert all(delay <= 100 for delay in delays)
        assert delays[-1] >= 70  # still near the cap, never zero

    def test_jitter_spreads_retries(self) -> None:
        settings = make_settings()

        values = {retry_delay(settings, 3) for _ in range(20)}

        assert len(values) > 1  # not all workers retry at the same instant

    def test_a_rate_limit_waits_at_least_as_long_as_asked(self) -> None:
        settings = make_settings(task_retry_backoff_seconds=1, task_retry_backoff_max_seconds=5)
        limited = RateLimitError("slow down", provider="api_football", retry_after=42)

        wait = max(float(limited.retry_after or 0), retry_delay(settings, 1))

        assert wait == 42  # the provider's instruction wins over our backoff
        assert retry_delay(settings, 1) < 42


@pytest.mark.integration
class TestRecording:
    def test_a_successful_job_is_recorded_with_its_counts(
        self, session_factory: sessionmaker[Session], session: Session
    ) -> None:
        with record("jobs.fetch_results", session_factory, attempt=1, task_id="abc") as report:
            report.add(competitions=2)
            report.increment(matches_inserted=3)
            report.increment(matches_inserted=4)

        row = _only(session_factory)
        assert row.task == "jobs.fetch_results"
        assert row.status is JobStatus.SUCCEEDED
        assert row.attempt == 1 and row.task_id == "abc"
        assert row.detail == {"competitions": 2, "matches_inserted": 7}
        assert row.error is None
        assert row.finished_at is not None and row.duration_ms is not None
        assert row.finished_at >= row.started_at

    def test_a_skipped_job_records_the_reason_and_does_not_raise(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with record("jobs.fetch_fixtures", session_factory) as report:
            report.add(looked_at=0)
            raise JobSkipped("no competitions configured")

        row = _only(session_factory)
        assert row.status is JobStatus.SKIPPED
        assert row.detail["reason"] == "no competitions configured"
        assert row.error is None

    def test_a_failing_job_is_recorded_and_the_error_propagates(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with (
            pytest.raises(ProviderUnavailableError),
            record("jobs.fetch_results", session_factory, attempt=3) as report,
        ):
            report.add(competitions=1)
            raise ProviderUnavailableError("api down")

        row = _only(session_factory)
        assert row.status is JobStatus.FAILED
        assert row.attempt == 3
        assert "api down" in (row.error or "")
        assert row.detail == {"competitions": 1}
        assert row.finished_at is not None

    def test_runs_are_separate_rows_per_attempt(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        for attempt in (1, 2):
            with record("jobs.fetch_results", session_factory, attempt=attempt):
                pass

        with session_factory() as session:
            rows = list(session.scalars(select(JobRun).order_by(JobRun.attempt)))
        assert [row.attempt for row in rows] == [1, 2]
        assert all(row.status is JobStatus.SUCCEEDED for row in rows)

    def test_a_started_job_is_visible_before_it_finishes(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        with record("jobs.recalculate_features", session_factory):
            row = _only(session_factory)
            assert row.status is JobStatus.RUNNING
            assert row.finished_at is None
            assert row.started_at <= datetime.now(UTC)


def _only(session_factory: sessionmaker[Session]) -> JobRun:
    with session_factory() as session:
        rows = list(session.scalars(select(JobRun).order_by(JobRun.id.desc())))
    assert rows, "no job run was recorded"
    return rows[0]


class TestOverlappingRuns:
    """A schedule shorter than a job's runtime must not start it twice.

    Two fixture fetches for the same competition spend the provider quota twice for
    the same data, and the free tier is 100 requests a day.
    """

    def test_a_live_run_is_reported(self, session_factory: sessionmaker[Session]) -> None:
        from datetime import UTC, datetime, timedelta

        from app.tasks.recording import running_since

        now = datetime.now(UTC)
        with session_factory() as session, session.begin():
            session.add(
                JobRun(
                    task="jobs.fetch_results",
                    status=JobStatus.RUNNING,
                    attempt=1,
                    started_at=now - timedelta(minutes=5),
                    detail={},
                )
            )

        found = running_since(
            session_factory, "jobs.fetch_results", stale_after=timedelta(minutes=30), now=now
        )

        assert found is not None
        assert (
            running_since(
                session_factory, "jobs.fetch_fixtures", stale_after=timedelta(minutes=30), now=now
            )
            is None
        )

    def test_a_run_left_behind_by_a_dead_worker_does_not_block_forever(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """Celery kills a task at its hard time limit; past that the row is a corpse."""
        from datetime import UTC, datetime, timedelta

        from app.tasks.recording import running_since

        now = datetime.now(UTC)
        with session_factory() as session, session.begin():
            session.add(
                JobRun(
                    task="jobs.retrain_models",
                    status=JobStatus.RUNNING,
                    attempt=1,
                    started_at=now - timedelta(hours=6),
                    detail={},
                )
            )

        assert (
            running_since(
                session_factory,
                "jobs.retrain_models",
                stale_after=timedelta(minutes=30),
                now=now,
            )
            is None
        )

    def test_a_finished_run_does_not_block_the_next_one(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        from datetime import UTC, datetime, timedelta

        from app.tasks.recording import running_since

        now = datetime.now(UTC)
        with session_factory() as session, session.begin():
            session.add(
                JobRun(
                    task="jobs.fetch_results",
                    status=JobStatus.SUCCEEDED,
                    attempt=1,
                    started_at=now - timedelta(minutes=1),
                    finished_at=now,
                    detail={},
                )
            )

        assert (
            running_since(
                session_factory, "jobs.fetch_results", stale_after=timedelta(minutes=30), now=now
            )
            is None
        )

    def test_the_skip_is_recorded_as_history(self, session_factory: sessionmaker[Session]) -> None:
        from sqlalchemy import select

        from app.tasks.recording import record_skip

        record_skip(session_factory, "jobs.fetch_results", "already running", task_id="abc")

        with session_factory() as session:
            row = session.scalars(select(JobRun).where(JobRun.task == "jobs.fetch_results")).one()
        assert row.status is JobStatus.SKIPPED
        assert row.detail["reason"] == "already running"
        assert row.finished_at is not None and row.duration_ms == 0
