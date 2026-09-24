"""The jobs command line: run a job now, see what ran, print the schedule.

This is what an operator reaches for when something looks wrong at three in the
morning, and until Sprint 17 none of it was covered by a test.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import JobRun
from app.models.enums import JobStatus
from app.tasks import __main__ as cli
from app.tasks import runtime
from tests.tasks.test_celery_and_recording import make_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def settings() -> Settings:
    return make_settings(job_competitions=["E0"], schedule_results_minutes=45.0)


@pytest.fixture(autouse=True)
def runtime_state(
    session_factory: sessionmaker[Session], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    runtime.configure(new_settings=settings, new_session_factory=session_factory)
    # `main` reads settings through the cached getter, not through runtime.
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    yield
    runtime.reset()


class TestSchedule:
    def test_prints_every_task_with_its_interval(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["schedule"]) == 0

        printed = capsys.readouterr().out
        for task in cli.TASKS:
            assert task in printed
        assert "45.0 min" in printed  # the configured results interval
        assert "SCHEDULE_RESULTS_MINUTES" in printed  # the variable that sets it

    def test_says_whether_jobs_and_retraining_are_on(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "get_settings", lambda: make_settings(jobs_enabled=False))

        cli.main(["schedule"])

        printed = capsys.readouterr().out
        assert "jobs disabled" in printed
        assert "retraining: every" in printed


class TestRun:
    def test_runs_a_job_in_this_process_and_prints_its_outcome(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`evaluate_predictions` on an empty database does nothing, successfully:
        the point is that the CLI runs the task and shows what came back."""
        assert cli.main(["run", "jobs.evaluate_predictions"]) == 0

        printed = capsys.readouterr().out
        assert '"evaluated": 0' in printed

    def test_queueing_hands_the_job_to_the_worker_instead(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[str] = []

        class FakeResult:
            id = "task-42"

        class FakeCelery:
            def send_task(self, name: str) -> FakeResult:
                sent.append(name)
                return FakeResult()

        monkeypatch.setattr(cli, "create_celery", lambda _: FakeCelery())

        assert cli.main(["run", "jobs.fetch_results", "--queue"]) == 0

        assert sent == ["jobs.fetch_results"]
        assert "queued jobs.fetch_results as task-42" in capsys.readouterr().out

    def test_an_unknown_task_is_refused_with_the_options(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main(["run", "jobs.make_tea"])

        assert exit_info.value.code == 2
        assert "jobs.fetch_results" in capsys.readouterr().err  # the choices are listed

    def test_every_scheduled_task_can_be_run_from_the_command_line(self) -> None:
        """A task that the scheduler runs but the CLI cannot is a gap an operator
        only discovers when they need it."""
        from app.tasks.celery_app import SCHEDULES

        assert set(SCHEDULES) <= set(cli.TASKS)


class TestStatus:
    @pytest.fixture
    def recorded_runs(self, session_factory: sessionmaker[Session]) -> None:
        now = datetime.now(UTC)
        with session_factory() as session, session.begin():
            session.add_all(
                [
                    JobRun(
                        task="jobs.fetch_results",
                        status=JobStatus.SUCCEEDED,
                        started_at=now - timedelta(hours=1),
                        finished_at=now - timedelta(hours=1),
                        duration_ms=1200,
                        detail={"matches": 3},
                    ),
                    JobRun(
                        task="jobs.fetch_results",
                        status=JobStatus.FAILED,
                        started_at=now - timedelta(hours=2),
                        finished_at=now - timedelta(hours=2),
                        duration_ms=900,
                        error="provider unavailable",
                    ),
                    JobRun(
                        task="jobs.refresh_predictions",
                        status=JobStatus.SUCCEEDED,
                        started_at=now - timedelta(days=3),  # outside the default window
                        finished_at=now - timedelta(days=3),
                        duration_ms=100,
                        detail={},
                    ),
                ]
            )

    def test_summarises_recent_runs_and_lists_them(
        self, recorded_runs: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["status"]) == 0

        printed = capsys.readouterr().out
        assert "last 24 h:" in printed
        assert "jobs.fetch_results" in printed
        assert "succeeded" in printed and "failed" in printed
        assert "provider unavailable" in printed  # the failure is not hidden
        assert "jobs.refresh_predictions" not in printed  # older than the window

    def test_a_longer_window_reaches_further_back(
        self, recorded_runs: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["status", "--hours", "96"])

        assert "jobs.refresh_predictions" in capsys.readouterr().out

    def test_one_task_can_be_singled_out(
        self, recorded_runs: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["status", "--task", "jobs.fetch_results", "--hours", "96"])

        printed = capsys.readouterr().out
        assert "jobs.fetch_results" in printed
        # The counts above the table still cover everything in the window, so the
        # filter is checked on the listing rows.
        rows = [line for line in printed.splitlines() if line.startswith("20")]
        assert rows and all("jobs.fetch_results" in row for row in rows)

    def test_the_listing_is_capped(
        self, recorded_runs: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["status", "--hours", "96", "--limit", "1"])

        rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("20")]
        assert len(rows) == 1

    def test_an_empty_history_prints_a_table_with_no_rows(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["status"]) == 0

        printed = capsys.readouterr().out
        assert "last 24 h:" in printed
        assert [line for line in printed.splitlines() if line.startswith("20")] == []


def test_the_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2


def test_the_help_text_lists_what_the_tool_does(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    printed: Any = capsys.readouterr().out
    assert "run" in printed and "status" in printed and "schedule" in printed
