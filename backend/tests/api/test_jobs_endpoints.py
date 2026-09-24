"""GET /jobs, /jobs/summary and /jobs/{run_id}."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import JobRun
from app.models.enums import JobStatus

pytestmark = pytest.mark.integration


@pytest.fixture
def recorded_runs(session_factory: sessionmaker[Session]) -> dict[str, int]:
    now = datetime.now(UTC)
    rows = [
        JobRun(
            task="jobs.fetch_results",
            status=JobStatus.SUCCEEDED,
            attempt=1,
            started_at=now - timedelta(minutes=10),
            finished_at=now - timedelta(minutes=9),
            duration_ms=1200,
            detail={"matches_updated": 4},
        ),
        JobRun(
            task="jobs.fetch_results",
            status=JobStatus.FAILED,
            attempt=3,
            started_at=now - timedelta(hours=2),
            finished_at=now - timedelta(hours=2),
            duration_ms=900,
            detail={"competitions": 1},
            error="ProviderUnavailableError('api down')",
        ),
        JobRun(
            task="jobs.refresh_predictions",
            status=JobStatus.SKIPPED,
            attempt=1,
            started_at=now - timedelta(days=3),  # outside the default window
            finished_at=now - timedelta(days=3),
            detail={"reason": "no upcoming matches within the horizon"},
        ),
    ]
    with session_factory() as session, session.begin():
        session.add_all(rows)
        session.flush()
        return {row.task + str(row.status.value): row.id for row in rows}


class TestJobRuns:
    def test_lists_runs_newest_first(
        self, client: TestClient, recorded_runs: dict[str, int]
    ) -> None:
        response = client.get("/jobs")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        assert [item["task"] for item in body["items"]] == [
            "jobs.fetch_results",
            "jobs.fetch_results",
            "jobs.refresh_predictions",
        ]
        first = body["items"][0]
        assert first["status"] == "succeeded"
        assert first["detail"] == {"matches_updated": 4}
        assert first["duration_ms"] == 1200
        assert first["error"] is None

    def test_filters_by_task_and_status(
        self, client: TestClient, recorded_runs: dict[str, int]
    ) -> None:
        by_task = client.get("/jobs", params={"task": "jobs.refresh_predictions"}).json()
        failures = client.get("/jobs", params={"status": "failed"}).json()

        assert [item["task"] for item in by_task["items"]] == ["jobs.refresh_predictions"]
        assert by_task["items"][0]["detail"]["reason"].startswith("no upcoming")
        assert failures["total"] == 1
        assert failures["items"][0]["attempt"] == 3
        assert "api down" in failures["items"][0]["error"]

    def test_rejects_an_unknown_status(self, client: TestClient) -> None:
        assert client.get("/jobs", params={"status": "exploded"}).status_code == 422

    def test_reads_one_run(self, client: TestClient, recorded_runs: dict[str, int]) -> None:
        run_id = recorded_runs["jobs.fetch_resultsfailed"]

        response = client.get(f"/jobs/{run_id}")

        assert response.status_code == 200
        assert response.json()["id"] == run_id
        assert response.json()["status"] == "failed"

    def test_unknown_run_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get("/jobs/424242")

        assert response.status_code == 404
        assert "job run 424242 was not found" in response.json()["detail"]


class TestSummary:
    def test_summarises_the_window_and_names_failing_tasks(
        self, client: TestClient, recorded_runs: dict[str, int]
    ) -> None:
        body = client.get("/jobs/summary", params={"hours": 24}).json()

        assert body["window_hours"] == 24
        tasks = {entry["task"]: entry for entry in body["tasks"]}
        assert tasks["jobs.fetch_results"]["counts"] == {"succeeded": 1, "failed": 1}
        assert "jobs.refresh_predictions" not in tasks  # older than the window
        assert body["failing_tasks"] == ["jobs.fetch_results"]
        assert tasks["jobs.fetch_results"]["last_run_at"] is not None

    def test_a_longer_window_includes_older_runs(
        self, client: TestClient, recorded_runs: dict[str, int]
    ) -> None:
        body = client.get("/jobs/summary", params={"hours": 24 * 7}).json()

        tasks = {entry["task"]: entry for entry in body["tasks"]}
        assert tasks["jobs.refresh_predictions"]["counts"] == {"skipped": 1}

    def test_a_quiet_window_has_no_failures(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        body = client.get("/jobs/summary", params={"hours": 1}).json()

        assert body["tasks"] == [] and body["failing_tasks"] == []


def test_job_endpoints_are_documented(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert {"/jobs", "/jobs/summary", "/jobs/{run_id}"} <= set(spec["paths"])
    assert spec["paths"]["/jobs"]["get"]["tags"] == ["jobs"]
    summary = spec["paths"]["/jobs/summary"]["get"]
    assert summary["summary"] == "Job health over a window"
    assert "scheduler is not working" in summary["description"]
