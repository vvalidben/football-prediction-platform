"""Each scheduled job, run against a real database with a mock provider."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import (
    Competition,
    FeatureSnapshot,
    JobRun,
    Match,
    ModelVersion,
    Prediction,
    PredictionResult,
    Team,
)
from app.models.enums import CompetitionType, JobStatus, MatchStatus
from app.providers.base import ProviderOperation
from app.providers.errors import ProviderUnavailableError, RateLimitError
from app.providers.mock import MockDataset, MockFootballDataProvider
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ScorePair,
)
from app.tasks import jobs, runtime
from tests.tasks.test_celery_and_recording import make_settings

pytestmark = pytest.mark.integration

TEAMS = ("Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford")
#: Cycled so every season has home wins, draws and away wins (models need all three).
SCORES = ((2, 0), (1, 1), (0, 1), (3, 1), (2, 2), (0, 2), (1, 0), (0, 0), (1, 2))
SEASON_YEARS = (2023, 2024, 2025)


@pytest.fixture
def settings() -> Settings:
    return make_settings(job_competitions=["E0"], prediction_horizon_days=30)


@pytest.fixture(autouse=True)
def runtime_state(session_factory: sessionmaker[Session], settings: Settings) -> Iterator[None]:
    """Point the tasks at the test database and a mock provider."""
    runtime.configure(
        new_settings=settings,
        new_session_factory=session_factory,
        new_provider_factory=lambda _: MockFootballDataProvider(provider_dataset()),
    )
    yield
    runtime.reset()


def provider_dataset() -> MockDataset:
    """One competition, three seasons of a four-team double round robin.

    Three seasons is what a season split needs, so retraining can really run.
    """
    dataset = MockDataset(
        competitions=[
            ProviderCompetition(
                external_id="E0",
                name="Premier League",
                type=CompetitionType.LEAGUE,
                country="England",
            )
        ]
    )
    fixture = 0
    for index, year in enumerate(SEASON_YEARS):
        season_id = f"E0-{year}"
        dataset.seasons.append(
            ProviderSeason(
                external_id=season_id,
                competition_external_id="E0",
                name=f"{year}/{year + 1}",
                start_date=date(year, 8, 1),
                end_date=date(year + 1, 5, 31),
                is_current=index == len(SEASON_YEARS) - 1,
                statistics_available=True,
            )
        )
        dataset.teams[season_id] = [
            ProviderTeam(external_id=name, name=name, country="England") for name in TEAMS
        ]
        kickoff = datetime(year, 9, 1, 15, tzinfo=UTC)
        for home in TEAMS:
            for away in TEAMS:
                if home == away:
                    continue
                dataset.matches.append(
                    ProviderMatch(
                        external_id=f"{year}-{home}-{away}",
                        competition_external_id="E0",
                        season_external_id=season_id,
                        home_team_external_id=home,
                        away_team_external_id=away,
                        kickoff_at=kickoff,
                        status=MatchStatus.FINISHED,
                        score=ScorePair(
                            home=SCORES[fixture % len(SCORES)][0],
                            away=SCORES[fixture % len(SCORES)][1],
                        ),
                    )
                )
                fixture += 1
                kickoff += timedelta(days=3)
    return dataset


MATCHES_PER_SEASON = len(TEAMS) * (len(TEAMS) - 1)
TOTAL_MATCHES = MATCHES_PER_SEASON * len(SEASON_YEARS)


def run(task: Any) -> dict[str, Any]:
    """Run a task in this process, as the worker would."""
    outcome: dict[str, Any] = task.apply().get()
    return outcome


def _collect(sink: list[Any], key: str) -> Any:
    """A stand-in for `task.retry` that records one argument and stops the task."""

    def retry(**kwargs: Any) -> Exception:
        sink.append(kwargs[key])
        return RuntimeError("retry")

    return retry


def last_run(session_factory: sessionmaker[Session], task: str) -> JobRun:
    with session_factory() as session:
        row = session.scalars(
            select(JobRun).where(JobRun.task == task).order_by(JobRun.id.desc()).limit(1)
        ).one_or_none()
    assert row is not None, f"{task} recorded no run"
    return row


class TestFetchJobs:
    def test_fixtures_are_imported_and_recorded(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        result = run(jobs.fetch_fixtures)

        with session_factory() as session:
            assert session.scalars(select(Competition)).all()
            assert len(session.scalars(select(Team)).all()) == len(TEAMS)
            assert session.scalars(select(Match)).all()
        row = last_run(session_factory, "jobs.fetch_fixtures")
        assert row.status is JobStatus.SUCCEEDED
        assert row.detail["competitions"] == 1
        # The job keeps the current season up to date, not the whole archive.
        assert row.detail["matches_inserted"] == MATCHES_PER_SEASON
        assert result["matches_inserted"] == row.detail["matches_inserted"]

    def test_results_update_existing_matches(self, session_factory: sessionmaker[Session]) -> None:
        run(jobs.fetch_fixtures)

        result = run(jobs.fetch_results)

        assert result["matches_inserted"] == 0  # nothing new
        row = last_run(session_factory, "jobs.fetch_results")
        assert row.status is JobStatus.SUCCEEDED
        with session_factory() as session:
            finished = session.scalars(
                select(Match).where(Match.status == MatchStatus.FINISHED)
            ).all()
        assert finished and all(m.home_score is not None for m in finished)

    def test_statistics_are_fetched_for_played_matches(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run(jobs.fetch_fixtures)

        run(jobs.fetch_statistics)

        row = last_run(session_factory, "jobs.fetch_statistics")
        assert row.status is JobStatus.SUCCEEDED
        assert "statistics_inserted" in row.detail

    def test_a_provider_without_statistics_skips_the_job(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """Otherwise every run fails and retries, for data this provider never has."""

        class NoStatistics(MockFootballDataProvider):
            @property
            def capabilities(self) -> frozenset[ProviderOperation]:
                return super().capabilities - {ProviderOperation.TEAM_STATISTICS}

        run(jobs.fetch_fixtures)
        runtime.configure(new_provider_factory=lambda _: NoStatistics(provider_dataset()))

        run(jobs.fetch_statistics)

        row = last_run(session_factory, "jobs.fetch_statistics")
        assert row.status is JobStatus.SKIPPED
        assert "statistics" in row.detail["reason"]

    def test_without_configured_competitions_the_job_is_skipped(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        runtime.configure(new_settings=make_settings(job_competitions=[]))

        result = run(jobs.fetch_fixtures)

        row = last_run(session_factory, "jobs.fetch_fixtures")
        assert row.status is JobStatus.SKIPPED
        assert "no competitions configured" in row.detail["reason"]
        assert "JOB_COMPETITIONS" in row.detail["reason"]
        assert "skipped" in result

    def test_provider_credentials_are_never_task_arguments(self) -> None:
        import inspect

        for task in (jobs.fetch_fixtures, jobs.fetch_results, jobs.fetch_statistics):
            parameters = set(inspect.signature(task.run).parameters) - {"self"}
            assert parameters == set(), task.name


class TestRetries:
    def test_an_unavailable_provider_is_retried_with_backoff(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[float] = []

        def failing(_: Settings) -> Any:
            raise ProviderUnavailableError("api down", provider="mock")

        def fake_retry(**kwargs: Any) -> Exception:
            attempts.append(kwargs["countdown"])
            return RuntimeError("retry scheduled")

        runtime.configure(new_provider_factory=failing)
        monkeypatch.setattr(jobs.fetch_results, "retry", lambda **kw: fake_retry(**kw))

        with pytest.raises(RuntimeError, match="retry scheduled"):
            jobs.fetch_results.apply(throw=True)

        assert attempts and 7 <= attempts[0] <= 10  # first backoff, jittered
        row = last_run(session_factory, "jobs.fetch_results")
        assert row.status is JobStatus.FAILED
        assert "api down" in (row.error or "")

    def test_a_rate_limit_waits_for_the_provider_s_retry_after(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        countdowns: list[float] = []

        def limited(_: Settings) -> Any:
            raise RateLimitError("too many", provider="mock", retry_after=90)

        runtime.configure(new_provider_factory=limited)
        monkeypatch.setattr(
            jobs.fetch_results,
            "retry",
            _collect(countdowns, "countdown"),
        )

        with pytest.raises(RuntimeError):
            jobs.fetch_results.apply(throw=True)

        assert countdowns == [90.0]

    def test_the_retry_limit_comes_from_settings(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[int] = []
        runtime.configure(
            new_settings=make_settings(job_competitions=["E0"], task_max_retries=2),
            new_provider_factory=lambda _: (_ for _ in ()).throw(ProviderUnavailableError("down")),
        )
        monkeypatch.setattr(
            jobs.fetch_results,
            "retry",
            _collect(seen, "max_retries"),
        )

        with pytest.raises(RuntimeError):
            jobs.fetch_results.apply(throw=True)

        assert seen == [2]


class TestFeatureAndPredictionJobs:
    def test_features_are_recalculated_for_finished_matches(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        run(jobs.fetch_fixtures)

        run(jobs.recalculate_features)

        with session_factory() as session:
            snapshots = session.scalars(select(FeatureSnapshot)).all()
        assert snapshots
        row = last_run(session_factory, "jobs.recalculate_features")
        assert row.status is JobStatus.SUCCEEDED
        assert row.detail["finished_matches"] > 0

    def test_predictions_are_refreshed_for_upcoming_matches(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        result = run(jobs.refresh_predictions)

        row = last_run(session_factory, "jobs.refresh_predictions")
        assert row.status is JobStatus.SUCCEEDED, row.detail
        assert row.detail["predicted"] >= 1
        assert row.detail["model"].startswith("elo ")
        with session_factory() as session:
            stored = session.scalars(select(Prediction)).all()
            upcoming = {
                match.id
                for match in session.scalars(
                    select(Match).where(Match.status == MatchStatus.SCHEDULED)
                )
            }
        assert {p.match_id for p in stored} <= upcoming
        assert result["predicted"] == len(stored)

    def test_running_twice_reuses_stored_predictions(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        run(jobs.refresh_predictions)

        run(jobs.refresh_predictions)

        row = last_run(session_factory, "jobs.refresh_predictions")
        assert row.detail["predicted"] == 0 and row.detail["reused"] >= 1

    def test_without_an_active_model_the_job_is_skipped(
        self, session_factory: sessionmaker[Session], league_in_database: None
    ) -> None:
        run(jobs.refresh_predictions)

        row = last_run(session_factory, "jobs.refresh_predictions")
        assert row.status is JobStatus.SKIPPED
        assert "no model is active" in row.detail["reason"]


class TestEvaluationJob:
    def test_predictions_are_scored_once_their_match_is_played(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        run(jobs.refresh_predictions)
        with session_factory() as session, session.begin():
            for match in session.scalars(
                select(Match).where(Match.status == MatchStatus.SCHEDULED)
            ):
                match.status = MatchStatus.FINISHED
                match.home_score, match.away_score = 2, 1
                match.kickoff_at = datetime.now(UTC) - timedelta(days=1)

        result = run(jobs.evaluate_predictions)

        row = last_run(session_factory, "jobs.evaluate_predictions")
        assert row.status is JobStatus.SUCCEEDED
        assert result["evaluated"] >= 1
        with session_factory() as session:
            results = session.scalars(select(PredictionResult)).all()
            predictions = {p.id: p for p in session.scalars(select(Prediction))}
        assert len(results) == result["evaluated"]
        for evaluation in results:
            prediction = predictions[evaluation.prediction_id]
            assert evaluation.outcome_correct == (
                max(
                    ("home", prediction.home_win_probability),
                    ("draw", prediction.draw_probability),
                    ("away", prediction.away_win_probability),
                    key=lambda pair: pair[1],
                )[0]
                == "home"
            )
            assert 0 <= float(evaluation.brier_score) <= 2
            assert float(evaluation.log_loss) >= 0

    def test_with_nothing_to_score_the_job_is_skipped(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        run(jobs.evaluate_predictions)

        row = last_run(session_factory, "jobs.evaluate_predictions")
        assert row.status is JobStatus.SKIPPED
        assert "waiting for a result" in row.detail["reason"]


class TestRetraining:
    def test_retraining_is_skipped_until_it_is_due(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        runtime.configure(
            new_settings=make_settings(job_competitions=["E0"], retrain_min_new_matches=10_000)
        )

        run(jobs.retrain_models)

        row = last_run(session_factory, "jobs.retrain_models")
        assert row.status is JobStatus.SKIPPED
        assert "new results since the last training" in row.detail["reason"]
        assert row.detail["new_matches"] >= 0

    def test_retraining_can_be_disabled(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        runtime.configure(
            new_settings=make_settings(job_competitions=["E0"], retrain_enabled=False)
        )

        run(jobs.retrain_models)

        row = last_run(session_factory, "jobs.retrain_models")
        assert row.status is JobStatus.SKIPPED
        assert "disabled" in row.detail["reason"]

    def test_a_recent_successful_retrain_blocks_the_next_one(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion
    ) -> None:
        runtime.configure(
            new_settings=make_settings(
                job_competitions=["E0"], retrain_min_new_matches=0, retrain_every_days=30
            )
        )
        with session_factory() as session, session.begin():
            session.add(
                JobRun(
                    task="jobs.retrain_models",
                    status=JobStatus.SUCCEEDED,
                    attempt=1,
                    started_at=datetime.now(UTC) - timedelta(days=2),
                    finished_at=datetime.now(UTC) - timedelta(days=2),
                    detail={},
                )
            )

        run(jobs.retrain_models)

        row = last_run(session_factory, "jobs.retrain_models")
        assert row.status is JobStatus.SKIPPED
        assert "days ago" in row.detail["reason"]

    def test_when_due_it_trains_and_registers_an_inactive_version(
        self, session_factory: sessionmaker[Session], active_elo: ModelVersion, tmp_path: Path
    ) -> None:
        runtime.configure(
            new_settings=make_settings(
                job_competitions=["E0"], retrain_min_new_matches=0, retrain_every_days=0.0001
            )
        )
        run(jobs.recalculate_features)

        run(jobs.retrain_models)

        row = last_run(session_factory, "jobs.retrain_models")
        assert row.status is JobStatus.SUCCEEDED, row.error or row.detail
        assert row.detail["trained"] is True
        with session_factory() as session:
            versions = session.scalars(select(ModelVersion)).all()
            active = [v for v in versions if v.is_active]
        assert len(versions) > 1  # the new one was registered
        assert [v.id for v in active] == [active_elo.id]  # activation stays deliberate


class TestFailureReport:
    def test_recent_failures_are_summarised(
        self, session_factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
    ) -> None:
        with session_factory() as session, session.begin():
            session.add_all(
                [
                    JobRun(
                        task="jobs.fetch_results",
                        status=JobStatus.FAILED,
                        attempt=3,
                        started_at=datetime.now(UTC) - timedelta(hours=1),
                        finished_at=datetime.now(UTC) - timedelta(hours=1),
                        detail={},
                        error="ProviderUnavailableError('api down')",
                    ),
                    JobRun(
                        task="jobs.fetch_results",
                        status=JobStatus.SUCCEEDED,
                        attempt=1,
                        started_at=datetime.now(UTC) - timedelta(minutes=30),
                        finished_at=datetime.now(UTC) - timedelta(minutes=30),
                        detail={},
                    ),
                ]
            )

        with caplog.at_level("ERROR"):
            result = run(jobs.report_failures)

        assert result["failed"] == 1
        assert result["tasks"]["jobs.fetch_results"] == {"failed": 1, "succeeded": 1}
        assert "1 job failure(s)" in caplog.text
        assert "api down" in caplog.text

    def test_a_quiet_window_reports_nothing_to_fix(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        result = run(jobs.report_failures)

        assert result["failed"] == 0
        row = last_run(session_factory, "jobs.report_failures")
        assert row.status is JobStatus.SUCCEEDED


@pytest.fixture
def league_in_database(session_factory: sessionmaker[Session]) -> None:
    """Every season imported, with a few matches still to be played."""
    from app.ingestion.sync import ProviderSync

    provider = MockFootballDataProvider(provider_dataset())
    runtime.configure(new_provider_factory=lambda _: provider)
    ProviderSync(provider, session_factory).sync_competition("E0", seasons="all")
    with session_factory() as session, session.begin():
        upcoming = list(session.scalars(select(Match).order_by(Match.kickoff_at.desc()).limit(2)))
        for match in upcoming:
            match.status = MatchStatus.SCHEDULED
            match.home_score = match.away_score = None
            match.kickoff_at = datetime.now(UTC) + timedelta(days=3)
    run(jobs.recalculate_features)


@pytest.fixture
def active_elo(
    session_factory: sessionmaker[Session], league_in_database: None, tmp_path: Path
) -> ModelVersion:
    """A real Elo model, registered and activated, as the jobs expect to find."""
    from app.features.config import FeatureConfig
    from app.ml.dataset import load_dataset
    from app.ml.predictors import EloModel
    from app.registry.service import ModelRegistry, NewModelVersion

    version = FeatureConfig().version()
    with session_factory() as session:
        data = load_dataset(session, version)
        competition = session.scalars(select(Competition)).first()
        assert competition is not None
        competition_id = competition.id
    model = EloModel().fit(data)
    directory = tmp_path / "elo"
    model.save(directory, {"feature_set_version": version})
    with session_factory() as session, session.begin():
        registry = ModelRegistry(session)
        row = registry.register(
            NewModelVersion(
                name="elo",
                version="v1",
                algorithm="EloModel",
                feature_set_version=version,
                dataset_version=data.fingerprint(),
                training_data_start=data.examples[0].kickoff_at.date(),
                training_data_end=data.examples[-1].kickoff_at.date(),
                competitions=[{"id": competition_id, "name": "Premier League"}],
                hyperparameters={"feature": "elo_diff"},
                metrics={"validation": {"log_loss": 1.0}},
                artifact_uri=directory.as_uri(),
                trained_at=datetime.now(UTC),
            )
        )
        registry.activate("elo", "v1", reason="tests")
        model_id = row.id
    with session_factory() as session:
        registered = session.get(ModelVersion, model_id)
        assert registered is not None
        session.expunge(registered)
        return registered


class TestPermanentFailures:
    def test_a_rejected_key_fails_without_retrying(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key the provider rejects is a real fault: recorded, never retried."""
        from app.providers.errors import AuthenticationError

        retries: list[Any] = []
        provider = MockFootballDataProvider(provider_dataset())
        monkeypatch.setattr(
            provider,
            "get_competitions",
            lambda: (_ for _ in ()).throw(AuthenticationError("invalid API key")),
        )
        runtime.configure(new_provider_factory=lambda _: provider)
        monkeypatch.setattr(jobs.fetch_fixtures, "retry", _collect(retries, "countdown"))

        with pytest.raises(AuthenticationError):
            jobs.fetch_fixtures.apply(throw=True)

        assert retries == []  # waiting cannot fix a rejected key
        row = last_run(session_factory, "jobs.fetch_fixtures")
        assert row.status is JobStatus.FAILED
        assert "invalid API key" in (row.error or "")

    def test_an_exhausted_quota_is_not_retried(
        self, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.providers.errors import QuotaExceededError

        retries: list[Any] = []
        runtime.configure(
            new_provider_factory=lambda _: (_ for _ in ()).throw(QuotaExceededError("daily limit"))
        )
        monkeypatch.setattr(jobs.fetch_results, "retry", _collect(retries, "countdown"))

        with pytest.raises(QuotaExceededError):
            jobs.fetch_results.apply(throw=True)

        assert retries == []
        assert last_run(session_factory, "jobs.fetch_results").status is JobStatus.FAILED

    def test_an_unconfigured_provider_is_skipped_not_failed(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """No key at all is a configuration state; a rejected key is a fault."""
        from app.providers.errors import AuthenticationError

        runtime.configure(
            new_provider_factory=lambda _: (_ for _ in ()).throw(
                AuthenticationError("API_FOOTBALL_API_KEY is not set")
            )
        )

        result = run(jobs.fetch_fixtures)

        row = last_run(session_factory, "jobs.fetch_fixtures")
        assert row.status is JobStatus.SKIPPED
        assert "the provider is not configured" in row.detail["reason"]
        assert "skipped" in result


class TestOverlapGuard:
    def test_a_tick_is_skipped_while_the_previous_run_is_still_going(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """The scheduler fires on a clock, not on whether the last run finished."""
        now = datetime.now(UTC)
        with session_factory() as session, session.begin():
            session.add(
                JobRun(
                    task="jobs.evaluate_predictions",
                    status=JobStatus.RUNNING,
                    attempt=1,
                    started_at=now - timedelta(minutes=2),
                    detail={},
                )
            )

        outcome = jobs.evaluate_predictions.apply().get()

        assert "skipped" in outcome
        assert "in progress" in outcome["skipped"]
        with session_factory() as session:
            statuses = [
                row.status
                for row in session.scalars(
                    select(JobRun).where(JobRun.task == "jobs.evaluate_predictions")
                )
            ]
        assert JobStatus.SKIPPED in statuses  # the skip is visible in job status

    def test_the_job_runs_when_nothing_else_is(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        """It may still skip for its own reasons — "nothing to evaluate" is one —
        but it is not held back by the guard, and it did its own work."""
        outcome = jobs.evaluate_predictions.apply().get()

        assert "in progress" not in outcome.get("skipped", "")
        assert "evaluated" in outcome
