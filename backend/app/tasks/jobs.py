"""The scheduled jobs.

Together they keep the platform current without anyone running a command:

1. `fetch_fixtures` — competitions, seasons, teams and the fixture list.
2. `fetch_results` — the same matches again, so finished scores arrive.
3. `fetch_statistics` — team statistics of recently played matches.
4. Each of those writes to the database through the idempotent Sprint 2 sync.
5. `recalculate_features` — snapshots for matches that do not have one.
6. `refresh_predictions` — predictions for upcoming matches, with the active model.
7. `evaluate_predictions` — scores predictions whose match has been played.

Plus `retrain_models` (only when retraining is due) and `report_failures`.

**Reliability.** A provider that is unavailable or rate-limited raises; the task
retries with exponential backoff and jitter, honouring `Retry-After` when the
provider sends one. Every run is recorded in `job_runs` (status, attempt, counts,
error), and jobs that cannot do anything useful are `skipped`, not failed.

**Credentials** come from settings only (`app.providers.registry`); no task takes a
key as an argument.
"""

import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import Task, shared_task
from sqlalchemy import func, select

from app.core.config import Settings
from app.features.__main__ import main as features_main
from app.models import JobRun, Match, ModelVersion
from app.models.enums import JobStatus, MatchStatus
from app.providers.base import ProviderOperation
from app.providers.errors import (
    AuthenticationError,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    UnsupportedOperationError,
)
from app.services.evaluation import evaluate_pending
from app.tasks import runtime
from app.tasks.recording import JobSkipped, record, record_skip, running_since

logger = logging.getLogger(__name__)

#: Retrying these cannot help: the key, the plan or the provider's capabilities have
#: to change first. They are recorded as failures so the failure report shows them.
PERMANENT = (AuthenticationError, QuotaExceededError, UnsupportedOperationError)
#: Errors worth another attempt: the provider or the network, not our data.
RETRYABLE = (ProviderError, ConnectionError, TimeoutError)


def retry_delay(settings: Settings, attempt: int) -> float:
    """Exponential backoff with jitter, capped: 10s, 20s, 40s, … by default."""
    base = settings.task_retry_backoff_seconds * (2 ** max(attempt - 1, 0))
    capped = min(base, settings.task_retry_backoff_max_seconds)
    jitter: float = random.uniform(0.7, 1.0)  # noqa: S311 - spreading retries, not security
    delay: float = round(capped * jitter, 3)
    return delay


def _retry(task: Task, exc: Exception, settings: Settings) -> None:
    """Retry with backoff; a rate limit waits as long as the provider asked."""
    attempt = (task.request.retries or 0) + 1
    delay = retry_delay(settings, attempt)
    if isinstance(exc, RateLimitError) and exc.retry_after:
        delay = max(float(exc.retry_after), delay)
    logger.warning(
        "Job %s attempt %s failed (%s); retrying in %.1fs",
        task.name,
        attempt,
        type(exc).__name__,
        delay,
    )
    raise task.retry(exc=exc, countdown=delay, max_retries=settings.task_max_retries)


def _competitions(settings: Settings) -> list[str]:
    if not settings.job_competitions:
        raise JobSkipped(
            "no competitions configured; set JOB_COMPETITIONS to the provider's "
            "competition ids (e.g. 39,140)"
        )
    return list(settings.job_competitions)


def _provider_sync(settings: Settings) -> Any:
    """The sync, once the jobs are configured for it.

    Which competitions to keep up to date is checked first (it needs no credentials),
    and a provider that is not configured at all is a skip, not a failure: a key that
    is rejected *by the provider* still fails, because that is a real fault.
    """
    _competitions(settings)
    from app.ingestion.sync import ProviderSync

    try:
        configured = runtime.provider()
    except AuthenticationError as exc:
        raise JobSkipped(f"the provider is not configured: {exc}") from exc
    return ProviderSync(configured, runtime.session_factory())


def _run(task: Task, name: str, body: Callable[[Any, Settings], None]) -> dict[str, Any]:
    """Record the run, retry what is worth retrying."""
    settings = runtime.settings()
    attempt = (task.request.retries or 0) + 1
    outcome: dict[str, Any] = {}
    factory = runtime.session_factory()
    started = running_since(
        factory, name, stale_after=timedelta(seconds=settings.task_time_limit_seconds)
    )
    if started is not None and attempt == 1:
        # A retry is the same piece of work coming back, so it is allowed through;
        # a fresh schedule tick while the last run is still going is not.
        reason = f"another run has been in progress since {started.isoformat()}"
        record_skip(factory, name, reason, attempt=attempt, task_id=task.request.id)
        logger.warning("Job %s %s; this tick is skipped", name, reason)
        return {"skipped": reason}
    try:
        with record(name, factory, attempt=attempt, task_id=task.request.id) as report:
            try:
                body(report, settings)
            except JobSkipped as skipped:
                # `record` marks the run skipped and stops the exception there, so the
                # outcome has to be captured before it leaves this block.
                outcome = {"skipped": str(skipped), **report.values}
                raise
            outcome = dict(report.values)
    except PERMANENT:
        # No retry: the configuration has to change first.
        logger.error("Job %s cannot succeed as configured", name)
        raise
    except RETRYABLE as exc:
        _retry(task, exc, settings)
        raise  # pragma: no cover - retry always raises
    return outcome


# 1-4: provider data into the database ---------------------------------------------------


@shared_task(bind=True, name="jobs.fetch_fixtures")
def fetch_fixtures(self: Task) -> dict[str, Any]:
    """Competitions, seasons, teams and fixtures of the configured competitions."""

    def body(report: Any, settings: Settings) -> None:
        sync = _provider_sync(settings)
        for competition in _competitions(settings):
            for run in sync.sync_competition(competition):
                report.increment(
                    **{
                        f"{run.job}_inserted": run.inserted,
                        f"{run.job}_updated": run.updated,
                        f"{run.job}_failed": run.failed,
                    }
                )
        report.add(competitions=len(_competitions(settings)))

    return _run(self, "jobs.fetch_fixtures", body)


@shared_task(bind=True, name="jobs.fetch_results")
def fetch_results(self: Task) -> dict[str, Any]:
    """Re-read the current season's matches, so finished scores arrive.

    The provider returns fixtures and results together; this job runs far more often
    than `fetch_fixtures` and touches only the matches of the current season.
    """

    def body(report: Any, settings: Settings) -> None:
        sync = _provider_sync(settings)
        for competition in _competitions(settings):
            for season in sync.select_seasons(competition, "current"):
                run = sync.sync_matches(competition, season)
                report.increment(
                    matches_inserted=run.inserted,
                    matches_updated=run.updated,
                    matches_failed=run.failed,
                )
        report.add(competitions=len(_competitions(settings)))

    return _run(self, "jobs.fetch_results", body)


@shared_task(bind=True, name="jobs.fetch_statistics")
def fetch_statistics(self: Task) -> dict[str, Any]:
    """Team statistics of the current season's played matches.

    `sync_team_statistics` picks the matches that still need statistics itself, so
    this job only has to run often enough to catch up after each matchday.
    """

    def body(report: Any, settings: Settings) -> None:
        sync = _provider_sync(settings)
        if not sync.provider.supports(ProviderOperation.TEAM_STATISTICS):
            raise JobSkipped(f"{sync.provider.name} offers no team statistics")
        for competition in _competitions(settings):
            for season in sync.select_seasons(competition, "current"):
                run = sync.sync_team_statistics(competition, season)
                report.increment(
                    statistics_inserted=run.inserted,
                    statistics_updated=run.updated,
                    statistics_failed=run.failed,
                )
        report.add(competitions=len(_competitions(settings)))

    return _run(self, "jobs.fetch_statistics", body)


# 5: features -----------------------------------------------------------------------------


@shared_task(bind=True, name="jobs.recalculate_features")
def recalculate_features(self: Task) -> dict[str, Any]:
    """Feature snapshots for finished matches that do not have one yet."""

    def body(report: Any, settings: Settings) -> None:
        code = features_main(["generate"], session_factory=runtime.session_factory())
        if code != 0:
            raise RuntimeError(f"feature generation failed with exit code {code}")
        with runtime.session_factory()() as session:
            report.add(
                finished_matches=session.scalar(
                    select(func.count())
                    .select_from(Match)
                    .where(Match.status == MatchStatus.FINISHED)
                )
            )

    return _run(self, "jobs.recalculate_features", body)


# 6: predictions ---------------------------------------------------------------------------


@shared_task(bind=True, name="jobs.refresh_predictions")
def refresh_predictions(self: Task) -> dict[str, Any]:
    """Predict every upcoming match within the horizon with the active model."""

    def body(report: Any, settings: Settings) -> None:
        from app.services.predictions import (
            PredictionError,
            PredictionService,
            select_model,
        )

        now = datetime.now(UTC)
        horizon = now + timedelta(days=settings.prediction_horizon_days)
        factory = runtime.session_factory()
        with factory() as session, session.begin():
            matches = list(
                session.scalars(
                    select(Match)
                    .where(
                        Match.kickoff_at >= now,
                        Match.kickoff_at < horizon,
                        Match.status == MatchStatus.SCHEDULED,
                    )
                    .order_by(Match.kickoff_at)
                )
            )
            report.add(upcoming=len(matches), horizon_days=settings.prediction_horizon_days)
            if not matches:
                raise JobSkipped("no upcoming matches within the horizon")
            try:
                model_version = select_model(session, None)
                service = PredictionService(factory, cache=runtime.model_cache())
                outcomes = service.predict(session, matches, model_version, now=now)
            except PredictionError as exc:
                raise JobSkipped(str(exc)) from exc
            report.add(
                model=f"{model_version.name} {model_version.version}",
                predicted=sum(1 for outcome in outcomes if outcome.created),
                reused=sum(1 for outcome in outcomes if not outcome.created),
            )

    return _run(self, "jobs.refresh_predictions", body)


# 7: evaluation ------------------------------------------------------------------------------


@shared_task(bind=True, name="jobs.evaluate_predictions")
def evaluate_predictions(self: Task) -> dict[str, Any]:
    """Score stored predictions whose match has been played."""

    def body(report: Any, settings: Settings) -> None:
        with runtime.session_factory()() as session, session.begin():
            result = evaluate_pending(session)
        report.add(evaluated=result.evaluated, matches=len(set(result.match_ids)))
        if result.evaluated == 0:
            raise JobSkipped("no predictions are waiting for a result")

    return _run(self, "jobs.evaluate_predictions", body)


# Retraining ----------------------------------------------------------------------------------


def retrain_due(session: Any, settings: Settings, *, now: datetime) -> tuple[bool, str, int]:
    """Whether retraining is due: enough time *and* enough new results.

    Retraining after every match would be wasteful and would churn the registry, so
    both thresholds are configurable and both must be met.
    """
    if not settings.retrain_enabled:
        return False, "retraining is disabled (RETRAIN_ENABLED=false)", 0
    last = session.scalar(
        select(func.max(JobRun.finished_at)).where(
            JobRun.task == "jobs.retrain_models", JobRun.status == JobStatus.SUCCEEDED
        )
    )
    reference = last or session.scalar(select(func.min(ModelVersion.trained_at)))
    new_matches = session.scalar(
        select(func.count())
        .select_from(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            Match.kickoff_at > (reference or datetime.min.replace(tzinfo=UTC)),
        )
    )
    new_matches = int(new_matches or 0)
    if last is not None:
        days = (now - last).total_seconds() / 86400
        if days < settings.retrain_every_days:
            return (
                False,
                f"last retrained {days:.1f} days ago; every {settings.retrain_every_days} days",
                new_matches,
            )
    if new_matches < settings.retrain_min_new_matches:
        return (
            False,
            f"only {new_matches} new results since the last training; "
            f"{settings.retrain_min_new_matches} needed",
            new_matches,
        )
    return True, f"{new_matches} new results", new_matches


@shared_task(bind=True, name="jobs.retrain_models")
def retrain_models(self: Task) -> dict[str, Any]:
    """Retrain and register a model when retraining is due.

    Never after every match: both `RETRAIN_EVERY_DAYS` and `RETRAIN_MIN_NEW_MATCHES`
    must be satisfied. The new version is registered **inactive**; activating it
    stays a deliberate step (see docs/model-registry.md).
    """

    def body(report: Any, settings: Settings) -> None:
        now = datetime.now(UTC)
        with runtime.session_factory()() as session:
            due, reason, new_matches = retrain_due(session, settings, now=now)
        report.add(new_matches=new_matches, decision=reason)
        if not due:
            raise JobSkipped(f"retraining is not due: {reason}")
        try:
            from app.ml.__main__ import main as ml_main
        except ImportError as exc:
            raise JobSkipped(
                f"this worker cannot train models ({exc}); build the image with the "
                "'ml' extra to enable retraining"
            ) from exc
        code = ml_main(
            ["train", "--cv-folds", "2", "--register", "--experiment", "scheduled-retraining"],
            session_factory=runtime.session_factory(),
        )
        if code != 0:
            raise RuntimeError(f"training failed with exit code {code}")
        report.add(trained=True)

    return _run(self, "jobs.retrain_models", body)


# Failure reporting -------------------------------------------------------------------------


@shared_task(bind=True, name="jobs.report_failures")
def report_failures(self: Task) -> dict[str, Any]:
    """Summarise recent job failures, so a broken schedule is noticed."""

    def body(report: Any, settings: Settings) -> None:
        since = datetime.now(UTC) - timedelta(hours=settings.failure_report_hours)
        with runtime.session_factory()() as session:
            rows = session.execute(
                select(JobRun.task, JobRun.status, func.count())
                .where(JobRun.started_at >= since)
                .group_by(JobRun.task, JobRun.status)
            ).all()
            failures = list(
                session.scalars(
                    select(JobRun)
                    .where(JobRun.started_at >= since, JobRun.status == JobStatus.FAILED)
                    .order_by(JobRun.started_at.desc())
                    .limit(20)
                )
            )
        counts: dict[str, dict[str, int]] = {}
        for task_name, status, total in rows:
            counts.setdefault(task_name, {})[str(status.value)] = int(total)
        report.add(window_hours=settings.failure_report_hours, tasks=counts)
        report.add(failed=len(failures))
        if failures:
            logger.error(
                "%s job failure(s) in the last %.0f h: %s",
                len(failures),
                settings.failure_report_hours,
                "; ".join(
                    f"{row.task} at {row.started_at:%Y-%m-%d %H:%M} ({row.error})"
                    for row in failures
                ),
            )
        silent = [task for task in counts if not counts[task]]
        if silent:  # pragma: no cover - defensive
            logger.warning("Jobs with no runs in the window: %s", silent)

    return _run(self, "jobs.report_failures", body)
