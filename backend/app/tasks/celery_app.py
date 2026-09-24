"""The Celery application and its beat schedule.

Broker and result backend are the Redis instance the platform already runs; every
schedule comes from settings (minutes), so a deployment can slow a job down, speed
it up or switch the whole scheduler off without changing code:

    JOBS_ENABLED=true
    SCHEDULE_RESULTS_MINUTES=30
    JOB_COMPETITIONS=39,140

Run the worker and the scheduler from the backend image:

    celery -A app.tasks.celery_app:celery_app worker --loglevel=info
    celery -A app.tasks.celery_app:celery_app beat --loglevel=info

Provider credentials are never passed here: tasks build providers from settings
(`app.providers.registry.create_provider`), which read the environment.
"""

from datetime import timedelta
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from celery import Celery
from celery.schedules import schedule

from app.core.config import Settings, get_settings

#: Task name -> the setting holding its period, in minutes.
SCHEDULES: dict[str, str] = {
    "jobs.fetch_fixtures": "schedule_fixtures_minutes",
    "jobs.fetch_results": "schedule_results_minutes",
    "jobs.fetch_statistics": "schedule_statistics_minutes",
    "jobs.recalculate_features": "schedule_features_minutes",
    "jobs.refresh_predictions": "schedule_predictions_minutes",
    "jobs.evaluate_predictions": "schedule_evaluation_minutes",
    "jobs.retrain_models": "schedule_retrain_minutes",
    "jobs.report_failures": "schedule_failure_report_minutes",
}


def redis_url(settings: Settings, *, db_offset: int = 0) -> str:
    password = (
        f":{quote(settings.redis_password.get_secret_value(), safe='')}@"
        if settings.redis_password
        else ""
    )
    return (
        f"redis://{password}{settings.redis_host}:{settings.redis_port}"
        f"/{settings.redis_db + db_offset}"
    )


def broker_url(settings: Settings) -> str:
    return settings.celery_broker_url or redis_url(settings, db_offset=1)


def result_backend(settings: Settings) -> str:
    return settings.celery_result_backend or redis_url(settings, db_offset=2)


def beat_schedule(settings: Settings) -> dict[str, dict[str, Any]]:
    """Every job, with the period configured for it. Empty when jobs are disabled."""
    if not settings.jobs_enabled:
        return {}
    return {
        task: {
            "task": task,
            "schedule": schedule(timedelta(minutes=getattr(settings, field))),
            # A job that is already queued is not queued again when beat catches up.
            "options": {"expires": getattr(settings, field) * 60},
        }
        for task, field in SCHEDULES.items()
    }


def create_celery(settings: Settings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery("football", broker=broker_url(settings), backend=result_backend(settings))
    app.conf.update(
        task_default_queue="football",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        # The broker redelivers a task if the worker dies mid-run; jobs are written to
        # be safe to repeat (every write is an upsert or is keyed by its inputs).
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_time_limit=settings.task_time_limit_seconds,
        task_soft_time_limit=max(settings.task_time_limit_seconds - 30, 30),
        result_expires=timedelta(days=7),
        broker_connection_retry_on_startup=True,
        beat_schedule=beat_schedule(settings),
        imports=("app.tasks.jobs",),
    )
    return app


@lru_cache(maxsize=1)
def get_celery() -> Celery:
    return create_celery()


def __getattr__(name: str) -> Any:
    # `celery -A app.tasks.celery_app:celery_app` resolves the app lazily, so importing
    # this module never requires a full configuration (tests build their own).
    if name == "celery_app":
        return get_celery()
    raise AttributeError(name)
