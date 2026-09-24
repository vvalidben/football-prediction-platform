"""Run a scheduled job now, or see what the scheduler has been doing.

    python -m app.tasks run jobs.fetch_results     # run one job in this process
    python -m app.tasks run jobs.refresh_predictions --queue   # hand it to the worker
    python -m app.tasks status [--task jobs.fetch_results] [--hours 24]
    python -m app.tasks schedule                   # the configured schedule

The worker and scheduler themselves are started with Celery:

    celery -A app.tasks.celery_app:celery_app worker --loglevel=info
    celery -A app.tasks.celery_app:celery_app beat --loglevel=info
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.models import JobRun
from app.tasks import jobs as job_module
from app.tasks import runtime
from app.tasks.celery_app import SCHEDULES, create_celery

TASKS = {
    "jobs.fetch_fixtures": job_module.fetch_fixtures,
    "jobs.fetch_results": job_module.fetch_results,
    "jobs.fetch_statistics": job_module.fetch_statistics,
    "jobs.recalculate_features": job_module.recalculate_features,
    "jobs.refresh_predictions": job_module.refresh_predictions,
    "jobs.evaluate_predictions": job_module.evaluate_predictions,
    "jobs.retrain_models": job_module.retrain_models,
    "jobs.report_failures": job_module.report_failures,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.tasks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one job")
    run.add_argument("task", choices=sorted(TASKS))
    run.add_argument(
        "--queue", action="store_true", help="send it to the worker instead of running it here"
    )
    status = commands.add_parser("status", help="recent job runs")
    status.add_argument("--task", choices=sorted(TASKS))
    status.add_argument("--hours", type=float, default=24.0)
    status.add_argument("--limit", type=int, default=20)
    commands.add_parser("schedule", help="the configured schedule")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "schedule":
        print(f"jobs {'enabled' if settings.jobs_enabled else 'disabled'}")
        for task, field in SCHEDULES.items():
            minutes = getattr(settings, field)
            print(f"  {task:<30}every {minutes:>7.1f} min   ({field.upper()})")
        print(
            f"\nretraining: every {settings.retrain_every_days} days and at least "
            f"{settings.retrain_min_new_matches} new results "
            f"({'enabled' if settings.retrain_enabled else 'disabled'})"
        )
        return 0

    if args.command == "run":
        if args.queue:
            celery = create_celery(settings)
            result = celery.send_task(args.task)
            print(f"queued {args.task} as {result.id}")
            return 0
        outcome = TASKS[args.task].apply().get()
        print(json.dumps(outcome, indent=2, default=str))
        return 0

    since = datetime.now(UTC) - timedelta(hours=args.hours)
    with runtime.session_factory()() as session:
        query = select(JobRun).where(JobRun.started_at >= since)
        if args.task:
            query = query.where(JobRun.task == args.task)
        rows = list(session.scalars(query.order_by(JobRun.started_at.desc()).limit(args.limit)))
        counts = session.execute(
            select(JobRun.task, JobRun.status, func.count())
            .where(JobRun.started_at >= since)
            .group_by(JobRun.task, JobRun.status)
        ).all()
    print(f"last {args.hours:g} h:")
    for task, status_value, count in sorted(counts):
        print(f"  {task:<30}{status_value.value:<12}{count}")
    print(f"\n{'started':<22}{'task':<30}{'status':<11}{'ms':>8}  detail")
    for row in rows:
        detail = json.dumps(row.detail)[:70]
        print(
            f"{row.started_at:%Y-%m-%d %H:%M:%S}  {row.task:<30}{row.status.value:<11}"
            f"{row.duration_ms or 0:>8}  {row.error or detail}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
