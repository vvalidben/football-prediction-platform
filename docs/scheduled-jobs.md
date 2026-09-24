# Scheduled jobs

The platform keeps itself up to date: it fetches new fixtures, results and
statistics, recalculates features, refreshes predictions, scores the ones whose
match has been played, and retrains only when retraining is due. Celery runs the
jobs over the Redis the platform already has; `celery beat` schedules them. Code:
`backend/app/tasks/`.

## The jobs

| Task | What it does | Default period |
| --- | --- | --- |
| `jobs.fetch_fixtures` | Competitions, seasons, teams and fixtures of the configured competitions | 6 h |
| `jobs.fetch_results` | The current season's matches again, so finished scores arrive | 1 h |
| `jobs.fetch_statistics` | Team statistics for matches that still lack them | 3 h |
| `jobs.recalculate_features` | Feature snapshots for matches that do not have one | 2 h |
| `jobs.refresh_predictions` | Predictions for upcoming matches with the active model | 3 h |
| `jobs.evaluate_predictions` | Scores predictions whose match has been played | 2 h |
| `jobs.retrain_models` | Retrains **only when due** (see below) | daily check |
| `jobs.report_failures` | Summarises recent failures into the log | daily |

Writing to the database is not a separate job: the three fetch jobs write through
the idempotent Sprint 2 sync, which upserts and records every run in
`data_ingestion_runs`.

## Configuration

Everything is environment variables; nothing is hard-coded, least of all
credentials. Providers are built by `app.providers.registry` from settings, and no
task takes a key as an argument (a test asserts the tasks have no parameters).

```bash
JOBS_ENABLED=true              # false leaves beat with an empty schedule
JOB_COMPETITIONS=39,140        # provider competition ids; empty means the fetch jobs skip
JOB_PROVIDER=api_football
SCHEDULE_RESULTS_MINUTES=60    # one per job, in minutes
PREDICTION_HORIZON_DAYS=7
RETRAIN_ENABLED=true
RETRAIN_EVERY_DAYS=30
RETRAIN_MIN_NEW_MATCHES=100
TASK_MAX_RETRIES=5
TASK_RETRY_BACKOFF_SECONDS=10
TASK_RETRY_BACKOFF_MAX_SECONDS=900
# CELERY_BROKER_URL / CELERY_RESULT_BACKEND default to the platform's Redis,
# databases +1 and +2.
```

## Running them

Docker Compose starts a `worker` and a `beat` service from the backend image:

```bash
docker compose up -d worker beat
docker compose logs -f worker
```

By hand, or to try one job now:

```bash
celery -A app.tasks.celery_app:celery_app worker --loglevel=info
celery -A app.tasks.celery_app:celery_app beat --loglevel=info --schedule=/tmp/celerybeat-schedule

python -m app.tasks schedule                      # what is configured
python -m app.tasks run jobs.fetch_results        # run it here, now
python -m app.tasks run jobs.fetch_results --queue  # hand it to the worker
python -m app.tasks status --hours 24             # what has been running
```

## Reliability

- **Retries with exponential backoff and jitter.** A provider that is unavailable,
  or the network, is retried: 10 s, 20 s, 40 s … up to `TASK_RETRY_BACKOFF_MAX_SECONDS`,
  each multiplied by a random 0.7–1.0 so workers do not retry in lockstep, at most
  `TASK_MAX_RETRIES` times.
- **Rate limits.** `RateLimitError` carries the provider's `Retry-After`; the job
  waits at least that long, even when the backoff would be shorter.
- **Permanent problems are not retried.** A rejected key, an exhausted quota or an
  unsupported operation cannot be fixed by waiting: the job fails once, loudly.
- **Not configured is not a failure.** Without competitions configured, or without a
  provider key at all, the fetch jobs are recorded as `skipped` with the reason, so
  an unconfigured deployment does not look broken. A key the provider *rejects*
  still fails.
- **Interrupted work is redelivered.** `task_acks_late` means a job whose worker dies
  is given to another; every job is safe to repeat (writes are upserts, predictions
  are keyed by their feature snapshot).
- **Job status** is a `job_runs` row per run: task, status (running, succeeded,
  skipped, failed), attempt, start, finish, duration, counts, and the error.

```
GET /jobs?task=jobs.fetch_results&status=failed   # recent runs
GET /jobs/summary?hours=24                        # per-task counts and failures
GET /jobs/{run_id}
```

- **Failure reporting.** `jobs.report_failures` summarises the window and logs the
  failures at ERROR with their messages; `/jobs/summary` gives the same view live,
  including `failing_tasks`.

## Retraining

Retraining after every match would churn the registry for no measurable gain
(Sprints 6 and 8 showed how small the differences between models are). Both
thresholds must be met:

- at least `RETRAIN_EVERY_DAYS` since the last successful retrain, **and**
- at least `RETRAIN_MIN_NEW_MATCHES` results since then.

Otherwise the job is `skipped` with the arithmetic in its reason, for example
*"only 12 new results since the last training; 100 needed"*. When it does run, the
new version is registered **inactive**: activating it stays a deliberate step
([model-registry.md](model-registry.md)).

A worker built with only the serving extra cannot train; the job then skips with
*"this worker cannot train models … build the image with the 'ml' extra"* rather
than failing. The scheduled-retraining path is tested end to end (it trains,
registers a new version, and leaves the active one alone).

## Overlapping runs

The scheduler fires on a clock, not on whether the last run finished. If a job
takes longer than its interval, a tick that finds the same job still running is
**skipped and recorded as skipped**, rather than starting a second copy — two
fixture fetches would spend the provider quota twice for the same data.

A run is treated as live only until the task time limit (`TASK_TIME_LIMIT_SECONDS`)
has passed, so a row left behind by a worker that died does not block the job
forever. A Celery retry is the same work coming back and is always allowed through.

## Tests (`backend/tests/tasks/`, 56 tests)

- **Configuration:** broker and backend URLs (including a password with characters
  that need escaping), every job scheduled from its setting, the scheduler switched
  off, worker settings that make redelivery safe, and every scheduled task existing.
- **Backoff:** growth, cap, jitter, and a rate limit's `Retry-After` winning.
- **Operator CLI** (`python -m app.tasks`): running a job in this process and
  queueing it to the worker, the recent-run listing with its window, task filter and
  cap, the printed schedule, and that every scheduled task can be run by hand.
- **Recording:** a successful run with its counts, a skip with its reason, a failure
  with its error and the exception still propagating, one row per attempt, and a
  running job visible before it finishes.
- **Each job** against a real database with a mock provider: fixtures imported,
  results updated, statistics fetched, features recalculated, predictions refreshed
  (and reused on a second run), predictions scored once their match is played, and
  each job's skip conditions.
- **Retries:** an unavailable provider retried with a jittered backoff, a rate limit
  honoured, the retry limit taken from settings, a rejected key and an exhausted
  quota never retried.
- **Retraining:** skipped until due, when disabled, and after a recent retrain; when
  due it trains and registers an inactive version.
- **Failure reporting** and the `/jobs` endpoints.

## Verified in Docker

`worker` and `beat` run from the backend image. With no API key configured, the
fetch jobs record themselves as skipped with the reason; `jobs.report_failures`
and the `/jobs` endpoints show the history:

```
$ docker compose exec backend python -m app.tasks run jobs.fetch_results --queue
queued jobs.fetch_results as 5cae21cb…
$ curl -s localhost:8000/jobs?limit=1
jobs.fetch_results  failed  attempt 1  AuthenticationError('API_FOOTBALL_API_KEY is not set')
```

(That run predates the "not configured is a skip" refinement; with it, the same
state is recorded as `skipped` with the same explanation.)

## Limitations

- One worker queue; no priorities or per-provider concurrency limits.
- Beat's schedule file lives in the container (`/tmp`), so a restarted scheduler may
  run a job earlier than its period suggests.
- The fetch jobs keep the **current** season up to date; historical backfill is
  still `python -m app.ingestion backfill`.
- Retraining in the container needs an image built with the `ml` extra; the default
  image serves models but does not train them.
- No alerting integration: failures are logged and exposed through `/jobs`;
  forwarding them to email or chat is future work.
