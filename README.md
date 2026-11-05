# Football Prediction Platform

A web application that will predict football matches from historical data using
machine-learning models: exact scores, home/draw/away probabilities, expected goals,
over/under, both-teams-to-score, score distributions and historical evaluation across
multiple competitions.

**Current state: complete.** The stack runs end to end (React →
FastAPI → PostgreSQL/Redis), the full database schema is in place
([docs/database.md](docs/database.md)), and a provider-independent ingestion layer
syncs data idempotently, with API-Football as the first adapter
([docs/data-providers.md](docs/data-providers.md)). A historical backfill imports
every season a provider offers, including team statistics, with normalisation,
validation, a data-quality issue log and a quality report
([docs/historical-pipeline.md](docs/historical-pipeline.md)). A leak-free feature
system builds versioned, reproducible feature snapshots for every match
([docs/features.md](docs/features.md)). Four baseline models (historical, Elo,
logistic regression, Poisson) and three tree ensembles (random forest, XGBoost,
LightGBM) share one model interface and are tuned with time-aware cross-validation,
compared chronologically and tracked in MLflow ([docs/models.md](docs/models.md),
[docs/model-comparison.md](docs/model-comparison.md)). A goal prediction engine turns
any model's exact-score distribution into expected goals, the most probable score,
1X2, over/under and both-teams-to-score, all derived from one matrix
([docs/prediction-engine.md](docs/prediction-engine.md)). Ensembles (simple and
weighted averages, stacking) are fitted on out-of-fold validation predictions and
recommended only when they measurably beat the best single model
([docs/ensembles.md](docs/ensembles.md)). A walk-forward backtest predicts every
historical match from the past only, stores each prediction before its result is
revealed, and measures calibration, with Platt and isotonic calibration fitted only
on earlier predictions ([docs/backtesting.md](docs/backtesting.md)). Every trained
model is registered with its algorithm, feature and dataset versions, training
period, competitions, hyperparameters and metrics, with one active version per model
name, activation history, rollback and loading, mirrored into the MLflow Model
Registry ([docs/model-registry.md](docs/model-registry.md)). A REST API serves
competitions, teams, matches, model metadata and predictions: each prediction is
computed from features built on demand, stored, and returned with its model version
and data freshness ([docs/api.md](docs/api.md)). Celery jobs keep everything current
on configurable schedules: fixtures, results and statistics are fetched, features
recalculated, predictions refreshed and scored, and models retrained only when due
([docs/scheduled-jobs.md](docs/scheduled-jobs.md)). The React application presents
all of it in five pages — dashboard, match prediction, team, model performance and
data status — with charts, explicit loading and error states, and empty states that
say what is missing rather than showing invented numbers
([docs/frontend.md](docs/frontend.md)). Every prediction can be opened up: exact
feature contributions (linear Shapley values, TreeSHAP for the tree models) show
what pushed the model's score up and what pushed it down, stated as model
behaviour rather than cause, and refused outright for models where no method is
valid ([docs/explainability.md](docs/explainability.md)). Every prediction is kept as
it was served and scored against the result once the match is played, and the
history page shows what was predicted, what happened and how wrong it was,
filtered by competition, season, team, model or date
([docs/prediction-history.md](docs/prediction-history.md)). The system has been
reviewed end to end for production: per-client rate limits, request ids in every
log line and error, security headers and a content-security policy, brief caching
of the status panels, measured database indexes, hardened containers, an
automated secret scan and a clean dependency audit
([docs/operations.md](docs/operations.md)). The suite now stands at 887 backend
and 145 frontend tests at 97% backend line coverage, with the eight guarantees
the platform must not break — leak-free features, no post-match information in a
prediction, normalised distributions, reproducibility, missing data, unknown
teams, model loading and chronological backtesting — stated and proven in one
place ([docs/testing.md](docs/testing.md)). The guides below cover installation, data, models, the API and deployment, and `docker-compose.prod.yml` configures a production stack ([DEPLOYMENT.md](DEPLOYMENT.md)) — nothing has been deployed. A final audit reviewed the whole system, fixed four
measured defects and states plainly what the platform does not do — it produces
probability distributions, not reliable exact scores ([AUDIT.md](AUDIT.md)). See
[ARCHITECTURE.md](ARCHITECTURE.md) for
how the system is put together.

| Layer          | Technology                                                     |
| -------------- | -------------------------------------------------------------- |
| Backend        | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic       |
| Database       | PostgreSQL 17                                                  |
| Cache / queues | Redis 7.4                                                      |
| Frontend       | React 19, TypeScript, Vite, Tailwind CSS v4                    |
| Infrastructure | Docker, Docker Compose, nginx (serves the built frontend)      |
| Quality        | pytest, Vitest + Testing Library, Ruff, mypy (strict), ESLint, Prettier |


## Documentation

| Guide | What it answers |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pieces fit together and why |
| [DATA.md](DATA.md) | Providers, ingestion, the database, features, adding a competition or a provider |
| [ML.md](ML.md) | Models, training, evaluation, backtesting, serving, explanations |
| [API.md](API.md) | Every endpoint, request and response, errors, rate limits |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production configuration, deployment options, upgrades, backups |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setting up, house rules, migrations, conventions |
| [AUDIT.md](AUDIT.md) | The final audit: what the system is, what it cannot do, findings and fixes |
| [docs/](docs/README.md) | The reference pages behind each of the above |

## Repository layout

```
.
├── backend/              FastAPI service (app/, alembic/, tests/, pyproject.toml)
├── frontend/             React + Vite single-page app (src/, tests next to code)
├── docker/               Dockerfiles, container entrypoint and nginx config
│   ├── backend/
│   └── frontend/
├── DATA.md ML.md API.md DEPLOYMENT.md CONTRIBUTING.md   Guides (see above)
├── docker-compose.yml            Development stack
├── docker-compose.prod.yml       Production stack (standalone)
├── docs/                 database.md, data-providers.md, historical-pipeline.md, features.md, models.md, model-comparison.md, prediction-engine.md, ensembles.md, backtesting.md, model-registry.md, api.md, scheduled-jobs.md, frontend.md, explainability.md, prediction-history.md, operations.md, testing.md
├── scripts/              Developer tooling (check.py runs every linter and test)
├── tests/smoke/          End-to-end smoke test against the running stack
├── docker-compose.yml    Local stack: postgres, redis, backend, frontend
├── .env.example          Template for all configuration
├── README.md
└── ARCHITECTURE.md
```

## Quick start (Docker)

Prerequisites: Docker Desktop (or Docker Engine) with Compose v2.

```bash
cp .env.example .env
# Edit .env: replace both `change-me` passwords with your own values.

docker compose up -d --build --wait
```

Then open:

| URL                                  | What                                            |
| ------------------------------------ | ----------------------------------------------- |
| http://127.0.0.1:8080                | Frontend (dashboard with live system status)    |
| http://127.0.0.1:8000/health         | Backend health check                            |
| http://127.0.0.1:8000/docs           | OpenAPI docs (disabled when `ENVIRONMENT=production`) |
| http://127.0.0.1:8000/competitions   | Competitions, teams, matches, models ([docs/api.md](docs/api.md)) |

```bash
# Predict a match (the active model version is used; see docs/model-registry.md)
curl -s http://127.0.0.1:8000/matches/upcoming | head
curl -s -X POST http://127.0.0.1:8000/predictions/match      -H 'content-type: application/json' -d '{"match_id": 1}'
curl -s -X POST http://127.0.0.1:8000/predictions/match      -H 'content-type: application/json'      -d '{"home_team": "Arsenal", "away_team": "Chelsea", "date": "2026-05-24"}'
```

Verify the whole stack:

```bash
python tests/smoke/check_stack.py
```

Useful commands:

```bash
docker compose ps                    # status and health of each service
docker compose logs -f backend       # follow backend logs
docker compose down                  # stop (data volumes are kept)
docker compose down -v               # stop and delete PostgreSQL/Redis data
```

All published ports bind to `127.0.0.1` only. If a port is taken on your machine,
change `POSTGRES_HOST_PORT`, `REDIS_HOST_PORT`, `BACKEND_HOST_PORT` or
`FRONTEND_HOST_PORT` in `.env`.

For a production deployment use `docker-compose.prod.yml` and follow
[DEPLOYMENT.md](DEPLOYMENT.md) — it publishes no database ports, withholds
`/docs`, and pins images by tag.

## Local development (without containerising the app)

The quickest loop is to run PostgreSQL and Redis in Docker and the backend and
frontend directly on your machine, with hot reload.

```bash
cp .env.example .env              # if you have not already
docker compose up -d --wait postgres redis
```

### Backend

Requires Python 3.12+.

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev,ml]"      # ml = numpy, scipy, scikit-learn, mlflow

alembic upgrade head
uvicorn app.main:create_app --factory --reload --port 8000
```

The backend reads the repository-root `.env` automatically. Real environment
variables take precedence over it.

### Frontend

Requires Node.js 22+.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`. Set
`VITE_API_PROXY_TARGET` to point it elsewhere.

### Database migrations

```bash
cd backend
alembic upgrade head                                     # apply all migrations
alembic revision --autogenerate -m "describe the change" # after editing app/models/
alembic check                                            # models and migrations agree?
alembic downgrade -1
```

The database tests (`backend/tests/db/`) create and drop their own
`<POSTGRES_DB>_test` database, so they never touch your development data.

In Docker, the backend container runs `alembic upgrade head` on start (set
`RUN_MIGRATIONS=false` on the service to disable that).

### Importing football data

Set `API_FOOTBALL_API_KEY` in `.env`, then:

```bash
cd backend
python -m app.ingestion competitions --country Morocco   # find competition ids
python -m app.ingestion sync --competition 39            # Premier League, current season
python -m app.ingestion backfill --competition 39        # all seasons, with statistics
python -m app.ingestion audit                            # data-quality checks (no key needed)
python -m app.ingestion report                           # data-quality report (no key needed)
# or inside Docker:
docker compose exec backend python -m app.ingestion sync --competition 39
```

Then build features (no key needed):

```bash
python -m app.features generate                          # snapshots for every match
python -m app.features show --match-id 1                 # inspect one match's features
python -m app.ml train --cv-folds 3 --report report.md  # all models, tracked in MLflow
```

Syncs are idempotent and logged in `data_ingestion_runs`. See
[docs/data-providers.md](docs/data-providers.md) for details.

## Environment variables

All configuration comes from environment variables. `.env.example` lists every one
of them, and `.env.production.example` the production set. `.env` and
`.env.production` are git-ignored; never commit real secrets — a test scans for
them ([docs/operations.md](docs/operations.md)).

| Variable                       | Default         | Description |
| ------------------------------ | --------------- | ----------- |
| `ENVIRONMENT`                  | `development`   | `development`, `test`, `staging` or `production`. `production` disables `/docs`. |
| `LOG_LEVEL`                    | `INFO`          | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `CORS_ALLOWED_ORIGINS`         | *(empty)*       | Comma-separated origins allowed to call the API directly. Not needed behind the Vite/nginx proxy. |
| `DEPENDENCY_TIMEOUT_SECONDS`   | `2`             | Connect/read timeout for PostgreSQL and Redis. |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `3`             | Overall deadline for `/health` dependency checks. |
| `POSTGRES_HOST`                | `127.0.0.1`     | Compose overrides this to `postgres`. |
| `POSTGRES_PORT`                | `5432`          | |
| `POSTGRES_USER`                | **required**    | |
| `POSTGRES_PASSWORD`            | **required**    | |
| `POSTGRES_DB`                  | **required**    | |
| `REDIS_HOST`                   | `127.0.0.1`     | Compose overrides this to `redis`. |
| `REDIS_PORT`                   | `6379`          | |
| `REDIS_DB`                     | `0`             | |
| `REDIS_PASSWORD`               | *(none)*        | Required by Compose, which starts Redis with `requirepass`. |
| `POSTGRES_HOST_PORT`, `REDIS_HOST_PORT`, `BACKEND_HOST_PORT`, `FRONTEND_HOST_PORT` | `5432`, `6379`, `8000`, `8080` | Host ports published by Compose. |
| `API_FOOTBALL_API_KEY`         | *(none)*        | API-Football key. Required only for ingestion commands. |
| `API_FOOTBALL_BASE_URL`        | `https://v3.football.api-sports.io` | |
| `API_FOOTBALL_REQUESTS_PER_MINUTE` | `10`        | Client-side pacing; set to your plan's limit (Free 10, Pro 300). |
| `PROVIDER_TIMEOUT_SECONDS`     | `10`            | Per-request timeout for provider calls. |
| `PROVIDER_MAX_ATTEMPTS`        | `4`             | Attempts per provider request (first try + retries). |
| `RATE_LIMIT_ENABLED`           | `true`          | Per-client rate limiting (counters in Redis; skipped if Redis is down). |
| `RATE_LIMIT_REQUESTS`          | `300`           | Requests per window from one address, except predictions. |
| `RATE_LIMIT_PREDICT_REQUESTS`  | `30`            | Requests per window for `POST /predictions/*`. |
| `RATE_LIMIT_WINDOW_SECONDS`    | `60`            | Length of the rate-limit window. |
| `CACHE_ENABLED`                | `true`          | Caching of `/data/status` and `/data/freshness`. |
| `STATUS_CACHE_SECONDS`         | `30`            | How long those two answers are cached. `0` disables it. |
| `JOBS_ENABLED`                 | `true`          | Master switch for the scheduled background jobs. |
| `JOB_COMPETITIONS`             | *(empty)*       | Provider competition ids the jobs keep current. Empty means the fetch jobs skip. |
| `JOB_PROVIDER`                 | `api_football`  | Which provider the jobs use. |
| `SCHEDULE_*_MINUTES`           | see `.env.example` | Interval of each job, in minutes. |
| `RETRAIN_ENABLED`, `RETRAIN_EVERY_DAYS`, `RETRAIN_MIN_NEW_MATCHES` | `true`, `30`, `100` | Retraining cadence: both conditions must hold. |
| `TASK_MAX_RETRIES`, `TASK_RETRY_BACKOFF_SECONDS` | `5`, `10` | Job retries and their exponential backoff. |
| `WORKER_CONCURRENCY`           | `2`             | Celery worker processes. |
| `RUN_MIGRATIONS`               | `true`          | The backend container applies migrations on start. |
| `DATABASE_POOL_SIZE`           | `5`             | SQLAlchemy connection pool size. |
| `DATABASE_MAX_OVERFLOW`        | `10`            | Connections allowed above the pool size. |
| `DATABASE_STATEMENT_TIMEOUT_SECONDS` | `30`      | PostgreSQL cancels an API query that overruns this, so one slow statement cannot exhaust the pool. The background jobs opt out. `0` disables it. |
| `CELERY_BROKER_URL`            | *(from Redis)*  | Overrides the broker; defaults to the Redis above, database +1. |
| `CELERY_RESULT_BACKEND`        | *(from Redis)*  | Overrides the result backend; defaults to Redis database +2. |
| `TASK_TIME_LIMIT_SECONDS`      | `1800`          | Hard limit on one job before the worker kills it. |
| `STATISTICS_LOOKBACK_DAYS`     | `7`             | How far back the statistics job re-checks matches. |
| `FAILURE_REPORT_HOURS`         | `24`            | Window the daily failure report covers. |
| `VITE_API_BASE_URL`            | `/api`          | Frontend build-time API base path. |
| `VITE_API_PROXY_TARGET`        | `http://127.0.0.1:8000` | Vite dev-server proxy target (local dev only). |

`APP_NAME` and `APP_VERSION` are the service's identity rather than configuration;
they come from the package and are not meant to be overridden.

The backend refuses to start if a required variable is missing or invalid, and
names the offending field in the error.

## Running tests and linting

Run everything (linters, type checks, unit tests, frontend build):

```bash
python scripts/check.py
python scripts/check.py --integration   # also hits real PostgreSQL + Redis
python scripts/check.py --audit         # also audits dependencies (needs network)
```

Or individually:

```bash
# Backend (from backend/, virtualenv active)
ruff format --check .
ruff check .
mypy
pytest                                    # unit tests; integration tests are skipped
RUN_INTEGRATION_TESTS=1 pytest            # + integration and database tests; needs
                                          # `docker compose up -d postgres redis`

# Frontend (from frontend/)
npm run format:check
npm run lint
npm run typecheck
npm test

# Whole running stack (from the repo root)
python tests/smoke/check_stack.py
```

## Troubleshooting

- **Health checks take ~2 s each when running the backend on Windows/macOS hosts.**
  `localhost` resolves to IPv6 `::1` first, but Compose publishes ports on IPv4
  `127.0.0.1` only. Use `POSTGRES_HOST=127.0.0.1` and `REDIS_HOST=127.0.0.1` (the
  values in `.env.example`).
- **`port is already allocated`.** Change the matching `*_HOST_PORT` in `.env`.
- **`set POSTGRES_PASSWORD in .env`.** You have no `.env` yet. Copy `.env.example`.
- **The dashboard shows "Degraded".** One dependency is unreachable. The row shows
  which one; `docker compose ps` and `docker compose logs <service>` show why.
