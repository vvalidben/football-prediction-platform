# Architecture

This document describes the platform as of **Sprint 18 (documentation and
deployment)**. It covers what exists today and where upcoming features are expected
to fit. It does not describe features that have not been built.

The guides at the root are the entry points — [DATA.md](DATA.md), [ML.md](ML.md),
[API.md](API.md), [DEPLOYMENT.md](DEPLOYMENT.md), [CONTRIBUTING.md](CONTRIBUTING.md)
— and the pages under `docs/` are the detail. This document is the shape of the
whole.

## System overview

```
                 ┌───────────────────────────── Docker network ─────────────────────────────┐
                 │                                                                          │
 Browser ──HTTP──┼──► frontend (nginx :80)                                                  │
 127.0.0.1:8080  │      ├── /            → static React build (SPA fallback to index.html)  │
                 │      └── /api/*       → proxy ──► backend (uvicorn/FastAPI :8000)         │
                 │                                    ├──► postgres :5432  (system of record)│
                 │                                    └──► redis :6379     (cache / queues)  │
                 └──────────────────────────────────────────────────────────────────────────┘
```

| Service    | Image / build                     | Responsibility | Persistence |
| ---------- | --------------------------------- | -------------- | ----------- |
| `postgres` | `postgres:17-alpine`              | Primary relational store: competitions, teams, matches, provider data, features, predictions and evaluation results (see docs/database.md). | `postgres-data` volume |
| `redis`    | `redis:7.4-alpine`, password-protected, AOF enabled | Cache now. Later a broker for background jobs such as data ingestion and model training. | `redis-data` volume |
| `backend`  | `docker/backend/Dockerfile`       | HTTP API, business logic, database migrations on start. | stateless |
| `frontend` | `docker/frontend/Dockerfile`      | Serves the compiled SPA and reverse-proxies `/api` to the backend. | stateless |

Startup order is enforced with health checks: `postgres` and `redis` must be
healthy before `backend` starts, and `backend` must be healthy before `frontend`
starts.

### Why the frontend proxies the API

The browser only talks to one origin (the frontend). nginx in Docker, and the Vite
dev server locally, forward `/api/<path>` to the backend's `/<path>`. As a result:

- no CORS configuration is needed in the default setup (`CORS_ALLOWED_ORIGINS` exists
  for deployments where the API is called cross-origin);
- the frontend bundle has no environment-specific backend URL baked in;
- the backend keeps clean root paths (`/health`, later `/matches`, …).

nginx resolves `backend` through Docker DNS at request time
(`resolver 127.0.0.11` plus a variable `proxy_pass`). The proxy therefore keeps
working when the backend container is recreated with a new IP.

## Backend

```
backend/
├── app/
│   ├── main.py            Application factory: settings → logging → lifespan resources → routes
│   ├── core/
│   │   ├── config.py      Settings (pydantic-settings); single source of configuration
│   │   └── logging.py     Process-wide logging config (app + uvicorn → stdout)
│   ├── api/
│   │   ├── dependencies.py  FastAPI dependencies: settings, engine, Redis, DB session
│   │   └── routes/        One module per resource (health.py today)
│   ├── models/            SQLAlchemy ORM models, one module per domain area (docs/database.md)
│   ├── schemas/           Pydantic request/response models (the API contract)
│   ├── providers/         Provider-independent data interface + adapters (docs/data-providers.md)
│   │   ├── base.py        FootballDataProvider contract; schemas.py neutral records; errors.py
│   │   ├── http.py        JSON HTTP client: timeouts, retries with backoff, rate limiting
│   │   ├── api_football/  API-Football v3 adapter (payload models + normalisation)
│   │   └── mock.py        In-memory provider for tests
│   ├── ingestion/         Idempotent sync, normalisation, historical backfill, run log; CLI
│   ├── quality/           Validation rules, issue log, database audit, quality report
│   ├── features/          Leak-free features: timeline, ratings, builder, snapshots (docs/features.md)
│   ├── ml/                Models, ensembles, time-aware CV, metrics, MLflow, reports, score engine
│   ├── registry/          Model versions: active model, history, rollback, MLflow mirror (docs/model-registry.md)
│   ├── tasks/             Celery jobs, schedule and job status (docs/scheduled-jobs.md)
│   ├── services/          Business logic, free of HTTP concerns (health checks today)
│   ├── db/
│   │   ├── base.py        Declarative Base with constraint naming conventions
│   │   └── session.py     Engine and session-factory construction
│   └── cache/redis.py     Redis client construction
├── alembic/               Migrations; env.py takes the URL from Settings
└── tests/                 pytest: unit tests + opt-in integration tests
```

**Layering.** Routes stay thin: they parse input, call a service, and shape the
response. Services hold the logic and receive their collaborators (engine, Redis
client) as arguments, so they are testable without HTTP. Schemas define the public
contract separately from the ORM models in `app/models/`.

**Application factory and lifespan.** `create_app(settings)` builds the app. The
engine, session factory and Redis client are created in the lifespan handler,
stored on `app.state`, and disposed on shutdown. Nothing connects at import time.
Uvicorn runs with `--factory`. Tests call `create_app()` with explicit settings and
swap resources through `app.dependency_overrides`.

**Sync SQLAlchemy.** Endpoints that touch the database are plain `def` functions,
which FastAPI runs in its threadpool. This keeps the ORM, Alembic and future
data-science code (pandas, scikit-learn) on one simple execution model. Moving hot
paths to `AsyncEngine` stays possible later without changing the API contract.

**Configuration.** `Settings` reads environment variables, falling back to the
repository-root `.env` for local runs. Required values (`POSTGRES_USER`,
`POSTGRES_PASSWORD`, `POSTGRES_DB`) have no defaults, so a misconfigured deployment
fails at startup rather than at first use. Passwords are `SecretStr` and never
appear in logs or reprs. The database URL is built with `URL.create`, so special
characters in credentials need no manual escaping.

**Error handling.** A global exception handler logs the full traceback and returns
`{"detail": "Internal server error"}` with status 500. Internal details never reach
clients. Validation errors keep FastAPI's standard 422 format.

**Logging.** Plain-text logs go to stdout, which is where Docker collects them.
Uvicorn's loggers propagate to the same root handler, so there is a single format.
`LOG_LEVEL` controls verbosity.

### Health endpoint

`GET /health` reports the API's own status plus connectivity to each dependency:

```json
{
  "status": "ok",
  "service": "football-prediction-api",
  "version": "0.1.0",
  "environment": "development",
  "checks": {
    "database": { "status": "ok", "latency_ms": 0.9, "error": null },
    "redis":    { "status": "ok", "latency_ms": 0.4, "error": null }
  }
}
```

- **200** when every check passes. **503** with the same body and
  `"status": "degraded"` when any check fails. Load balancers and the Docker
  healthcheck can therefore rely on the status code alone.
- Checks run **in parallel under one deadline** (`HEALTH_CHECK_TIMEOUT_SECONDS`,
  default 3 s). A check still running at the deadline is reported as
  `"error": "Timeout"`. Without this, a stopped container can stall the endpoint:
  Docker DNS takes about 4 s to fail each lookup, and client retries multiply that.
- `error` contains only the exception class name (e.g. `OperationalError`), never
  the message, because messages can include hostnames or connection details. The
  full message is logged server-side.
- The Redis client uses a **bounded retry policy** (2 retries, short jittered
  backoff) instead of redis-py's default of 10.

### Database and migrations

- The schema (27 tables: competitions, seasons, teams, players, matches, provider ID
  mappings, events, lineups, statistics, injuries, odds, standings, ingestion log,
  feature snapshots, model versions, predictions and their evaluation) is
  described in **[docs/database.md](docs/database.md)**, including the entity
  relationship diagram and design decisions.
- Migrations live in `backend/alembic/versions/`; the initial one creates the whole
  schema. `Base.metadata` sets a naming convention, so every constraint has a
  predictable name (e.g. `uq_team_external_ids_data_source_id_external_id`).
- `alembic/env.py` accepts an open connection through `config.attributes`, which
  the test suite uses to migrate its own throwaway database.
- The backend container applies `alembic upgrade head` before starting uvicorn
  (`docker/backend/entrypoint.sh`). Once there are several backend replicas,
  migrations should move to a separate one-off job; this is noted as a known
  limitation.

## Frontend

```
frontend/src/
├── main.tsx                 Entry point (BrowserRouter)
├── App.tsx                  Routes
├── config.ts                Runtime config from VITE_* variables
├── format.ts                Dates, percentages and scorelines, formatted in one place
├── api/client.ts            fetch wrapper: ApiError, FastAPI detail, query strings
├── api/endpoints.ts         One function per endpoint; api/types.ts the response shapes
├── api/health.ts            Typed health client + runtime response validation
├── hooks/useApi.ts          AsyncState<T>, abort on change/unmount, optional refresh interval
├── hooks/useHealth.ts       The health poll: initial load, polling, manual refresh, cancellation
├── pages/                   Dashboard, MatchPrediction, History, Team, Teams, ModelPerformance,
│                            DataStatus
├── components/              Layout, ErrorBoundary, Async, states (loading/error/empty/card),
│                            Stat, MatchList,
│                            Explanation (why this prediction),
│                            HealthPanel, StatusBadge, charts/Charts.tsx
└── test/                    Vitest setup, the fetch mock (render.tsx) and API fixtures
```

See [docs/frontend.md](docs/frontend.md) for what each page shows.

- **Typed API boundary.** `fetchHealth` checks the response shape at runtime
  (`isHealthResponse`) before handing it to components, so a contract mismatch
  shows up as a clear error state instead of a crash in rendering code. It resolves
  on both 200 and 503, since both carry a valid report, and rejects with `ApiError`
  for network failures, unexpected statuses and malformed bodies.
- **Data fetching.** `useApi` gives every page one `AsyncState<T>`
  (`loading | success | error`), aborts the request in flight when its dependencies
  change or the component unmounts, and takes an optional refresh interval (30 s on
  the dashboard and data status). A data-fetching library (e.g. TanStack Query) is
  still deferred: caching across pages is the first thing that would justify it.
- **Honest emptiness.** Pages render only what the API returned. Where there is no
  data — no predictions yet, a model never evaluated, xG no provider supplied — the
  page says so instead of showing a placeholder number, and a failing endpoint shows
  the backend's own message with a retry rather than blanking the page.
- **Charts.** Hand-drawn SVG (`components/charts/Charts.tsx`): bars, columns, lines
  and a reliability diagram, each with an `aria-label`. No charting dependency, so
  the bundle stays small and the charts are testable with Testing Library.
- **Styling.** Tailwind CSS v4 through its Vite plugin. Theme tokens (the `pitch`
  palette) live in `src/index.css`.
- **Production build.** A multi-stage image runs `npm ci && npm run build` on Node 24
  and serves `dist/` from `nginx:1.29-alpine`, with long-lived caching for
  fingerprinted `/assets/`, `no-cache` for `index.html`, and basic security headers.

## Docker images

| Image    | Base                     | Notes |
| -------- | ------------------------ | ----- |
| backend  | `python:3.12-slim`, multi-stage | Dependencies installed into `/opt/venv` in a builder stage. Runs as a non-root `app` user. Has a `HEALTHCHECK` on `/health`. |
| frontend | `node:24-alpine` → `nginx:1.29-alpine` | The nginx config is an envsubst template (`BACKEND_UPSTREAM`, `NGINX_RESOLVER`). |

Both images build with the **repository root as context**, so all Docker assets live
under `docker/`. Each Dockerfile has its own allow-list `.dockerignore`
(`Dockerfile.dockerignore`), so only the files an image needs are sent to the
builder.

## Testing strategy

| Level        | Where | What | Runs |
| ------------ | ----- | ---- | ---- |
| Backend unit | `backend/tests/` | Config parsing and validation, health semantics (ok, degraded, timeout, no detail leakage), generic 500 handler, Redis fail-fast. Dependencies are replaced with test doubles through `dependency_overrides`. | always |
| Backend integration | `backend/tests/test_integration.py` (`@pytest.mark.integration`) | Real PostgreSQL and Redis round trips and `/health` against live services. | `RUN_INTEGRATION_TESTS=1` |
| Database | `backend/tests/db/` (`@pytest.mark.integration`) | Migration applies, round-trips and matches the models; every FK indexed; entity creation and relationships; duplicate provider IDs; natural keys and CHECK constraints; prediction storage. Runs against a throwaway `<db>_test` database, one rolled-back transaction per test. | `RUN_INTEGRATION_TESTS=1` |
| Providers | `backend/tests/providers/` | HTTP retries, backoff, Retry-After, rate limiter; API-Football normalisation and error mapping against recorded-style fixtures; mock provider; one contract suite run against every provider. | always |
| Ingestion | `backend/tests/ingestion/` (`@pytest.mark.integration`) | Sync into PostgreSQL: first sync, idempotent re-sync, updates, missing data, partial and total failures, retries end to end, run log, cross-provider linking, concurrent syncs. | `RUN_INTEGRATION_TESTS=1` |
| Historical pipeline and quality | `backend/tests/ingestion/`, `backend/tests/quality/` | Normalisation and validation rules (unit); backfill across seasons with statistics, chronology preserved, quota-saving re-runs, coverage-driven skipping, quota stop; issues detected, recorded, de-duplicated and resolved, never deleted; audit checks; report figures and CLI commands (PostgreSQL). | unit: always; DB: `RUN_INTEGRATION_TESTS=1` |
| Features | `backend/tests/features/` | **No-leakage proofs** (future matches, in-progress matches, future and post-match statistics, the target's own result, odds and absences after the cutoff, whole-dataset canary, batch vs single agreement), mutation-checked; timeline guards and ratings (unit); hand-calculated feature values; snapshot immutability, drift, reproducibility and versioning; CLI. | unit: always; DB: `RUN_INTEGRATION_TESTS=1` |
| Models | `backend/tests/ml/` | Score-grid maths and normalisation; hand-computed metrics; chronological splits (ordering, no overlap, no shuffling); training and prediction of each model on synthetic data with a known signal; selection that cannot see the test set (a poisoned test set changes nothing); persistence round trip; MLflow runs and artifacts; end-to-end snapshots → train → register on PostgreSQL. | unit: always; DB: `RUN_INTEGRATION_TESTS=1` |
| Critical guarantees | `backend/tests/ml/test_critical_guarantees.py`, `backend/tests/api/test_prediction_guarantees.py` | The eight properties in docs/testing.md, stated in one place: normalised distributions for every model, reproducibility across a save and reload, missing data, unknown teams end to end (both start from the default rating and two unknown fixtures get the same answer), artifact loading, dataset chronology, and that recording a 7–0 result does not move an earlier prediction. | unit: always; end to end: `RUN_INTEGRATION_TESTS=1` |
| Jobs CLI | `backend/tests/tasks/test_cli.py` | `run` (in process and queued), `status` (window, task filter, cap, empty history) and `schedule`; every scheduled task is runnable from the command line. | `RUN_INTEGRATION_TESTS=1` |
| Hardening | `backend/tests/api/test_hardening.py`, `backend/tests/test_no_secrets.py` | Request ids (echoed, sanitised, truncated); one access log line per request; a 500 leaks nothing but its id; security headers; rate limits per bucket, with `Retry-After`, exempting `/health` and failing open when Redis is down; the response cache; docs and schema withheld in production; a repository-wide secret scan. | always |
| Prediction history | `backend/tests/api/test_history.py` | Rows pair prediction with result; goal errors equal expected minus actual; each filter and filters combined; the summary follows the filters and reports null rather than invented averages; pending becomes evaluated when the result arrives. | `RUN_INTEGRATION_TESTS=1` |
| Explainability | `backend/tests/ml/test_explain.py`, `backend/tests/api/test_explanations.py` | Exact additivity per model and target; the explained output equals the model's own; contributions match the column sklearn names (mutation-checked); missing-value flags; refusals for the baseline, ensembles and absent `shap`; the endpoints end to end. | unit: always; API: `RUN_INTEGRATION_TESTS=1` |
| Frontend unit/component | `frontend/src/**/*.test.ts(x)` | API client error handling and response validation; HealthPanel healthy, degraded, offline and recovery states; each page's loaded, empty and failed states, xG shown only when supplied, and the URLs the controls request. | always |
| Stack smoke | `tests/smoke/check_stack.py` | Backend health, SPA serving and fallback, and the nginx → backend proxy against the running Compose stack. | after `docker compose up` |

`scripts/check.py` runs every static check and test suite in one command;
`--audit` adds `pip-audit` and `npm audit` (kept opt-in because both need the
network).

## Where upcoming features fit

These are placement guidelines, not commitments:

- **Football data (competitions, teams, matches):** the tables exist (Sprint 1);
  next come Pydantic schemas in `app/schemas/` and read endpoints in
  `app/api/routes/`.
- **Data ingestion:** competitions, seasons, teams, matches and team statistics,
  with historical backfill and data-quality tracking (Sprints 2 and 3,
  [docs/historical-pipeline.md](docs/historical-pipeline.md)). Next: player
  statistics, lineups, injuries and odds, then automatic updates as scheduled
  background jobs (Redis as broker, a worker service built from the backend image).
- **Features:** implemented (Sprint 4, [docs/features.md](docs/features.md)); models
  read `feature_snapshots`, never the provider or raw tables.
- **Baseline models:** implemented (Sprint 5, [docs/models.md](docs/models.md)).
  Every model outputs an exact-score grid; 1X2 and expected goals derive from it.
- **Goal prediction engine:** implemented (Sprint 7,
  [docs/prediction-engine.md](docs/prediction-engine.md)). `ScorePrediction` holds
  the 11 × 11 score matrix and everything derived from it: expected goals, most
  probable score, 1X2, over/under, BTTS. Next: storing predictions and serving them
  through the API.
- **Ensembles:** implemented (Sprint 8, [docs/ensembles.md](docs/ensembles.md)).
  `EnsembleModel` combines members' score grids with weights fitted on out-of-fold
  validation predictions; it is tracked and registered as its own model version.
- **Backtesting and calibration:** implemented (Sprint 9,
  [docs/backtesting.md](docs/backtesting.md)). Walk-forward predictions are written to
  the `predictions` tables in real-world order; Platt and isotonic calibration are
  fitted only on earlier out-of-sample predictions.
- **Model registry:** implemented (Sprint 10,
  [docs/model-registry.md](docs/model-registry.md)). `app/registry/` has no ML
  dependencies, so the API can read model metadata; the schemas it will return are in
  `app/schemas/model_registry.py`.
- **Prediction API:** implemented (Sprint 11, [docs/api.md](docs/api.md)). Routers in
  `app/api/routes/` over services in `app/services/`; predictions are computed from
  on-demand features with the active model version and stored in the `predictions`
  tables. The image installs the `serve` extra (scikit-learn) so it can load models.
- **Machine-learning models:** implemented (Sprint 6): random forest, XGBoost and
  LightGBM behind the same `Predictor` interface, tuned with rolling-origin
  cross-validation and compared in [docs/model-comparison.md](docs/model-comparison.md).
  Models train offline (`python -m app.ml train`); versions are never overwritten
  and are registered inactive. Next: storing predictions and a deliberate
  activation step. The API will serve stored predictions rather than training in
  the request path.
- **Automatic updates:** implemented (Sprint 12,
  [docs/scheduled-jobs.md](docs/scheduled-jobs.md)). A Celery worker and beat run from
  the backend image over Redis; every schedule is configurable, each run is recorded in
  `job_runs`, and `/jobs` exposes the status.
- **Evaluation:** stored predictions compared with actual results, exposed through
  the API and displayed in the frontend.

## Known limitations

- No CI pipeline yet. `python scripts/check.py --integration --audit` is the
  local equivalent and the intended gate.
- Migrations run inside the backend container at startup. This is fine for one
  replica; use a dedicated migration job before scaling out.
- Logs are plain text, with a request id on every line. Switch to structured JSON
  when a log aggregator is chosen.
- Deployment is single-host: `docker-compose.prod.yml` sets resource limits, log
  rotation, read-only containers and dropped capabilities, and
  [DEPLOYMENT.md](DEPLOYMENT.md) covers upgrades, rollback and backups — but TLS
  is terminated outside the stack, there is no secrets manager, and `beat` must
  stay at one replica.
- No authentication, deliberately: the API is a public read-only view whose only
  write is storing a prediction it just made. Rate limiting bounds that instead;
  the reasoning is in [docs/operations.md](docs/operations.md).
- Database: see "Known limitations" in [docs/database.md](docs/database.md).
- Features: see "Known limitations" in [docs/features.md](docs/features.md).
- Models: see "Known limitations" in [docs/models.md](docs/models.md).
- Ingestion: see "Known limitations" in [docs/data-providers.md](docs/data-providers.md)
  and [docs/historical-pipeline.md](docs/historical-pipeline.md).
