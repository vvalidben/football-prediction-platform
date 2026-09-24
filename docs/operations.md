# Operations and hardening

What the platform does to stay up and stay safe, what was reviewed in Sprint 16,
and what is deliberately not done. Code: `backend/app/api/middleware.py`,
`backend/app/api/rate_limit.py`, `backend/app/cache/responses.py`,
`docker/frontend/nginx.conf.template`, `docker-compose.yml`.

## Requests: identity, logging, headers

Every request gets an id — the caller's `X-Request-ID` if it sent a plausible one,
otherwise a fresh one — and is logged once with method, path, status and duration.
The same id is returned in the response header, and in the body of a 500:

```json
{ "detail": "Internal server error", "request_id": "c2a22784e26548d79b2b72e875f5dcf1" }
```

That is the whole error contract: a generic message, plus the one value that ties a
user's report to the log line. Nothing about the failure reaches the client.

A forged id cannot poison the logs: only printable ASCII is accepted, and it is cut
to 64 characters. Behind the bundled nginx the id is set by the proxy
(`$request_id`), so a client cannot choose it at all.

Query strings, bodies and headers are never logged — they carry provider keys, and
in other deployments would carry personal data.

Every API response carries `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` and, unless a route says
otherwise, `Cache-Control: no-store`.

## Rate limiting

Two fixed-window buckets per client address, counted in Redis:

| Bucket | Default | What it covers |
| --- | --- | --- |
| `read` | 300 / minute | Everything except predictions |
| `predict` | 30 / minute | `POST /predictions/*` — builds features and runs a model |

Over the limit is `429` with `Retry-After` and a message saying what the limit is;
`X-RateLimit-Limit` and `X-RateLimit-Remaining` are on every response.
`/health` is exempt, because the container's own health check uses it.

**The limiter fails open.** If Redis is unavailable, requests are allowed and a
warning is logged: a rate limiter that is down must not take the API down with it.

nginx limits the proxied API as well (10 r/s per address, burst 20), so a flood is
shed before it reaches Python.

Settings: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`,
`RATE_LIMIT_PREDICT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`.

## Authentication: why there is none

The deployment is a public, read-only view of football predictions. There are no
user accounts, no personal data and no destructive operation to protect: the only
state a caller can create is a stored prediction, which is exactly what the site is
for. Adding a login would not protect anything that exists, and putting an API key
in the browser bundle would protect nothing at all — a `VITE_` variable is compiled
into the JavaScript and served to everyone (a test enforces that only
`VITE_API_BASE_URL` is read).

What the expensive path actually needed was a bound on how much one caller can
consume, which is the rate limiter above. If this ever serves private data or paid
access, authentication belongs at the edge (a proxy or gateway) so the browser app
stays free of credentials.

## Caching

Only `/data/status` and `/data/freshness` are cached, for `STATUS_CACHE_SECONDS`
(default 30) in Redis, with a matching `Cache-Control: public, max-age=…`. They
aggregate across several tables and describe imports that run every few minutes.

Predictions, history, explanations and model metrics are never cached: they are
records, and a caller asking for them should see what is stored now. **The cache
fails open** — an unreachable Redis means the endpoint computes its answer as
before.

## Query cost

Listing endpoints load what they serialise in one query. Before the Sprint 19
audit, `/matches`, `/matches/upcoming`, `/matches/{id}` and `/predictions` read the
competition, season and both teams **per row**: 32 statements for 12 matches
against 12 for 2. They now join those relationships, and
`backend/tests/api/test_query_efficiency.py` fails if the count starts growing with
the page size again.

API connections also carry a `statement_timeout` (30 s by default) and an
`idle_in_transaction_session_timeout`, so one slow query cannot hold a connection
until the pool is exhausted and the whole API stalls. The background jobs create
their engine with the timeout off: a backfill or a retrain runs long queries on
purpose.

## Database indexes

Sprint 16 added two, after measuring on 60,000 predictions in PostgreSQL 17:

| Index | Query | Before | After |
| --- | --- | --- | --- |
| `ix_predictions_predicted_at (predicted_at DESC, id DESC)` | `GET /predictions`, the dashboard's "latest predictions" | Seq scan + top-N sort, 858 buffers, 7.1 ms | Index-only scan, 4 buffers, 0.50 ms |
| `ix_prediction_results_evaluated_at` | "what has been scored since…" (reporting) | Seq scan | Index scan |

Every foreign key is indexed, and a test proves it
(`tests/db/test_migrations.py`). Another test asserts both new indexes exist and
that the planner chooses the prediction one for the dashboard's ordering.

## Docker

- Images run as a non-root user (backend) and drop every capability. nginx keeps
  only what it needs to bind :80 and hand its workers to the `nginx` user.
- Root filesystems are read-only; the few writable paths are tmpfs
  (`/tmp`, nginx's cache, run and rendered-config directories).
- `no-new-privileges` on every service.
- Container logs are capped (10 MB × 3), so a failure loop cannot fill the disk.
- Memory and CPU limits are set per service as a starting point — raise them
  before blaming the application for an OOM kill.
- Host ports bind to `127.0.0.1` only: nothing is exposed to the network by
  default.
- The backend image installs the `serve` extra only: no training libraries, no
  MLflow, no `shap`.

## Secrets

- `.env` is git-ignored and is the only file holding real credentials;
  `.env.example` ships placeholders. A test asserts both.
- Secrets are `SecretStr` in settings, so they do not appear in logs, `repr()` or
  error output. A test asserts a password cannot be printed.
- The provider key travels in a request header, never in a URL, and no log line
  includes headers.
- `backend/tests/test_no_secrets.py` scans everything that would be committed for
  private keys, AWS keys, connection strings with passwords and filled-in provider
  keys, and checks that the browser bundle reads no secret and sends no
  credentials. It runs with every other test.

## Production mode

With `ENVIRONMENT=production`:

- `/docs` **and** `/openapi.json` are both withheld — before Sprint 16 the schema
  stayed public after the docs were disabled, which is the same information;
- SQL echo is off (it is never on), uvicorn does not reload, and the 500 handler
  returns nothing but the generic message and the request id.

## Dependency review

`python scripts/check.py --audit` runs `pip-audit` over the backend environment and
`npm audit` over the frontend, so the review is repeatable rather than a one-time
claim. It is opt-in because both reach the network: an offline run of the suite
should not fail for a reason unrelated to the code. Both report **no known
vulnerabilities** (September 2026). The frontend ships no runtime
dependency beyond React and the router; the API image installs a deliberately small
subset of the Python dependencies.

## Known limitations

- `uvicorn` runs with `--forwarded-allow-ips "*"`, so it trusts
  `X-Forwarded-For` from any peer. In this stack only nginx can reach it (the
  backend port binds to localhost), but a deployment that exposes the backend
  directly should narrow this, or a client could spoof the address the rate
  limiter counts against.
- Rate-limit windows are fixed, not sliding: a caller can use a full allowance at
  the end of one window and again at the start of the next.
- Migrations run from the container entrypoint. Alembic locks its version table,
  so parallel replicas are safe, but a long migration will delay every replica's
  start.
- There is no error-reporting service; the browser boundary logs to the console
  and the backend logs to stdout. Point the container logs at whatever collector a
  deployment already has.
- Resource limits and rate limits are starting points chosen without production
  traffic to measure against.
