# Deployment

How to run this in production, what each option costs, and what to do when it goes
wrong. Hardening details (rate limits, headers, secrets, container settings) are in
[docs/operations.md](docs/operations.md).

> Nothing here has been deployed. These are the configurations and the procedure;
> running them is your decision.

## What you are deploying

Six containers, from two images:

| Service | Image | Role |
| --- | --- | --- |
| `frontend` | `football-prediction-frontend` | nginx: serves the built SPA and proxies `/api` to the backend. The only service that publishes a port. |
| `backend` | `football-prediction-backend` | FastAPI. Applies migrations on start. |
| `worker` | same | Celery worker: the scheduled jobs actually run here. |
| `beat` | same | Celery scheduler. **Exactly one.** |
| `postgres` | `postgres:17-alpine` | The database. The only stateful thing that matters. |
| `redis` | `redis:7.4-alpine` | Broker, rate-limit counters, status cache. |

The browser only ever talks to the frontend's origin, so there is no CORS
configuration to get wrong.

## Before the first deploy

1. **A host.** 4 GB RAM is comfortable for the stack as configured; the worker is
   the hungry one when it retrains.
2. **A domain and a TLS certificate.** The stack speaks plain HTTP on purpose —
   terminate TLS in front of it (below).
3. **A data provider key**, if you want the scheduled jobs to fetch anything
   ([DATA.md](DATA.md)).
4. **A backup destination** for PostgreSQL. A prediction platform with no history
   is a blank page.

## Deploying with Docker Compose

The straightforward option, and the one this repository configures.

```bash
git clone <your fork> && cd prediction-platform
cp .env.production.example .env.production
# Fill in POSTGRES_PASSWORD, REDIS_PASSWORD and the provider key.
# Generate passwords: python -c "import secrets; print(secrets.token_urlsafe(32))"

docker compose -f docker-compose.prod.yml --env-file .env.production build
docker compose -f docker-compose.prod.yml --env-file .env.production up -d
docker compose -f docker-compose.prod.yml --env-file .env.production ps
```

`docker-compose.prod.yml` is **standalone** — use it instead of
`docker-compose.yml`, not on top of it. The development file publishes PostgreSQL
and Redis on localhost and runs in development mode; the production file does
neither, pins images by tag, sets `ENVIRONMENT=production` (which withholds `/docs`
and `/openapi.json`), runs read-only root filesystems with every capability
dropped, caps container logs, and limits memory and CPU per service.

Then check it:

```bash
curl -fsS http://localhost/api/health          # the proxy and the API
python tests/smoke/check_stack.py              # SPA, fallback and proxy
```

### First data and first model

A fresh deployment has an empty database, and says so honestly on every page —
that is the intended behaviour, not a failure.

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.production"
$C exec backend python -m app.ingestion competitions --country England
$C exec backend python -m app.ingestion sync --competition 39
$C exec backend python -m app.ingestion backfill --competition 39
$C exec backend python -m app.features generate
```

Training needs the `ml` extra, which the serving image does not carry. Train
somewhere that has it (a workstation or a build machine), then make the artifact
available to the backend and activate it:

```bash
python -m app.ml train --cv-folds 3 --register       # writes an artifact + registers it
python -m app.ml registry activate elo <version> --reason "first deploy"
```

The API reads artifacts from the `artifact_uri` recorded in the registry, so that
path must exist **inside the backend container**. Mount a shared models volume, or
copy the artifact in and register it with a path the container can see.

Finally, set `JOB_COMPETITIONS` to the ids you imported so the jobs keep them
current, and restart the worker and beat.

## Putting TLS in front

The stack serves HTTP on `PUBLIC_PORT`. Terminate TLS with whatever you already
run. With the terminator on the same host, bind the stack to localhost first
(`PUBLIC_BIND=127.0.0.1`).

**Caddy** — the shortest path to a certificate:

```caddyfile
predictions.example.com {
    reverse_proxy 127.0.0.1:80
}
```

**nginx** — terminate TLS and pass everything through:

```nginx
server {
    listen 443 ssl http2;
    server_name predictions.example.com;
    ssl_certificate     /etc/letsencrypt/live/predictions.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/predictions.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Add HSTS at the terminator, since it is the only part that knows TLS is in use.
The container nginx already sends the other security headers and a content-security
policy.

## Other deployment options

| Option | When it fits | What it costs |
| --- | --- | --- |
| **Single host, Docker Compose** (configured here) | One machine, a few competitions, one operator | No redundancy: the host is the single point of failure. Restarts are brief outages. |
| **Managed PostgreSQL + Compose for the rest** | You want backups, failover and point-in-time recovery without running them | Point `POSTGRES_HOST/PORT/USER/PASSWORD/DB` at the managed instance and drop the `postgres` service. Costs money; removes the scariest operational job. |
| **Kubernetes** | You already run it | One Deployment each for backend, worker and frontend; beat must stay at **one** replica; Redis and PostgreSQL as managed services or operators. The images and environment variables are the same — nothing in the code assumes Compose. |
| **PaaS (Render, Fly.io, Railway…)** | Smallest operational surface | Backend and frontend as services, worker and beat as background workers, managed PostgreSQL and Redis. Check the platform allows a long-running scheduler before relying on beat. |
| **Serving only, jobs elsewhere** | Predictions must stay up, freshness can lag | Run `backend` + `frontend` and set `JOBS_ENABLED=false`; run ingestion and training on a schedule you control. |

Whatever you choose: **exactly one `beat`**, and the backend needs to read the
model artifacts the registry points at.

## Upgrades and rollback

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.production"
$C build                      # or pull, if you publish images
IMAGE_TAG=2026-09-24 $C up -d # new tag; migrations run as the backend starts
```

Roll back by deploying the previous `IMAGE_TAG`. Migrations are not automatically
reversed — every migration in this repository has a `downgrade()`, but a rollback
across a schema change is a deliberate act:

```bash
$C exec backend alembic downgrade -1
```

Roll back a *model* instead of the stack when a model is the problem:

```bash
$C exec backend python -m app.ml registry rollback elo
```

## Backups

The database is the only thing you cannot rebuild. Predictions are immutable
history; features and models can be regenerated, but slowly.

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.production"
$C exec -T postgres pg_dump -U "$POSTGRES_USER" -Fc "$POSTGRES_DB" \
  > "backups/$(date +%F).dump"
```

`BACKUP_DIR` is mounted into the `postgres` container at `/backups` for exactly
this. Copy the dumps off the host — a backup on the same disk is not a backup —
and **restore one into a scratch database before you need it**:

```bash
$C exec -T postgres pg_restore -U "$POSTGRES_USER" -d restore_test --clean < backups/2026-09-24.dump
```

## Operating it

```bash
C="docker compose -f docker-compose.prod.yml --env-file .env.production"
$C ps                                        # health of every service
$C logs -f backend                           # one access line per request, with ids
$C exec backend python -m app.tasks status   # what the scheduled jobs did
$C exec backend python -m app.tasks schedule # what they are configured to do
$C exec backend python -m app.ingestion report   # data coverage and open issues
```

The **Data status** page in the application shows the same figures without a shell.
A request that failed is traceable: the response carries `X-Request-ID`, and so
does every log line for it.

## When something is wrong

- **Every page is empty.** No data has been imported yet; the pages say so. Follow
  "First data and first model".
- **Predictions return 409 "no model is active".** Register and activate one.
- **Predictions return 503.** The artifact is missing inside the container, or it
  is an XGBoost/LightGBM model the serving image cannot load
  ([API.md](API.md)).
- **429 for ordinary browsing.** Raise `RATE_LIMIT_REQUESTS`; the defaults assume
  one person per address.
- **The dashboard says "Degraded".** A dependency is unreachable; the row names it.
- **Jobs are not running.** Check `JOBS_ENABLED`, that `beat` is up, and that
  `JOB_COMPETITIONS` is not empty — an empty list makes the fetch jobs skip by
  design.
- **`beat` is running twice.** Every job doubles. Scale it back to one.

## What is not configured here

- **No TLS inside the stack** — deliberately; terminate it in front.
- **No metrics or tracing backend.** Logs go to stdout with request ids; point your
  collector at the containers.
- **No secret manager.** Secrets come from the environment file. If you have Docker
  Swarm secrets, Kubernetes secrets or a vault, use them: every secret is read from
  an environment variable.
- **No CI/CD pipeline.** `python scripts/check.py --integration --audit` is the
  gate; wiring it into CI is one step ([CONTRIBUTING.md](CONTRIBUTING.md)).
- **No autoscaling or multi-region.** One host, one scheduler.
