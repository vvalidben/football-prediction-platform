# Contributing

What this project expects from a change, and how to make one without guessing.

## Setting up

```bash
git clone <repo> && cd prediction-platform
cp .env.example .env            # replace both `change-me` passwords

docker compose up -d postgres redis

cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -e ".[serve,ml,dev]"

cd ../frontend
npm ci
```

Run the backend and frontend against those services:

```bash
cd backend && alembic upgrade head && uvicorn app.main:create_app --factory --reload
cd frontend && npm run dev            # http://localhost:5173, proxying /api
```

## Before you push

```bash
python scripts/check.py --integration        # everything, against real services
python scripts/check.py --audit              # adds pip-audit and npm audit
```

That runs ruff (format and lint), mypy in strict mode, the backend suite, prettier,
eslint, tsc and the frontend suite and build. **It has to pass.** If a check is
wrong, fix the check deliberately and say why — do not work around it.

## What a change is expected to include

- **Tests that would fail without it.** For a bug fix, write the failing test first.
  For anything touching leakage, chronology or probabilities, add the assertion to
  the guarantees in [docs/testing.md](docs/testing.md).
- **Documentation, if behaviour changed.** The root guides ([DATA.md](DATA.md),
  [ML.md](ML.md), [API.md](API.md), [DEPLOYMENT.md](DEPLOYMENT.md)) are the entry
  points; `docs/` holds the detail. An endpoint change also means
  [API.md](API.md) and the OpenAPI test that pins the route list.
- **A migration, if the schema changed** (below).
- **No new dependency without a reason in the pull request.** The serving image is
  deliberately small.

## House rules

These are not style preferences; they are what makes the output trustworthy.

1. **No fabricated data.** Not in the database, not in the UI, not in a test
   fixture pretending to be production. Missing data is shown as missing — a dash,
   `null`, "not available" — never a zero standing in for a number nobody has.
2. **No leakage.** Features are computed as of a cutoff and never see anything
   after it, including a result that has only just become knowable. The proofs are
   mutation-checked; if you change the feature path, run them and check they still
   fail when the protection is removed.
3. **The test set never chooses anything.** Selection and tuning use validation
   data only.
4. **No secrets in the repository.** Secrets are `SecretStr` settings read from the
   environment. `backend/tests/test_no_secrets.py` scans every committable file and
   fails on a private key, a connection string with a password, or a filled-in
   provider key.
5. **Say what is not known.** A model with no valid explanation method returns a
   reason, not an approximation; a metric with nothing behind it is `null`.
6. **Do not rewrite working functionality** to accommodate a new feature if
   extending it will do.

## Writing a migration

```bash
cd backend
alembic revision -m "add thing"          # then write upgrade() and downgrade()
alembic upgrade head
alembic downgrade -1 && alembic upgrade head    # prove the round trip
```

`backend/tests/db/test_migrations.py` applies every migration from scratch,
downgrades to base and upgrades again, and checks the result matches the models and
that every foreign key is indexed. Add an index when you add a query pattern that
sorts or filters on a new column — and measure it, as
[docs/operations.md](docs/operations.md) does.

## Adding things

| You want to… | Start at |
| --- | --- |
| Add a competition | [DATA.md](DATA.md) — ingest, build features, add to `JOB_COMPETITIONS` |
| Add a data provider | [DATA.md](DATA.md) — implement the interface, register it, pass the contract suite |
| Add a model | [docs/models.md](docs/models.md) — implement `Predictor`, produce a score grid, add it to the comparison |
| Add a feature | [docs/features.md](docs/features.md) — register it, compute it point-in-time, add a leakage test; the feature-set version changes, so snapshots and models must be rebuilt |
| Add an endpoint | [API.md](API.md) — route, schema, service, tests, and the OpenAPI route-list test |
| Add a scheduled job | [docs/scheduled-jobs.md](docs/scheduled-jobs.md) — task, schedule setting, `JobRun` recording, CLI entry |

## Conventions

**Python.** Ruff (format and lint) and mypy strict; line length 100. Type
annotations everywhere, including tests. Services take a `Session`; routes stay
thin. Secrets are `SecretStr`. Lazy-import the ML stack in code the serving image
runs, so the small image keeps working.

**TypeScript.** Prettier and ESLint with type-aware rules; no `any`. Pages read data
through `useApi`, which owns loading, error and abort behaviour. Every list has an
empty state that says what is missing.

**Tests.** Name the behaviour, not the function: `test_an_unknown_team_name_is_
refused_rather_than_guessed`. Arrange, act, assert, with blank lines between. Tests
that need PostgreSQL or Redis are marked `@pytest.mark.integration`.

**Commits.** A message says what changed and why, in the imperative. A pull request
says what it changes, how it was verified, and what it deliberately does not do.
