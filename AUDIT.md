# Final audit

A review of the whole system: what it is, what was found, what was fixed, and what
it cannot do. Sprint 19.

> **What this system does not do.** It does not reliably predict exact football
> scores, and nothing here should be read as a claim that it can. It produces
> *probability distributions* over outcomes: the "most likely score" typically
> carries a probability around 10–13%, which means it is wrong roughly nine times
> out of ten. Football is close to irreducibly uncertain at the scoreline level.
> What the platform aims at is honest, calibrated probabilities — the outcomes it
> gives 30% to should happen about 30% of the time — and it measures whether it
> achieves that rather than asserting it.

## Findings and fixes

Four defects were found by measurement during this audit. All four are fixed, and
each fix carries a test that fails if it regresses.

| # | Area | Finding | Evidence | Fix |
| --- | --- | --- | --- | --- |
| 1 | Backend performance | Listing endpoints loaded the competition, season and both teams **per row** (N+1). Earlier tests missed it because their fixture has six teams, so the identity map absorbed the repeats. | 32 SQL statements for 12 matches vs 12 for 2 | `joinedload` on the match queries and the stored-prediction listing; `find_match` loads a match and its relations in one query. Listings are now a constant ≤4 statements. `tests/api/test_query_efficiency.py` fails if the count grows with the page size. |
| 2 | Database safety | No `statement_timeout`. One runaway query could hold a connection until the pool (5 + 10) was exhausted and the whole API stalled. | `SHOW statement_timeout` → `0` | API connections set `statement_timeout` (30 s) and `idle_in_transaction_session_timeout`. Jobs opt out — a backfill runs long queries on purpose. Tests prove PostgreSQL actually cancels an over-long query. |
| 3 | Frontend UX | Every page had the same browser-tab title, so tabs, bookmarks and history were indistinguishable. | `document.title` never changed | `useDocumentTitle` names each page and restores the previous title on unmount; the parameterised pages use the team or fixture name. |
| 4 | Scheduled jobs | A job slower than its interval was started again on the next tick. Two fixture fetches spend the provider quota twice for the same data, and the free tier is 100 requests a day. | No overlap guard in the task runner | A tick that finds the same job running is skipped and recorded as skipped. A run counts as live only until the task time limit, so a dead worker cannot block a job forever; retries are always allowed through. |

Two smaller changes came out of the wording review: the dashboard and the history
table now say "most likely score" rather than presenting a bare scoreline.

## What was checked and found sound

- **Leakage** — re-verified by mutation: removing the result delay from the feature
  builder makes 5 leakage tests fail; restoring it makes all 15 pass.
- **Layering** — no route imports the ML stack directly, no service imports a route,
  no model imports a service.
- **Documented commands and variables** — every documented CLI subcommand exists,
  every documented environment variable maps to a real setting, and no setting is
  undocumented (checked programmatically).
- **Dependencies** — `pip-audit` and `npm audit` report no known vulnerabilities.
- **Secrets** — a test scans every committable file for private keys, connection
  strings with passwords and filled-in provider keys, and asserts the browser bundle
  reads no secret and sends no credentials.

## 1. Architecture

```
Browser ─► nginx (SPA + /api proxy) ─► FastAPI ─┬─► PostgreSQL   (system of record)
                                                └─► Redis        (queue, counters, cache)
                                   Celery worker ─┘   ▲
                                   Celery beat ───────┘
```

Layers, outermost first: **routes** (HTTP, validation, status codes) → **services**
(the work, taking a `Session`) → **models** (SQLAlchemy) → the database. The ML
stack sits beside them and is imported lazily, so the serving image never needs the
training libraries. Providers sit behind one interface, so the ingestion layer and
everything above it are provider-independent.

**Maintainability.** 17.7k lines of backend Python across ~90 modules; the largest
is 986 lines (`ingestion/sync.py`). Strict mypy, ruff, and 901 tests at 97% line
coverage. Documentation is a root guide per topic over reference pages per
subsystem, and links, commands and variables are checked mechanically.

**Scalability.** The API is stateless and scales horizontally behind the proxy; the
database is the single writer. `beat` must stay at exactly one instance. Predictions
are computed synchronously inside the request, which is the first thing that would
need a queue under load.

## 2. Technology stack

| Layer | Choice |
| --- | --- |
| Backend | Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic |
| ML | numpy, scipy, scikit-learn, XGBoost, LightGBM, MLflow, SHAP |
| Database | PostgreSQL 17 |
| Cache / queue | Redis 7.4, Celery (worker + beat) |
| Frontend | React 19, TypeScript, Vite, Tailwind v4, react-router |
| Infrastructure | Docker, Docker Compose, nginx |
| Quality | pytest, Vitest + Testing Library, ruff, mypy (strict), ESLint, Prettier, pip-audit, npm audit |

The serving image installs a deliberately small subset (`serve`: numpy, scipy,
scikit-learn) — no MLflow, XGBoost, LightGBM or SHAP.

## 3. Data providers

One interface (`FootballDataProvider`), provider-neutral record schemas, shared
error types. **API-Football** is the adapter that ships; a mock provider serves
tests and is deliberately absent from the registry the CLI uses.

Reliability: client-side pacing to the plan's limit, retries with exponential
backoff and jitter, `Retry-After` honoured, permanent errors (auth, quota,
unsupported) never retried, every run logged in `data_ingestion_runs`.

**The honest gap:** no API key has ever been available in this project. The adapter
follows the documented v3 API and is tested against recorded-style fixtures and a
contract suite — but what the provider actually returns has never been observed.

## 4. Database design

One row per real-world thing; provider ids live in side tables
(`*_external_ids`), so two providers can describe the same match without colliding.
Matches carry a composite foreign key so a match's season always belongs to its
competition. Probabilities have CHECK constraints; predictions are immutable;
feature snapshots are keyed by `(match_id, feature_set_version, data_cutoff_at)`.

Every foreign key is indexed (proven by test), plus indexes for the two hot
orderings — measured: 7.1 ms → 0.50 ms on 60k predictions.

## 5. ML pipeline

```
snapshots (leak-free, versioned) ─► chronological split ─► tune on validation
   ─► fit ─► evaluate on test ─► register version ─► activate ─► serve ─► score
```

Features are computed **as of a cutoff** and never see anything after it; a result
counts as knowable only after a delay (3 h), so a match that has just finished
cannot leak into a prediction for one kicking off now. The feature-set version is a
hash of the feature list and configuration, so a model always states what it was
trained on.

## 6. Models

`historical_baseline` (the floor), `elo`, `poisson` (Dixon-Coles),
`logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, and fitted
`ensemble`s. Every one produces a single 11×11 exact-score distribution; 1X2,
expected goals, most probable score, over/under and both-teams-to-score are all
derived from that one grid, so they cannot contradict each other.

## 7. Evaluation methodology

Metrics that reward honest probabilities: log loss, Brier, ranked probability
score, alongside accuracy and exact-score accuracy. Comparisons carry **paired
bootstrap intervals** — "better" means measurably better, not better on one split.
An ensemble is recommended only when its 95% interval against the best single model
lies below zero. The test set never selects anything; a test poisons it and checks
the selection does not move.

Measured live performance is kept separate from training metrics in the UI and the
API, because training numbers are not evidence about future matches.

## 8. Backtesting methodology

Walk-forward over history: at each step the model sees only matches whose results
were knowable then, predicts the next block, and the prediction is stored **before**
the result is revealed. Expanding and rolling windows; calibration (Platt,
isotonic) fitted only on earlier predictions; a guard raises `BacktestLeakError` if
a step would train on anything later. Results break down by competition, season and
prediction type.

## 9. API

REST over the data and models: catalog, predictions (single and batch), stored
predictions, history with per-filter summaries, explanations, model registry and
performance, data status and job status. Every response carries a request id; a 500
returns that id and nothing else. Rate limited per address (300/min, 30/min for
predictions), `/docs` and `/openapi.json` withheld in production, security headers
on every response, and only the two status endpoints cached. Full reference:
[API.md](API.md).

## 10. Frontend

Six pages — dashboard, match prediction, prediction history, teams, model
performance, data status — in React with hand-drawn SVG charts (no charting
dependency). Every page has an explicit loading state, an error state showing the
backend's own message with a retry, and an empty state that says what is missing.
**No fabricated data anywhere:** a missing number is a dash, never a zero. An error
boundary keeps a broken view from blanking the application. 150 frontend tests.

## 11. Deployment

`docker-compose.prod.yml`: production mode, no published database ports, images
pinned by tag, read-only containers with capabilities dropped, resource limits, log
rotation, one scheduler. TLS terminates in front of the stack.
[DEPLOYMENT.md](DEPLOYMENT.md) covers the procedure, five deployment options with
their trade-offs, upgrades, rollback and backups. **Nothing has been deployed.**

## 12. Known limitations

**About the predictions themselves**

- Exact scores are not reliably predictable. The most likely score is usually right
  around 10–13% of the time, and the platform shows that probability rather than
  hiding it.
- Model quality has been measured on synthetic data and one real league's history.
  Nothing here establishes performance on competitions it has not seen.
- No model beats the market; nothing here is betting advice, and no odds-derived
  edge has been demonstrated.

**About the data**

- The provider has never been called with a real key, so its live behaviour,
  coverage and quirks are unverified.
- xG and other statistics exist only where a provider supplied them.
- Promoted and new teams start at the default rating, so their first matches are
  predicted from the league average plus home advantage.

**About the system**

- Predictions are computed synchronously inside the request.
- Single-host deployment; `beat` is a single point of scheduling.
- No authentication (deliberate, documented), no secrets manager, no CI pipeline,
  no metrics backend, no browser end-to-end test, no load testing.
- Rate limits, resource limits and schedules are starting points chosen without
  production traffic.

## 13. Future improvements

Ordered by what would most change the platform's usefulness:

1. **Run the provider with a real key** and verify coverage, then re-measure
   everything on live data. Every other improvement is speculative until this is
   done.
2. **Calibration in production**: serve calibrated probabilities (the machinery
   exists in backtesting) and show the reliability diagram for the live model.
3. **Move prediction off the request path** — queue it, serve the stored result —
   so a batch cannot occupy a worker.
4. **Player-level features**: lineups, availability and minutes, which is where the
   biggest modelling gain probably sits.
5. **Automatic promotion with a guard**: promote a newly trained version only when
   it beats the active one on a held-out period by a measurable margin.
6. **Operational maturity**: CI, structured logs, metrics, alerting on job failure,
   and a restore drill for the backups.
7. **Frontend depth**: pagination controls, a team type-ahead, and per-line
   over/under outcomes in the history.
