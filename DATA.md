# Data

Where the numbers come from, what happens to them, and how to add more. This is the
guide; the reference pages under `docs/` go deeper on each stage.

```
provider  ──►  ingestion  ──►  PostgreSQL  ──►  features  ──►  models
(API-Football)  (idempotent,     (one row per     (leak-free      (docs/models.md)
                 logged)          match, one        snapshots)
                                  truth)
```

| Stage | Reference |
| --- | --- |
| Providers and the adapter interface | [docs/data-providers.md](docs/data-providers.md) |
| Historical backfill, normalisation, quality | [docs/historical-pipeline.md](docs/historical-pipeline.md) |
| Schema, keys and constraints | [docs/database.md](docs/database.md) |
| Features and leakage prevention | [docs/features.md](docs/features.md) |

## The data provider

The platform is provider-independent: every adapter implements one interface
(`app/providers/base.py`) and returns provider-neutral records. **API-Football** is
the adapter that ships.

- The key lives in `API_FOOTBALL_API_KEY` and travels in a request header, never in
  a URL or a log line. Without a key, ingestion commands fail fast with a clear
  message — they never invent data.
- Requests are paced client-side to your plan's limit
  (`API_FOOTBALL_REQUESTS_PER_MINUTE`: Free 10, Pro 300), retried with exponential
  backoff, and honour `Retry-After` on 429.
- Every provider id is stored beside the row it refers to (`*_external_ids`
  tables), so two providers can describe the same match without colliding.

## Ingesting data

```bash
cd backend                                   # virtualenv active; see CONTRIBUTING.md
python -m app.ingestion competitions --country England   # find competition ids
python -m app.ingestion sync --competition 39            # current season
python -m app.ingestion backfill --competition 39        # every season the provider offers
python -m app.ingestion backfill --competition 39 --from 2015 --to 2023
python -m app.ingestion audit                            # database-wide checks (no key)
python -m app.ingestion report                           # quality report (no key)
```

In Docker: `docker compose exec backend python -m app.ingestion sync --competition 39`.

Exit codes: `0` success, `1` some records failed (partial), `2` an operation
failed, `3` configuration or provider setup error — so a scheduled run can tell
"nothing to do" from "this is broken".

**Every sync is idempotent.** Re-running updates what changed and inserts what is
new; it never duplicates. Each run is logged in `data_ingestion_runs` with what it
touched, so a partial import is visible rather than silent.

### Historical data

`backfill` walks every season a provider offers for a competition, oldest first,
and imports matches and team statistics. It skips seasons already covered (to save
quota), stops cleanly when the quota runs out, and records what it could not
normalise as a data-quality issue instead of dropping it silently.

Quality issues are recorded, de-duplicated and resolved — never deleted — so the
history of a problem survives the fix. `report` prints coverage and open issues;
the same figures are on the **Data status** page in the web application.

## The database

PostgreSQL 17. One row per real-world thing, with the provider ids kept to the
side. Matches carry a composite foreign key so a match's season always belongs to
its competition; probabilities have CHECK constraints; predictions are immutable.

- Schema, diagram and constraints: [docs/database.md](docs/database.md).
- Migrations are Alembic; the backend container applies them on start
  (`RUN_MIGRATIONS=false` to disable). See [CONTRIBUTING.md](CONTRIBUTING.md) for
  writing one.
- Every foreign key is indexed, and a test proves it.

## Feature engineering

Features are computed **as of a cutoff** — never from anything that happened after
it — and stored as immutable snapshots keyed by
`(match_id, feature_set_version, data_cutoff_at)`.

```bash
python -m app.features version                     # the feature-set version hash
python -m app.features generate                    # snapshots for every match
python -m app.features generate --competition-id 1 --lead-hours 24
python -m app.features show --match-id 42          # inspect one match
python -m app.features export --out features.csv
```

The feature-set version is a hash of the feature list and the configuration, so a
model always states the features it was trained on, and a snapshot built under a
different definition can never be mistaken for a current one.

**Leakage is the thing this project is most careful about.** A result is only
considered knowable some hours after kickoff (`result_delay`), so a match that has
just finished does not leak into a prediction for a match kicking off now. The
proofs are in `backend/tests/features/test_no_leakage.py` and are mutation-checked:
removing a protection makes them fail. See [docs/testing.md](docs/testing.md).

## Adding a competition

1. Find its provider id:
   `python -m app.ingestion competitions --country <country>`.
2. Import it: `python -m app.ingestion sync --competition <id>`, then
   `backfill --competition <id>` for history.
3. Build features for the new matches: `python -m app.features generate`.
4. Add the id to `JOB_COMPETITIONS` (comma separated) so the scheduled jobs keep it
   current — an empty list means the fetch jobs skip, by design.
5. Retrain if you want the model to learn from it: `python -m app.ml train`
   ([ML.md](ML.md)).

Nothing about a competition is hard-coded: cup competitions are marked as such in
the schema and the features use that (`competition_is_cup`), but no rule anywhere
names a league.

## Adding a data provider

1. Implement `FootballDataProvider` (`app/providers/base.py`): return the
   provider-neutral schemas in `app/providers/schemas.py`, and raise the shared
   errors (`AuthenticationError`, `RateLimitError`, `QuotaExceededError`,
   `ProviderUnavailableError`, `UnsupportedOperationError`) so retries and job
   handling behave the same for every provider.
2. Normalise in the adapter, not in the ingestion layer: names, dates,
   competition types and scores come out already in the platform's terms.
3. Register it in `app/providers/registry.py` under its `code`. The mock provider
   is deliberately absent from that registry — it serves test data and must never
   be reachable from the CLI.
4. Add its settings to `app/core/config.py` and `.env.example`; secrets are
   `SecretStr`.
5. Run the contract suite: `backend/tests/providers/test_provider_contract.py`
   runs the same expectations against every registered provider. A new adapter
   passes it or it is not done.
6. Select it per run with `--provider <code>`, or for the jobs with
   `JOB_PROVIDER`.

## What is *not* in the data

- No scraped data, and no fabricated rows anywhere — including in the UI, which
  shows "not available" rather than a zero.
- xG and other statistics exist only where a provider supplied them; the team page
  says how many matches have them.
- Odds are optional and excluded from the default feature set
  (`--with-odds` includes them at training time).
