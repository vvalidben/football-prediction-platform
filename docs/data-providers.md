# Football data providers and ingestion

How external football data enters the platform: provider selection, the
provider-independent interface, the API-Football adapter, and idempotent
synchronisation into PostgreSQL.

## Provider selection (September 2026)

Requirements: historical data, many leagues (the big five, Champions League,
Moroccan Botola, international tournaments), match statistics, players, injuries,
lineups, xG and odds.

| Provider | Coverage | Stats, lineups, injuries, players | xG | Odds | Cost | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **API-Football** (api-sports.io) | ~1,200 leagues and cups; per-season coverage flags | Yes: events, lineups, team and player stats, injuries | Only for some competitions (`expected_goals` in fixture statistics) | Pre-match and in-play | Free: 100 req/day, limited seasons. Pro $19/mo (7,500/day), Ultra $29, Mega $39 | **Primary.** Broadest all-in-one coverage at low cost |
| Sportmonks | 2,200+ leagues | Yes | Paid add-on (€24 entry, up to ~€300/mo for wide coverage) | Yes | From €29/mo for 5 leagues; realistic setups €500–1,000/mo | Best xG, too expensive to start with. Candidate second adapter |
| football-data.org | 12 competitions on the free tier | Fixtures, results, tables only | No | No | Free: 10 req/min | Too narrow (no stats, injuries, odds or Botola) |
| StatsBomb Open Data / Understat | Selected competitions / top-5 leagues | Event data (StatsBomb) | Yes | No | Free | Useful *supplementary* xG sources later |

**Choice: API-Football as the primary provider.** It is the only low-cost option
covering every data type in the brief, including injuries, lineups and odds. Botola
Pro is listed on API-Football's official coverage page under Morocco (checked by the
project reviewer on 2026-09-22; an earlier draft of this document stated it without
evidence). Which Botola seasons exist and what they contain is still unverified.
Its weak point is xG: it exists only for some competitions. The architecture therefore treats providers as
interchangeable, so an xG-focused source (Sportmonks, StatsBomb, Understat) can be
added as another adapter without touching ingestion or model code.

**Decision status: provisionally accepted, still unverified.** The historical
pipeline was built on this adapter, but the coverage check below has never run with
a real key — so what this provider actually returns for a given plan is documented
from its specification, not observed. Running it is the first thing to do when a key
exists.

**Credentials.** No API key was available in this project when the adapter was
written. The adapter therefore follows API-Football's documented v3 format and is
tested against recorded-style fixtures (see `backend/tests/fixtures/api_football/`).
Before relying on it in production, run
`python -m app.ingestion competitions --search "Premier"` with a real key and
confirm responses match the fixtures.

Sources: [API-Football pricing](https://www.api-football.com/pricing),
[API-Football v3 documentation](https://www.api-football.com/documentation-v3),
[API-Football on rate limits](https://www.api-football.com/news/post/how-ratelimit-works),
[Sportmonks pricing](https://www.sportmonks.com/football-api/plans-pricing/),
[football-data.org policies](https://docs.football-data.org/general/v4/policies.html),
[Highlightly: best football APIs 2026](https://highlightly.net/blogs/best-football-apis-in-2026),
[bruin-data/ingestr API-Football research notes](https://github.com/bruin-data/ingestr/blob/76c5a9b4d03ee2ed7f3f2942831222a78898a4fe/docs/soccer-sources/api-football.md).

## Verification status

What is verified, and how, as of 2026-09-22:

| Claim | Status |
| --- | --- |
| Botola Pro exists in API-Football | Verified at competition level (official coverage page, reviewer) |
| Botola seasons and per-season coverage | **Unverified.** Needs the coverage check |
| Historical depth of stats, lineups, injuries (big five, UCL, Botola) | **Unverified.** Needs the coverage check |
| `expected_goals` availability per competition | **Unverified.** One third-party report (Brasileirão); needs the coverage check |
| Pricing and quotas (Free 100/day, Pro 7,500/day) | Verified (official pricing page, reviewer) |
| Per-minute limits (Free 10, Pro 300) | Verified (official pricing page, reviewer) |
| Odds limited to the last 7 days | Verified (official World Cup 2026 guide, reviewer) |
| `ids` batching up to 20 fixtures | Reviewer-reported; exercised by the coverage check |
| Errors inside HTTP 200 bodies (`token`, `requests`, `rateLimit`) | Unverified; from client libraries. The coverage check saves raw responses |
| Live response shapes match the test fixtures | **Unverified.** Needs a key |

Official api-football.com pages return HTTP 403 to this project's tooling, so
facts marked "reviewer" were checked by a person on the official site.

### Coverage check (run once a key is available)

```bash
cd backend
python -m app.providers.api_football.coverage            # budget: 60 requests
python -m app.providers.api_football.coverage --older-offset 8 --budget 60
```

It finds Botola by searching Morocco's leagues and verifies each known league ID
against the name the API returns. For each target competition it samples the most
recent completed season and one season `--older-offset` years earlier, picks a
mid-season completed fixture, then:

- fetches all sampled fixtures' details in one batched `ids` request;
- checks injuries and odds per fixture;
- checks one upcoming fixture's odds and their `update` timestamp.

It writes `reports/api_football_coverage/<timestamp>/report.md` with:

- a matrix (Competition | Season | Fixtures | Stats | xG | Lineups | Injuries | Players | Odds | Notes) whose cells show what was **observed** next to the provider's **claimed** flag;
- the statistic labels actually returned;
- every season's claimed coverage flags;
- the account's quota before and after, the rate-limit headers, and a log of every request.

Every raw response is saved next to the report. About 54 requests are used with 7
competitions; the free plan allows 100/day.

## Architecture

```
 External API ─► Adapter (app/providers/<name>/) ─► neutral records ─► ProviderSync ─► PostgreSQL ─► (future) features / ML
                  payloads.py  normalize.py         (schemas.py)       (app/ingestion)
                       ▲
                 JsonHttpClient: timeouts · retries · rate limiting (app/providers/http.py)
```

- **`FootballDataProvider`** (`app/providers/base.py`) is the contract. Four
  operations are required (`get_competitions`, `get_seasons`, `get_teams`,
  `get_matches`). Six are optional (`get_match_details`, `get_team_statistics`,
  `get_player_statistics`, `get_injuries`, `get_lineups`, `get_odds`) and raise
  `UnsupportedOperationError` unless implemented. `capabilities` declares which
  ones a provider serves.
- **Neutral records** (`app/providers/schemas.py`) are frozen Pydantic models that
  validate on construction (e.g. a finished match must have a score, and kickoff
  must be timezone-aware). All identifiers are the provider's external IDs, as
  strings.
- **`FetchResult`** carries `records` plus `rejected`. One malformed record never
  fails a whole call.
- **Errors** (`app/providers/errors.py`) are provider-neutral. Adapters translate
  HTTP statuses and provider error payloads into `ProviderUnavailableError`,
  `RateLimitError`, `QuotaExceededError`, `AuthenticationError`,
  `ProviderRequestError`, `MalformedResponseError` and `UnsupportedOperationError`.
  Each says whether it is `retryable`.
- **Decoupling from ML.** Only adapters know provider formats. Ingestion knows
  neutral records and the database. Future feature and model code will read only
  the database, so swapping or adding providers never touches it.

### Resilience (`JsonHttpClient`)

| Situation | Behaviour |
| --- | --- |
| Timeout, connection error, HTTP 5xx | Retry with exponential backoff and jitter (1 s, 2 s, 4 s … capped at 30 s), up to `PROVIDER_MAX_ATTEMPTS` attempts in total |
| HTTP 429 | Retry after `Retry-After` (capped at 120 s), otherwise backoff |
| Rate limit reported in an HTTP 200 body (`errors.rateLimit`) | Retry after 60 s (per-minute window) |
| Daily quota exhausted (`errors.requests`) | `QuotaExceededError`, not retried |
| Bad key (HTTP 401/403 or `errors.token`) | `AuthenticationError`, not retried |
| Other 4xx or other `errors` entries | `ProviderRequestError`, not retried |
| Non-JSON body, missing envelope | `MalformedResponseError`, not retried |
| Malformed single record | Rejected, counted as a failed record; the rest proceed |
| Request pacing | Client-side limiter spaces calls to `API_FOOTBALL_REQUESTS_PER_MINUTE` |

## API-Football adapter

| Operation | Endpoint | Notes |
| --- | --- | --- |
| competitions | `GET /leagues` | `type` League/Cup → league/cup; "Friendlies" → friendly; super cups by name. Country `World` → international (no country) |
| seasons | `GET /leagues?id={league}` | Seasons are embedded in the league. External ID `"{league}:{year}"`, because API-Football's season year is not unique across competitions. Name `2023/2024` when the season spans two years, otherwise `2024` |
| teams | `GET /teams?league=&season=` | |
| team statistics | `GET /fixtures?ids=a-b-…` (≤ 20 per request) | Statistic labels mapped to columns (`Total Shots` → shots, `Shots on Goal` → shots on target, `Ball Possession` "55%" → 55, `expected_goals` "1.73" → 1.73, …); unmapped labels go to `extra`; nulls stay NULL. Seasons expose `coverage.fixtures.statistics_fixtures` as `statistics_available` |
| matches | `GET /fixtures?league=&season=` | Status codes mapped as below; kickoff normalised to UTC; `"Regular Season - 12"` → stage `regular_season`, matchday 12. `goals` (includes extra time) → score; `score.halftime` and `score.penalty` → half-time and shoot-out scores |

Status mapping: `TBD`/`NS` → scheduled; `1H` `HT` `2H` `ET` `BT` `P` `LIVE` `INT` →
live; `FT` `AET` `PEN` `AWD` `WO` → finished; `PST` → postponed; `CANC` →
cancelled; `ABD` → abandoned; `SUSP` → suspended. An unknown code rejects the
record.

**Competition IDs** for the targets. These are widely used API-Football IDs;
confirm them with `python -m app.ingestion competitions` once you have a key:
Premier League 39, La Liga 140, Serie A 135, Bundesliga 78, Ligue 1 61, Champions
League 2. Look up Botola Pro and international competitions rather than
hard-coding them:

```bash
python -m app.ingestion competitions --country Morocco
python -m app.ingestion competitions --search "World Cup"
```

### API limitations to plan around

- **Quota.** The free plan allows 100 requests/day and only some seasons, which is
  enough for development but not for backfilling history. Pro ($19/mo) allows
  7,500/day, per the official pricing page (confirmed by the reviewer).
- **Per-minute limit depends on the plan.** Per the official pricing page (confirmed
  by the reviewer): Free 10 requests/minute; Pro 300 requests/minute (5 per second).
  Set `API_FOOTBALL_REQUESTS_PER_MINUTE` to your plan's value. The default of 10 is
  only safe for the free plan and needlessly slow on Pro. The coverage check
  records the provider's actual rate-limit headers so the value can be confirmed.
- **Batch fixture requests.** `GET /fixtures?ids=a-b-c` takes up to 20 fixture IDs
  and returns their details (events, lineups, statistics, players) in one
  request. Historical ingestion should use it to save quota; the coverage check
  already does.
- **Coverage varies by league and season.** Each season carries `coverage` flags
  (lineups, statistics, injuries, odds …). Smaller leagues and older seasons often
  lack statistics, lineups or odds, and xG is only available for some
  competitions.
- **Errors can arrive with HTTP 200.** They are reported in the body's `errors`
  object; handled as described above.
- **Odds history is only 7 days deep.** API-Football's World Cup 2026 guide states
  that `/odds` exposes odds from the last 7 days only (confirmed by the reviewer on
  the official page). Old matches therefore have no odds, and a historical
  closing-odds archive must come from another source (e.g. football-data.co.uk CSVs
  for the major leagues) or be built going forward by snapshotting pre-match odds
  on a schedule. `/odds` paginates at 10 items per page.
- **Not yet verified live.** Envelope and error shapes follow the documentation and
  published client libraries. Confirm them with a real key before production use.

## Ingestion (`app/ingestion`)

`ProviderSync(provider, session_factory)` offers `sync_competitions`,
`sync_seasons`, `sync_teams`, `sync_matches`, and `sync_competition`, which runs
all four for the current (or chosen) seasons.

- **Idempotent.** Each record is resolved through `<entity>_external_ids`
  `(data_source, external_id)`. If a mapping exists, the row is updated in place;
  otherwise it is inserted together with its mapping. A second run inserts nothing
  (`unchanged` counts rise instead). A season or match that already exists under
  its natural key (competition + name, or home + away + kickoff) is linked rather
  than duplicated.
- **Isolated failures.** Each record is applied in its own savepoint. A record
  referencing an unsynced team, conflicting with stored data, or rejected by a
  database constraint is counted as failed; the rest commit and the run is
  `partial`.
- **Concurrency.** A transaction-scoped PostgreSQL advisory lock per data source
  serialises concurrent syncs of the same provider. A test runs two syncs in
  parallel and fails if the lock is removed.
- **Missing data.** Descriptive fields (logo, country, venue, referee, matchday,
  stage) are not erased when a provider omits them. State (status, kickoff,
  scores) always follows the provider, so postponements and score corrections
  apply.
- **Conflicts are not auto-resolved.** A fixture that already exists in a
  different season is reported as a failed record, not silently moved.

### Run log

Every operation writes one `data_ingestion_runs` row, in its own transaction so
the log survives a failed sync:

| Field | Content |
| --- | --- |
| `data_source_id` | provider |
| `job` | operation: `competitions`, `seasons`, `teams`, `matches` |
| `parameters` | e.g. `{"competition": "39", "season": "39:2023"}` |
| `started_at`, `finished_at` | UTC timestamps |
| `status` | `running` → `succeeded` / `partial` / `failed` |
| `records_fetched` / `_inserted` / `_updated` / `_failed` | counts (unchanged = fetched − others) |
| `error_message` | exception type and message for failed runs; the first 10 record errors for partial runs |

```sql
SELECT r.id, s.code, r.job, r.status, r.records_fetched, r.records_inserted,
       r.records_failed, r.finished_at - r.started_at AS duration, r.error_message
FROM data_ingestion_runs r JOIN data_sources s ON s.id = r.data_source_id
ORDER BY r.id DESC LIMIT 20;
```

### Command line

```bash
cd backend   # or: docker compose exec backend python -m app.ingestion …
python -m app.ingestion competitions --country England
python -m app.ingestion sync --competition 39                   # current season
python -m app.ingestion sync --competition 39 --season 39:2023  # specific season(s)
python -m app.ingestion sync --competition 39 --competition 140 --season all
```

Exit codes: `0` succeeded, `1` partial (some records failed), `2` an operation
failed, `3` configuration error (e.g. `API_FOOTBALL_API_KEY` not set).

## Adding a provider

1. Create `app/providers/<name>/` with payload models, normalisation functions and
   a `FootballDataProvider` subclass. Build on `JsonHttpClient` so it gets
   retries, timeouts and rate limiting.
2. Register its factory in `app/providers/registry.py` and add its settings to
   `app/core/config.py` and `.env.example`.
3. Add fixtures and tests, and add it to the parametrised contract tests in
   `tests/providers/test_provider_contract.py`.

Ingestion needs no changes. A second provider's teams and matches become separate
canonical rows until cross-provider entity matching is implemented (see
limitations).

## Known limitations

- **The live API is unverified**, because there was no key (see above).
- **Competitions, seasons, teams, matches and team statistics are ingested** (team
  statistics since Sprint 3). The interface and neutral types exist for match
  details, player statistics, lineups, injuries and odds, but nothing implements
  them yet.
- **No cross-provider entity matching.** The same club from two providers becomes
  two `teams` rows. Matching (by name, country and fixtures) is future work; the
  schema already supports several IDs per entity.
- **No team–season membership table.** Teams are synced per season but stored
  canonically.
- **No scheduler yet.** Syncs run via the CLI. Automatic updates (e.g. a Redis-backed
  worker or cron) come later.
- **Rate limiting is per process.** Parallel sync processes would each apply the
  limit; run one ingestion process per provider.
