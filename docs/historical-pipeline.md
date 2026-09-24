# Historical data pipeline and data quality

How match history is imported, normalised and validated, how questionable data is
recorded, and how to inspect data quality. Provider details are in
[data-providers.md](data-providers.md); the schema is in [database.md](database.md).

## Commands

```bash
cd backend      # or: docker compose exec backend python -m app.ingestion …
python -m app.ingestion backfill --competition 39                 # every season offered
python -m app.ingestion backfill --competition 39 --from 2015 --to 2023
python -m app.ingestion backfill --competition 39 --season 39:2023 --no-statistics
python -m app.ingestion backfill --competition 39 --retry-missing-statistics
python -m app.ingestion audit                                     # database-wide checks
python -m app.ingestion report                                    # data-quality report
python -m app.ingestion report --json
```

`backfill` needs a provider key. `audit` and `report` only read the database.
Exit codes: `0` success, `1` partial (some records failed), `2` failed, `3`
configuration error.

## What a backfill does

For one competition, `HistoricalImport` (`app/ingestion/history.py`):

1. Syncs the competition and its seasons.
2. Selects seasons: all the provider lists (default), or `--season` IDs, filtered by
   `--from`/`--to` on the season's starting year. **Historical depth is whatever the
   provider offers**; nothing is hard-coded.
3. Oldest season first, for each season: teams, then matches, then team match
   statistics.
   - Statistics are requested only for **finished matches that have none yet**,
     batched 20 fixtures per request.
   - If the provider's own coverage flag says a season has no statistics, none are
     requested. The gap is recorded as a season-level `missing_statistics` (info)
     issue.
   - A finished match for which the provider returns no statistics gets a
     match-level `missing_statistics` (info) issue. Later runs don't ask again unless
     `--retry-missing-statistics` is given.
4. Stops the whole import on `QuotaExceededError` or `AuthenticationError`, since
   nothing more can succeed today. Any other provider failure fails only that
   season.

Re-running is safe: stored rows are updated in place. Against the test fixtures,
the first run of two seasons used 8 provider requests; an unchanged re-run used 6,
with no statistics requests.

## Normalisation (`app/ingestion/normalize.py`)

Applied to every record from every provider before storage:

| What | Rule |
| --- | --- |
| Text (team, competition, venue, referee names) | Unicode NFC; repeated and surrounding whitespace collapsed; empty → NULL. Stored names keep the provider's spelling |
| Team codes | Upper-case |
| Season names | `2023-24`, `2023/24`, `2023–24`, `2023-2024` → `2023/2024`; `1999/00` → `1999/2000`; `2024` stays. Anything else is kept as is and flagged, never guessed |
| Kickoff times | Must be timezone-aware; converted to UTC; exact to the second. Naive times are rejected |
| Provider IDs | Trimmed strings; empty IDs are rejected |
| Name comparison key | Accents, case, punctuation and club affixes (FC, CF, AFC, SC, …) removed. Used **only** to detect possible duplicate teams, never to rename |

## Chronology (for leakage prevention later)

- `matches.kickoff_at` is a `timestamptz` holding the exact kickoff in UTC; it is
  never NULL. A test compares every stored kickoff with the provider's value.
- Kickoffs before 1860 or more than 3 years in the future are treated as corrupt:
  the match is not stored, and the issue keeps the full record.
- `data_ingestion_runs` records when each import ran, and every issue records when
  it was first and last seen. Together they show when a fact became known.

## Validation policy: store and flag, or reject and record

A record is **stored and flagged** when it is questionable but storable. It is
**not stored** only when storing it would break a database constraint or the
timeline; the issue then carries the complete provider record in `details`.
Nothing is ever deleted: records a provider stops listing stay in the database,
and issues are *resolved* (with `resolved_at`), never removed.

| Problem | Detected by | Type / severity | Stored? |
| --- | --- | --- | --- |
| Malformed provider record | adapter / ingestion | `malformed_record` / error | no (record kept in issue) |
| Match or statistics refer to an unknown team | ingestion | `missing_team` / error | no |
| Match listed under the wrong competition/season | ingestion | `inconsistent_competition` / error | no |
| Same provider ID twice in one response | ingestion | `duplicate_provider_id` / warning | first copy only; both kept in issue |
| Fixture already stored in another season | ingestion | `duplicate_match` / error | no (never moved silently) |
| Half-time score above final score; shoot-out after a decisive or level result | ingestion + audit | `impossible_score` / error | yes |
| More than 15 goals for one team; score on an unplayed match | ingestion + audit | `impossible_score` / warning | yes |
| Kickoff far outside its season (±45 days) | ingestion + audit | `invalid_date` / warning | yes |
| Finished match kicking off in the future | ingestion + audit | `invalid_date` / error | yes |
| Corrupt kickoff (before 1860, >3 years ahead) | ingestion | `invalid_date` / error | no |
| Season name in an unknown format; season without dates | ingestion | `incomplete_record` / warning, info | yes |
| Statistics for a team that did not play; shots on target > shots | ingestion + audit | `inconsistent_statistics` / error | no |
| Possession not adding up to 100% (±2) | audit | `inconsistent_statistics` / warning | yes |
| Finished match or season without statistics | ingestion | `missing_statistics` / info | — |
| Same fixture twice within 48 h; team playing twice within 12 h | audit | `duplicate_match` / warning | yes |
| Teams whose names match after normalisation | audit | `possible_duplicate_team` / warning | yes (never merged automatically) |
| Two seasons of a competition with overlapping dates | audit | `inconsistent_competition` / warning | yes |
| Match still "scheduled" 3+ days after kickoff | audit | `incomplete_record` / warning | yes |

### The `data_quality_issues` table

One row per problem, not per sighting:

- A `fingerprint` (source, type, entity, record, rule) is unique among open issues.
  Seeing the problem again updates `last_seen_at` and `occurrences`.
- `detector` records which check found it (`ingestion:matches`,
  `ingestion:team_statistics`, `audit`, …). Only that check resolves it, when a
  re-check no longer finds the problem. This scoping matters: re-syncing matches
  must not resolve statistics gaps on the same matches. A test covers this.
- `entity_type` + `entity_id` point at the stored row when there is one;
  `external_id` identifies the provider record, including records that were never
  stored.

```sql
-- Open errors, newest first
SELECT detector, issue_type, entity_type, entity_id, external_id, message, occurrences
FROM data_quality_issues
WHERE resolved_at IS NULL AND severity = 'error'
ORDER BY last_seen_at DESC;
```

Future feature code should exclude matches with open `error` issues.

## Data-quality report

`python -m app.ingestion report` prints (or `--json` emits):

- **Totals:** competitions, seasons, teams, matches, finished matches, statistics
  rows, data sources.
- **Missing statistics:** finished matches without statistics, with only one team's,
  and without xG.
- **Duplicate records:** open duplicate-match, duplicate-provider-ID and
  possible-duplicate-team issues.
- **Incomplete records:** missing half-time scores, venues, referees, matchday or
  stage, stale "scheduled" matches, seasons without dates, teams without country,
  statistics rows without shots or possession.
- **Open issues** by type and severity.
- **Coverage by competition and by season:** matches, finished, % with statistics,
  % with xG, first and last kickoff.

Output from the fixture-backed verification run:

```
Coverage by season
                                           seasons  matches finished   stats      xG  first      last
  Premier League 2022/2023                       1       10       10    0.0%    0.0%  2022-08-05 2022-08-07
  Premier League 2023/2024                       1       10       10   90.0%   90.0%  2023-08-11 2023-08-14
```

## Known limitations

- **Not yet run against the live API.** No key is available; the pipeline is verified
  against fixtures. The statistics *values* in those fixtures are synthetic (see
  `backend/tests/fixtures/api_football/README.md`).
- **Only team match statistics are imported.** Player statistics, events, lineups,
  injuries and odds are not (the interface allows them).
- **Duplicate detection only flags.** Possible duplicate teams and fixtures are never
  merged; resolving them is a manual decision, and there is no merge tool yet.
- **Statistics are fetched once per match.** A later provider correction to a match's
  statistics is not picked up automatically, because matches that have statistics
  are not re-requested.
- **The audit is whole-database.** The duplicate-fixture and team-clash self-joins
  will need date-window partitioning once the database holds many seasons across
  many leagues.
