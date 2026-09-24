# Feature engineering

How match features are computed without data leakage, what they are, and how they
are versioned and stored. Code: `backend/app/features/`.

## The rule

For a match kicking off at time T, predicted at cutoff C ≤ T, every feature may only
use information **known before C**. That excludes:

- the match's own result and statistics;
- future matches and their statistics;
- odds, absences and results that became known later.

## How leakage is prevented (by construction)

1. **One point-in-time timeline.** Every history-based feature (form, venue
   strength, Elo and goal ratings, rest and congestion, competition context,
   head-to-head) reads from indexes fed by a `Timeline`. The timeline holds finished
   matches ordered by `available_at` = kickoff + **3 hours** (the result delay).
   `advance(C)` adds only matches with `available_at ≤ C`, so matches still being
   played at C are invisible. It can never move backwards.
2. **The target is proven unseen.** Before features are built, the timeline checks
   that the target's own result is not in the history, and raises `LeakageError` if
   it is. That can only happen with a misconfigured (zero) result delay, and a test
   shows it is caught.
3. **Cutoff ≤ kickoff is enforced.** A target whose cutoff is after its kickoff
   cannot be created.
4. **Database-backed features filter on time explicitly:**
   - odds: `recorded_at < C`;
   - absences: an injury period started before C's date and not ended by kickoff;
     a per-fixture absence list only if we ingested it before C (see below);
   - player contributions: only matches whose result was known at C.
5. **No current-state fields.** Nothing reads mutable "now" attributes (e.g.
   `seasons.is_current`, `players.current_team_id`). Season progress uses the
   season's published dates, which are fixed before the season starts.
6. **Quality-flagged matches are excluded.** Matches with open *error*-level
   data-quality issues (Sprint 3) are left out of history.
7. **Batch and single-match generation share code.** A training set (many targets,
   one timeline, advanced in cutoff order) and a single prediction (one target,
   fresh timeline) produce identical features; a test checks this.

### Tests that prove it (`tests/features/test_no_leakage.py`)

Each test computes a target's features, adds or changes something the features must
not depend on, and requires **identical** features. Paired sanity tests show the
features *do* change when legitimate past data changes, so identical results
demonstrate exclusion rather than insensitivity.

| Leakage source | Test |
| --- | --- |
| Future matches (days to months later, extreme scores) | `test_matches_after_kickoff_change_nothing` |
| A match still in progress at the cutoff | `test_match_being_played_at_the_cutoff_is_not_used` |
| Matches between an early cutoff and kickoff | `test_with_a_lead_time_matches_between_cutoff_and_kickoff_are_excluded` |
| Statistics (xG) of future matches | `test_statistics_of_future_matches_change_nothing` |
| The target's own result | `test_the_targets_own_result_changes_nothing` |
| The target's own post-match statistics | `test_the_targets_own_statistics_change_nothing` |
| Misconfiguration that would leak | `test_zero_result_delay_would_leak_and_is_refused` |
| Odds recorded at or after the cutoff | `test_odds_recorded_after_the_cutoff_are_ignored` |
| Absences starting on or after match day | `test_absences_starting_after_the_cutoff_are_ignored` |
| Absence lists ingested after the cutoff | `test_fixture_absence_lists_ingested_after_the_cutoff_are_ignored` |
| Whole dataset: append a future season, rebuild all | `test_appending_a_future_season_changes_no_existing_features` |

The tests were **mutation-checked**: breaking each guard in turn (the timeline
ignoring the cutoff, odds or absence queries ignoring time, no result delay) made
8, 1, 1 and 5 of them fail respectively. On real data (below), deleting every match
after 1 January 2024 left the features of all 576 earlier matches identical.

## Features (version `v1`)

117 features, all always present; `None` means "no data", which is different from
zero. The full list, with descriptions, is in `app/features/registry.py`, and
`python -m app.features show` prints it for a match.

| Group | Features |
| --- | --- |
| **Form** | For each team and each window of the last 3, 5 and 10 matches (any competition): matches available, goals for/against, points, win/draw/loss rates per match, xG and xGA per match (only matches with xG), and how many had xG. xGA is the opponent's xG in the same match and source |
| **Venue** | Home team's scoring and conceding *at home*, away team's *away*, over the last 10 such matches, relative to the league's home/away goal rates (1.0 = average). Shrunk by 3 pseudo-matches at the average |
| **Strength** | Elo per team (K = 20, home advantage 60, goal-margin multiplier, 20% regression to the mean at each new season); win expectancy; attack and defence ratings in log-goals, updated by the Poisson gradient; ratings-implied expected goals; matches behind each rating |
| **Context** | Competition id and cup flag; the competition's home-win and draw rates and home/away goals per match over the past year; rest days; matches in the previous 7/14/30 days; matchday; season progress; matches already played this season |
| **Head-to-head** | Up to 6 meetings within 5 years, weighted by a 1-year half-life (a 4-year-old meeting counts 1/16), with goal difference capped at ±3 per meeting: weighted goal difference and points from the home team's view, total weight, days since the last meeting |
| **Players** | Players out and doubtful per team; share of the team's season goals + assists belonging to absent players |
| **Odds** (optional) | Bookmakers with match-winner odds; opening and latest implied home/draw/away probabilities (each bookmaker's margin removed, then averaged); latest overround |

Design choices worth knowing:

- **Small samples** are handled explicitly. Every window reports how many matches it
  used, venue strengths and league baselines are shrunk toward priors, and ratings
  report their match counts.
- **Head-to-head is deliberately weak.** A first version used a 2-year half-life
  without a cap; a test showed a single 5-0 from four years earlier dragged the
  figure from +1.0 to +0.33. Hence the 1-year half-life and the ±3 cap.
- **Odds are never required.** They are None when absent, and nothing else depends
  on them.
- **Absences are conservative.** The injuries table does not record when a provider
  *published* an absence, so a per-fixture absence list counts only if we ingested
  it before the cutoff. Historical backfills therefore get no per-fixture absence
  features (train/serve difference). Set `trust_provider_absence_timing` for a
  provider known to publish before kickoff.

## Versioning and snapshots

- The version is `v1-<hash>`, where the hash covers every parameter in
  `FeatureConfig` and every feature name. Changing anything yields a new version;
  old and new snapshots coexist.
- Snapshots go to `feature_snapshots`, keyed by (match, version, cutoff), with the
  exact values in JSON. **Snapshots are immutable.** Regenerating an existing
  snapshot never overwrites it. If recomputation now differs (history was
  corrected), it is reported as drift and the stored values are kept, because
  predictions reference them.
- `reproduce(builder, snapshot)` recomputes a snapshot; tests check that every
  stored snapshot reproduces exactly.

## Commands

```bash
cd backend
python -m app.features version                       # version and parameters
python -m app.features generate                      # all matches, cutoff = kickoff
python -m app.features generate --competition-id 1 --lead-hours 24
python -m app.features show --match-id 460           # grouped values for inspection
python -m app.features export --out features.csv     # snapshots (+ results as target_ columns)
```

In the export, result columns are prefixed `target_`. They are what a model will
predict, never an input.

## Verification on real data

To exercise the features on realistic data, two real Premier League seasons
(2022/23 and 2023/24: 760 matches with results, shots, shots on target, corners,
fouls, cards and Pinnacle/Bet365 pre-match and closing odds, from
football-data.co.uk) were loaded into a throwaway database through the normal
ingestion path. This was a verification exercise only: that source is not an
integrated provider, and the database was dropped afterwards.

- 760 snapshots generated; a re-run left all 760 unchanged.
- Inspected by hand: Arsenal v Man City (8 Oct 2023) has correct rest days (8.06),
  head-to-head (2 meetings, 1-3 and 1-4, last 165 days earlier), matches played this
  season (7), season progress (0.21). The newest information used became known
  the evening before. The first match in the data has empty form, Elo 1500 and
  neutral strengths, and no invented values.
- The end-of-data Elo ranking is Man City, Arsenal, Liverpool … Luton,
  Southampton, Sheffield United.
- Signal check on 2023/24 home wins (Brier score, lower is better): constant rate
  0.248; Elo win expectancy 0.224; opening odds 0.198; closing odds 0.194. The
  expected ordering, and a sign the features carry information without having
  seen the results.

## Known limitations

- **No xG in the verification data**, and no live API. xG features were tested on
  seeded data only.
- **Competition context uses one competition id.** Form, ratings, rest and
  congestion are cross-competition, but there is no neutral-venue flag, so
  international tournaments get home advantage they may not have.
- **Promoted and new teams start at Elo 1500.** Seeding them from their previous
  division, or below the league mean, is future work.
- **Elo and goal-rating parameters are fixed**, not fitted. The models fit their own
  coefficients on top of these features, but the rating system's `elo_k`,
  `elo_home_advantage` and goal-rating constants remain hand-chosen.
- **The 3-hour result delay is a fixed assumption.** Delayed publication or matches
  longer than 3 hours (with extra time) could in principle be affected. Raise it
  for competitions with late result publication.
- **Absence timing** is conservative (see above). There is no lineup-based
  availability yet.
- **Batch generation keeps all history in memory.** That is fine for tens of
  thousands of matches; beyond that, generate per competition.
