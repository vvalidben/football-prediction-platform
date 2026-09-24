# API-Football validation (Sprint 2)

Generated 2026-09-23T21:57:31.415239+00:00 by `python scripts/validate_api_football.py`. Plan **Free**, 13 of 100 daily requests already used before this run; the validation itself spent **22** of a **25** budget.

Two kinds of statement are kept apart. A **claim** is what the provider's own `coverage` object says. An **observation** is what a real response contained. Only observations are evidence.

> Season numbering: for European leagues, season `2024` is the 2024/25 season.

## A-B. Competition and season coverage

| Competition | ID | Country | Type | Seasons | Earliest | Latest | Current | 2024? | 2025? |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Premier League | 39 | England | League | 17 | 2010 | 2026 | 2026 | yes | yes |
| La Liga | 140 | Spain | League | 17 | 2010 | 2026 | 2026 | yes | yes |
| Serie A | 135 | Italy | League | 17 | 2010 | 2026 | 2026 | yes | yes |
| Bundesliga | 78 | Germany | League | 17 | 2010 | 2026 | 2026 | yes | yes |
| Ligue 1 | 61 | France | League | 17 | 2010 | 2026 | 2026 | yes | yes |
| UEFA Champions League | 2 | World | Cup | 16 | 2011 | 2026 | 2026 | yes | yes |
| Botola Pro | 200 | Morocco | League | 11 | 2016 | 2026 | 2026 | yes | yes |
| World Cup | 1 | World | Cup | 5 | 2010 | 2026 | 2026 | no | no |

### Coverage claims for season 2024 (provider flags, not evidence)

| Competition | fixtures.events | fixtures.lineups | fixtures.statistics_fixtures | fixtures.statistics_players | injuries | odds | players | predictions | standings | top_assists | top_cards | top_scorers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Premier League | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| La Liga | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| Serie A | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| Bundesliga | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| Ligue 1 | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| UEFA Champions League | yes | yes | yes | yes | yes | no | yes | yes | yes | yes | yes | yes |
| Botola Pro | yes | yes | yes | no | no | no | yes | yes | yes | yes | yes | yes |
| World Cup | — | — | — | — | — | — | — | — | — | — | — | — |

## C-K. What a real fixture actually returned

### Premier League

Fixture `1208213` - Bournemouth v Everton, 2025-01-04, status `FT`, season 2024.

- **Score**: goals {'home': 1, 'away': 0}, half-time {'home': 0, 'away': 0}, full-time {'home': 1, 'away': 0}
- **Venue / referee**: Vitality Stadium / J. Brooks
- **Statistics** (2 team rows)
  - present: Shots on Goal, Shots off Goal, Total Shots, Blocked Shots, Shots insidebox, Shots outsidebox, Fouls, Corner Kicks, Offsides, Ball Possession, Yellow Cards, Goalkeeper Saves, Total passes, Passes accurate, Passes %, expected_goals, goals_prevented
  - null: Red Cards
  - missing: none
- **Lineups**: 2 teams, formation 4-2-3-1, XI 11, subs 9, player ids yes, positions yes, coach yes
- **Odds**: 0 rows, 0 bookmakers, 0 markets
- **Events**: 13 rows, types ['Card', 'Goal', 'subst']
- **Provider predictions**: 1 rows, fields advice, goals, percent, under_over, win_or_draw, winner, percent {'home': '45%', 'draw': '45%', 'away': '10%'}
- **Injuries (league+season)**: 3168 rows, player ids yes, fixture linked yes, example reason `Ankle Injury`
- **Player season statistics**: 20 players on page 1 of 57; fields present: appearances, minutes, rating, assists, shots_total, shots_on, passes_total, key_passes, tackles, interceptions, duels_won, yellow_cards, red_cards
- **Standings**: 1 group(s), 20 rows

### Botola Pro

Fixture `1327544` - FUS Rabat v UTS Rabat, 2024-12-28, status `FT`, season 2024.

- **Score**: goals {'home': 2, 'away': 0}, half-time {'home': 1, 'away': 0}, full-time {'home': 2, 'away': 0}
- **Venue / referee**: Stade Municipal de Kénitra / N. Nachit
- **Statistics** (2 team rows)
  - present: Yellow Cards, Red Cards
  - null: Shots on Goal, Shots off Goal, Total Shots, Blocked Shots, Shots insidebox, Shots outsidebox, Fouls, Corner Kicks, Offsides, Ball Possession, Goalkeeper Saves, Total passes, Passes accurate, Passes %, expected_goals, goals_prevented
  - missing: none
- **Lineups**: 2 teams, formation 4-4-2, XI 11, subs 9, player ids yes, positions yes, coach yes
- **Odds**: 0 rows, 0 bookmakers, 0 markets
- **Injuries (league+season)**: 0 rows - none returned
- **Player season statistics**: 20 players on page 1 of 35; fields present: appearances, minutes, goals, yellow_cards, red_cards
- **Standings**: 1 group(s), 16 rows

### UEFA Champions League

Fixture `1299032` - Barcelona v Bayern München, 2024-10-23, status `FT`, season 2024.

- **Score**: goals {'home': 4, 'away': 1}, half-time {'home': 3, 'away': 1}, full-time {'home': 4, 'away': 1}
- **Venue / referee**: Estadi Olímpic Lluís Companys / S. Vinčić
- **Statistics** (2 team rows)
  - present: Shots on Goal, Shots off Goal, Total Shots, Blocked Shots, Shots insidebox, Shots outsidebox, Fouls, Corner Kicks, Offsides, Ball Possession, Yellow Cards, Goalkeeper Saves, Total passes, Passes accurate, Passes %, expected_goals, goals_prevented
  - null: Red Cards
  - missing: none
- **Lineups**: 2 teams, formation 4-2-3-1, XI 11, subs 11, player ids yes, positions yes, coach yes
- **Odds**: 0 rows, 0 bookmakers, 0 markets

## Problems encountered

- resumed from a previous run; 18 earlier requests are listed below and were not repeated

## Requirement matrix

| Requirement | API-Football result | Evidence | ML suitability | Action |
| --- | --- | --- | --- | --- |
| Target competitions exist with stable ids | All eight found | `/leagues` returned each id and the name matched | **READY** | Use the ids recorded in this artefact; never hard-code elsewhere. |
| Fixtures, kickoff times, final and half-time scores | Complete for every competition probed | Premier League fixture 1208213 and Botola fixture 1327544 carried goals, half-time and full-time | **READY** | This is the spine of the feature set; already ingested this way. |
| Team match statistics for the big five and UEFA | 17 of 18 wanted fields populated | Observed on Premier League and Champions League fixtures | **READY** | Ingest per match; treat strictly as history behind the result delay. |
| Team match statistics for Botola | only 2 of 18 populated (cards only) | Every shot, possession, corner and pass field returned `null` | **NOT AVAILABLE** | Botola models must run on results and ratings only, or take a second provider. |
| Expected goals (xG) | Present for the big five and UEFA, absent for Botola | `expected_goals` in `/fixtures/statistics`; null in Botola | **PARTIAL** | Use where present; the feature set already treats xG as optional. |
| Lineups | Complete where probed | XI 11, formation 4-2-3-1, player ids and positions present, also for Botola | **PARTIAL** | Rich, but no publication timestamp: snapshot ourselves before kickoff. |
| Player season statistics | 13 fields for the Premier League, 5 for Botola | `/players` page 1; Botola lacks shots, passes, tackles and ratings | **PARTIAL** | Recompute point-in-time from match data; do not store season aggregates. |
| Injuries, historically | 3168 rows for Premier League 2024, 0 for Botola | `/injuries?league=&season=` answers by season and links a fixture id | **NEEDS FURTHER VALIDATION** | No `reported_at`: unusable for backtesting until we snapshot it ourselves. |
| Odds, historically | none returned (0 rows for a completed fixture) | `/odds?fixture=` empty, and every season-2024 coverage flag says `odds: no` | **NOT AVAILABLE** | Drop odds features, or take a second provider for them. |
| Standings | Available | `/standings` returned a full table for both leagues probed | **PARTIAL** | Recompute from our own matches; the provider's table is as-of-request. |
| Provider predictions | Available with percentages and advice | `/predictions?fixture=` returned one row with `percent` | **PARTIAL** | Optional reference only — never a target, never a feature without a snapshot time. |
| Historical depth for backtesting | three seasons on this plan (2022-2024) | Season 2021 refused: `Free plans do not have access to this season` | **PARTIAL** | Enough to train and validate; a paid plan is required for real depth. |
| Current and upcoming fixtures | not reachable on this plan | Seasons 2025 and 2026 are listed but outside the Free window | **NOT AVAILABLE** | Live prediction needs a paid plan; today the platform can only re-predict history. |

## M. Historical depth

The plan, not the catalogue, is the binding constraint. `/leagues` lists seasons back to 2010, but this Free plan answers only **2022-2024** (`Free plans do not have access to this season, try from 2022 to 2024`), and rejects the `last`, `next` and `date` narrowing parameters outright.

| Competition | Seasons listed | Seasons reachable on this plan | Fixtures | Team stats | Players | Injuries | Odds | xG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Premier League | 2010-2026 | 2022-2024 | 380 in 2024 | 17/18 fields | 13 fields | 3168 rows | 0 rows | present |
| La Liga | 2010-2026 | 2022-2024 (plan) | not probed | not probed | not probed | not probed | not probed | not probed |
| Serie A | 2010-2026 | 2022-2024 (plan) | not probed | not probed | not probed | not probed | not probed | not probed |
| Bundesliga | 2010-2026 | 2022-2024 (plan) | not probed | not probed | not probed | not probed | not probed | not probed |
| Ligue 1 | 2010-2026 | 2022-2024 (plan) | not probed | not probed | not probed | not probed | not probed | not probed |
| UEFA Champions League | 2011-2026 | 2022-2024 | 269 in 2024 | 17/18 fields | 0 fields | — rows | 0 rows | present |
| Botola Pro | 2016-2026 | 2022-2024 | 244 in 2024 | 2/18 fields | 5 fields | 0 rows | 0 rows | null |
| World Cup | 2010-2026 | 2022-2024 (plan) | not probed | not probed | not probed | not probed | not probed | not probed |

## L. Data timing and leakage

| Data source | Pre-match usable? | Timing evidence | Leakage risk | Notes |
| --- | --- | --- | --- | --- |
| Fixture list, kickoff time, venue, teams | SAFE PRE-MATCH | Published when the fixture is scheduled; observed on finished fixtures with a `date` | None | The spine of any feature set. |
| Final score, half-time score | POST-MATCH ONLY | Only populated once played; `status` says FT | Severe if used for its own match | Safe only as history for *earlier* matches, behind the result delay. |
| Team match statistics (shots, possession, corners, cards, saves) | POST-MATCH ONLY | Returned by `/fixtures/statistics` for a finished fixture; no pre-match variant exists | Severe if used for its own match | Usable as history for earlier matches only. |
| expected_goals (xG) in fixture statistics | POST-MATCH ONLY | Appears in the same `/fixtures/statistics` payload as the final numbers | Severe if used for its own match | Same rule as any other match statistic. |
| Events (goals, cards, substitutions) | POST-MATCH ONLY | Minute-stamped in-play events | Severe | History only. |
| Lineups (XI, formation, coach) | TIMING-DEPENDENT | Published around an hour before kickoff, but the payload carries no publication timestamp; the response for a finished fixture is indistinguishable from the pre-match one | High | Cannot be used for backtesting without recording our own fetch time. Safe only for live prediction if we snapshot it ourselves before kickoff. |
| Injuries | TIMING-DEPENDENT | Queryable by league+season and linked to a fixture id; no `reported_at` field was observed, so when a row became true is unknown | High | Needs our own snapshotting to be backtest-safe. |
| Standings | TIMING-DEPENDENT | Reflects matches played to date; a request made after a round already includes it | High | Recompute from our own match history instead of storing the provider's table. |
| Player season statistics | TIMING-DEPENDENT | Season aggregates that include the match in question once it is played | High | Must be recomputed point-in-time from match-level data. |
| Odds | SAFE PRE-MATCH (when retained) | Pre-kickoff by nature; but this plan returned none for completed fixtures | None to the model, but unavailable here | See the odds limitation below. |
| Provider predictions (`/predictions`) | TIMING-DEPENDENT | Available for a fixture; the payload gives no generation timestamp, and the provider states it uses the teams' current form | High | Never a target, at most a reference feature, and only with our own snapshot time. |

The decisive point: **no endpoint in this API returns a publication timestamp**. For anything that changes as a match approaches — lineups, injuries, standings, provider predictions — the API cannot tell us what was true before kickoff. Those sources are therefore unusable for backtesting unless this platform snapshots them itself, at a recorded time, going forward. That is a property of the provider, not of our pipeline.

## Requests spent

| # | Endpoint | Parameters | Why it was needed | Result |
| ---: | --- | --- | --- | --- |
| 1 | `/status` | - | plan, daily quota and rate-limit headers | 0 results |
| 2 | `/leagues` | - | league ids, seasons and per-season coverage for every target | 1246 results |
| 3 | `/fixtures` | league=39, season=2024, status=FT | Premier League: reach one completed fixture from season 2024 | 380 results |
| 4 | `/fixtures/statistics` | fixture=1208213 | Premier League: which team statistics (including xG) a real fixture carries | 2 results |
| 5 | `/fixtures/lineups` | fixture=1208213 | Premier League: lineup shape — XI, formation, player ids, coach | 2 results |
| 6 | `/odds` | fixture=1208213 | Premier League: are odds retained for a completed fixture (historical depth) | 0 results |
| 7 | `/fixtures/events` | fixture=1208213 | Premier League: event feed shape (goals, cards, substitutions) | 13 results |
| 8 | `/predictions` | fixture=1208213 | Premier League: does the provider's own prediction exist, and what is in it | 1 results |
| 9 | `/injuries` | league=39, season=2024 | Premier League: can injuries be retrieved historically by league+season | 3168 results |
| 10 | `/players` | league=39, season=2024, page=1 | Premier League: player season statistics, page 1 only (no pagination) | 20 results |
| 11 | `/standings` | league=39, season=2024 | Premier League: standings availability | 1 results |
| 12 | `/fixtures` | league=200, season=2024, status=FT | Botola Pro: reach one completed fixture from season 2024 | 244 results |
| 13 | `/fixtures/statistics` | fixture=1327544 | Botola Pro: which team statistics (including xG) a real fixture carries | 2 results |
| 14 | `/fixtures/lineups` | fixture=1327544 | Botola Pro: lineup shape — XI, formation, player ids, coach | 2 results |
| 15 | `/odds` | fixture=1327544 | Botola Pro: are odds retained for a completed fixture (historical depth) | 0 results |
| 16 | `/injuries` | league=200, season=2024 | Botola Pro: can injuries be retrieved historically by league+season | 0 results |
| 17 | `/players` | league=200, season=2024, page=1 | Botola Pro: player season statistics, page 1 only (no pagination) | 20 results |
| 18 | `/standings` | league=200, season=2024 | Botola Pro: standings availability | 1 results |
| 1 | `/fixtures` | league=2, season=2024, status=FT | UEFA Champions League: reach one completed fixture from season 2024 | 269 results |
| 2 | `/fixtures/statistics` | fixture=1299032 | UEFA Champions League: which team statistics (including xG) a real fixture carries | 2 results |
| 3 | `/fixtures/lineups` | fixture=1299032 | UEFA Champions League: lineup shape — XI, formation, player ids, coach | 2 results |
| 4 | `/odds` | fixture=1299032 | UEFA Champions League: are odds retained for a completed fixture (historical depth) | 0 results |

Daily requests remaining after this run, from the `x-ratelimit-requests-remaining` header: **66**.
