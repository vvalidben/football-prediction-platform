# API-Football test fixtures

Recorded-style responses in the API-Football v3 format
(<https://www.api-football.com/documentation-v3>), used only by tests.

- The match data are real: Premier League 2023/24 matchday 1 results, half-time
  scores and UTC kick-off times.
- Provider identifiers (fixture ids 1035037–1035046, team ids), team codes and
  logo URLs follow the API's conventions but are **illustrative**. They have not been
  checked against the live API. Never load these files into a real database.
- `leagues.json` deliberately contains one malformed record (no league id) to
  exercise per-record rejection.
- Seasons are as they were during 2023/24, when the 2023 season was `current`.

Regenerate a file after a real API response when a key is available, keeping the
malformed record in `leagues.json`.

## Sprint 3 additions

- `teams_league_39_season_2022.json`, `fixtures_league_39_season_2022.json`:
  Premier League 2022/23 matchday 1. Results, half-time scores and UTC kick-offs
  are real; fixture ids (868000–868009) are illustrative.
- `fixture_details_league_39_season_2023.json`: the `GET /fixtures?ids=` shape for
  2023/24 matchday 1. **The statistics values are synthetic** (deterministic
  pseudo-random, generated for tests), because real per-match statistics were not
  available. Only the structure (statistic labels, `"55%"` possession strings,
  `expected_goals` as a string, nulls) follows the API documentation. Fixture
  1035046 deliberately has no statistics.
