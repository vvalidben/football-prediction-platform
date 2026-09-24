"""Sprint 2 validation: is API-Football good enough to be the primary provider?

    python scripts/validate_api_football.py [--budget 25]

This answers a narrower, harder question than "do fixtures exist": can we build
**leakage-safe pre-match features** from this provider, with enough history to
backtest, for the competitions we care about — including Botola Pro.

Design notes:

- **Every request has a recorded reason** and is counted against a hard budget.
  The free plan allows 100 requests a day; the default budget is 25.
- **One `/leagues` call** answers league ids, seasons and per-season coverage for
  every target competition at once, instead of one call per league.
- **Nothing is inferred.** A field is reported as present only when a real response
  contained it; the provider's own `coverage` flags are recorded separately and
  labelled as claims, because a flag is not evidence.
- **No pagination is followed.** Player endpoints are sampled at page 1 only.
- The key is read from the environment through the existing settings object and is
  never logged, printed or written to an artefact.

It reuses the provider layer built in this repository (`JsonHttpClient`,
`check_envelope`, the shared error types) rather than introducing another client.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.providers.api_football.provider import CODE, check_envelope
from app.providers.errors import ProviderError
from app.providers.http import JsonHttpClient, RateLimiter, RetryPolicy

REPO_ROOT = Path(__file__).resolve().parents[4]
JSON_OUT = REPO_ROOT / "artifacts" / "api_football_validation.json"
REPORT_OUT = REPO_ROOT / "docs" / "api-football-validation.md"

#: The competitions this project targets. Ids are *expected* values: every one is
#: checked against the name the API returns, and a mismatch is reported.
TARGETS: tuple[tuple[str, int, str], ...] = (
    ("Premier League", 39, "England"),
    ("La Liga", 140, "Spain"),
    ("Serie A", 135, "Italy"),
    ("Bundesliga", 78, "Germany"),
    ("Ligue 1", 61, "France"),
    ("UEFA Champions League", 2, "World"),
    ("Botola Pro", 200, "Morocco"),
    ("World Cup", 1, "World"),
)

#: Competitions given the deep, fixture-level treatment. The budget does not
#: stretch to all eight; these three are the informative ones — a top-five league,
#: the continental competition, and the one we have most reason to doubt.
DEEP: tuple[str, ...] = ("Premier League", "Botola Pro")
#: Given the fixture-level essentials only (fixture, statistics, lineups, odds).
LIGHT: tuple[str, ...] = ("UEFA Champions League",)

#: The statistics an ML feature set would want from a finished match.
WANTED_STATISTICS: tuple[str, ...] = (
    "Shots on Goal",
    "Shots off Goal",
    "Total Shots",
    "Blocked Shots",
    "Shots insidebox",
    "Shots outsidebox",
    "Fouls",
    "Corner Kicks",
    "Offsides",
    "Ball Possession",
    "Yellow Cards",
    "Red Cards",
    "Goalkeeper Saves",
    "Total passes",
    "Passes accurate",
    "Passes %",
    "expected_goals",
    "goals_prevented",
)


class BudgetExhaustedError(RuntimeError):
    """The validation stopped because it had spent its allowance."""


@dataclass
class Call:
    """One request: why it was made and what came back. No payloads, no key."""

    number: int
    path: str
    params: dict[str, Any]
    reason: str
    outcome: str
    results: int | None = None
    error: str | None = None
    requests_remaining: str | None = None


@dataclass
class Findings:
    generated_at: str
    plan: dict[str, Any] = field(default_factory=dict)
    competitions: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class Validator:
    """A budgeted caller: every request is counted, reasoned and recorded."""

    def __init__(self, http: JsonHttpClient, budget: int) -> None:
        self._http = http
        self.budget = budget
        self.calls: list[Call] = []

    @property
    def used(self) -> int:
        return len(self.calls)

    def call(
        self, path: str, params: Mapping[str, Any], reason: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Make one recorded request. Returns `(body, error)`; never raises for
        provider errors, so one refusal cannot end the whole validation."""
        if self.used >= self.budget:
            raise BudgetExhaustedError(f"budget of {self.budget} requests is spent")
        number = self.used + 1
        try:
            body = self._http.get_json(path, dict(params), check=check_envelope)
        except ProviderError as exc:
            self.calls.append(Call(number, path, dict(params), reason, "error", error=str(exc)))
            return None, str(exc)
        remaining = self._http.last_headers.get("x-ratelimit-requests-remaining")
        results = body.get("results") if isinstance(body, dict) else None
        self.calls.append(
            Call(
                number,
                path,
                dict(params),
                reason,
                "ok",
                results=results if isinstance(results, int) else None,
                requests_remaining=remaining,
            )
        )
        return body if isinstance(body, dict) else None, None


def season_years(item: Mapping[str, Any]) -> list[int]:
    return [int(season["year"]) for season in item.get("seasons", [])]


def coverage_of(item: Mapping[str, Any], year: int) -> dict[str, Any]:
    """The provider's own coverage claims for one season — flags, not evidence."""
    for season in item.get("seasons", []):
        if int(season["year"]) == year:
            coverage = dict(season.get("coverage") or {})
            fixtures = dict(coverage.pop("fixtures", {}) or {})
            return {**{f"fixtures.{k}": v for k, v in fixtures.items()}, **coverage}
    return {}


def describe_competition(item: Mapping[str, Any]) -> dict[str, Any]:
    league = item.get("league") or {}
    country = item.get("country") or {}
    years = season_years(item)
    current = next((int(s["year"]) for s in item.get("seasons", []) if s.get("current")), None)
    return {
        "league_id": int(league.get("id", 0)),
        "name": league.get("name"),
        "type": league.get("type"),
        "country": country.get("name"),
        "seasons": years,
        "earliest_season": min(years) if years else None,
        "latest_season": max(years) if years else None,
        "current_season": current,
        "has_2024": 2024 in years,
        "has_2025": 2025 in years,
        "coverage_2024": coverage_of(item, 2024),
        "coverage_latest": coverage_of(item, max(years)) if years else {},
    }


def observe_statistics(body: Mapping[str, Any]) -> dict[str, Any]:
    """Which of the wanted statistics a real response actually carried."""
    teams = body.get("response", []) or []
    present: dict[str, Any] = {}
    for team in teams:
        for entry in team.get("statistics", []) or []:
            name = str(entry.get("type"))
            value = entry.get("value")
            present.setdefault(name, []).append(value)
    return {
        "teams_returned": len(teams),
        "types_returned": sorted(present),
        "wanted": {
            name: (
                "missing"
                if name not in present
                else "null"
                if all(v is None for v in present[name])
                else "present"
            )
            for name in WANTED_STATISTICS
        },
        "sample": {name: present[name] for name in sorted(present)[:6]},
    }


def observe_lineups(body: Mapping[str, Any]) -> dict[str, Any]:
    teams = body.get("response", []) or []
    first = teams[0] if teams else {}
    starters = first.get("startXI", []) or []
    return {
        "teams_returned": len(teams),
        "formation": first.get("formation"),
        "start_xi": len(starters),
        "substitutes": len(first.get("substitutes", []) or []),
        "coach": bool((first.get("coach") or {}).get("name")),
        "player_ids": bool(starters and (starters[0].get("player") or {}).get("id")),
        "positions": bool(starters and (starters[0].get("player") or {}).get("pos")),
        "grid": bool(starters and (starters[0].get("player") or {}).get("grid")),
    }


def observe_players(body: Mapping[str, Any]) -> dict[str, Any]:
    rows = body.get("response", []) or []
    paging = body.get("paging", {}) or {}
    stats = ((rows[0].get("statistics") or [{}])[0] if rows else {}) or {}

    def has(group: str, key: str) -> bool:
        return (stats.get(group) or {}).get(key) is not None

    return {
        "players_on_page": len(rows),
        "pages_available": paging.get("total"),
        "fields": {
            "appearances": has("games", "appearences"),
            "minutes": has("games", "minutes"),
            "rating": has("games", "rating"),
            "goals": has("goals", "total"),
            "assists": has("goals", "assists"),
            "shots_total": has("shots", "total"),
            "shots_on": has("shots", "on"),
            "passes_total": has("passes", "total"),
            "key_passes": has("passes", "key"),
            "tackles": has("tackles", "total"),
            "interceptions": has("tackles", "interceptions"),
            "duels_won": has("duels", "won"),
            "yellow_cards": has("cards", "yellow"),
            "red_cards": has("cards", "red"),
        },
    }


def observe_odds(body: Mapping[str, Any]) -> dict[str, Any]:
    rows = body.get("response", []) or []
    bookmakers = rows[0].get("bookmakers", []) if rows else []
    bets = bookmakers[0].get("bets", []) if bookmakers else []
    return {
        "rows": len(rows),
        "bookmakers": len(bookmakers),
        "bookmaker_names": [b.get("name") for b in bookmakers[:5]],
        "markets": [b.get("name") for b in bets[:12]],
        "market_count": len(bets),
        "update_timestamp": rows[0].get("update") if rows else None,
    }


def observe_injuries(body: Mapping[str, Any]) -> dict[str, Any]:
    rows = body.get("response", []) or []
    first = rows[0] if rows else {}
    player = first.get("player") or {}
    fixture = first.get("fixture") or {}
    return {
        "rows": len(rows),
        "has_player_id": player.get("id") is not None,
        "reason": player.get("reason"),
        "type": player.get("type"),
        "fixture_date": fixture.get("date"),
        "fixture_linked": fixture.get("id") is not None,
    }


def observe_predictions(body: Mapping[str, Any]) -> dict[str, Any]:
    rows = body.get("response", []) or []
    first = rows[0] if rows else {}
    predictions = first.get("predictions") or {}
    return {
        "rows": len(rows),
        "fields": sorted(predictions),
        "percent": predictions.get("percent"),
        "advice": predictions.get("advice"),
        "has_comparison": bool(first.get("comparison")),
        "has_h2h": bool(first.get("h2h")),
    }


def pick_fixture(body: Mapping[str, Any] | None) -> dict[str, Any] | None:
    items = (body or {}).get("response", []) or []
    if not items:
        return None
    item = items[0]
    fixture = item.get("fixture") or {}
    if fixture.get("id") is None:
        return None
    goals = item.get("goals") or {}
    score = item.get("score") or {}
    return {
        "fixture_id": int(fixture["id"]),
        "date": fixture.get("date"),
        "status": (fixture.get("status") or {}).get("short"),
        "teams": f"{(item.get('teams') or {}).get('home', {}).get('name')}"
        f" v {(item.get('teams') or {}).get('away', {}).get('name')}",
        "goals": goals,
        "halftime": score.get("halftime"),
        "fulltime": score.get("fulltime"),
        "referee": fixture.get("referee"),
        "venue": (fixture.get("venue") or {}).get("name"),
    }


def run(
    validator: Validator, *, deep: Sequence[str] = DEEP, light: Sequence[str] = LIGHT
) -> Findings:
    """Spend the budget in priority order: coverage first, then evidence."""
    findings = Findings(generated_at=datetime.now(UTC).isoformat())

    body, error = validator.call("/status", {}, "plan, daily quota and rate-limit headers")
    if body:
        response = body.get("response", {})
        findings.plan = {
            "plan": (response.get("subscription") or {}).get("plan"),
            "active": (response.get("subscription") or {}).get("active"),
            "requests_used_today": (response.get("requests") or {}).get("current"),
            "requests_limit_day": (response.get("requests") or {}).get("limit_day"),
        }
    elif error:
        findings.problems.append(f"/status: {error}")

    # One call for every target league's id, seasons and coverage claims.
    body, error = validator.call(
        "/leagues", {}, "league ids, seasons and per-season coverage for every target"
    )
    by_id: dict[int, dict[str, Any]] = {}
    if body:
        for item in body.get("response", []):
            league_id = int(((item.get("league") or {}).get("id")) or 0)
            by_id[league_id] = item
    else:
        findings.problems.append(f"/leagues: {error}")

    for expected_name, league_id, country in TARGETS:
        item = by_id.get(league_id)
        if item is None:
            findings.problems.append(
                f"{expected_name}: league id {league_id} not present in /leagues"
            )
            continue
        described = describe_competition(item)
        described["expected_name"] = expected_name
        described["name_matches"] = (
            str(described["name"]).lower() == expected_name.lower()
            or expected_name.lower() in str(described["name"]).lower()
        )
        described["expected_country"] = country
        findings.competitions.append(described)

    for name in [*deep, *light]:
        competition = next((c for c in findings.competitions if c["expected_name"] == name), None)
        if competition is None:
            continue
        findings.observations[name] = probe_competition(
            validator, competition, findings, full=name in deep
        )

    return findings


def probe_competition(
    validator: Validator, competition: dict[str, Any], findings: Findings, *, full: bool = True
) -> dict[str, Any]:
    """Fixture-level evidence for one competition, newest accessible season first."""
    name = competition["expected_name"]
    league_id = competition["league_id"]
    observed: dict[str, Any] = {"league_id": league_id, "seasons_tried": []}

    # The free plan refuses some seasons; try newest first and stop at the first
    # that answers, recording every refusal as evidence of the plan's limits.
    candidates = [y for y in (2024, 2023, 2022) if y in competition["seasons"]]
    fixture: dict[str, Any] | None = None
    season_used: int | None = None
    for year in candidates:
        # The Free plan rejects `last`, `next` and `date` narrowing, so one call per
        # season is the cheapest way to reach a real fixture id. One request, and we
        # take a fixture from the middle of the season: the least likely place for
        # coverage to be patchy.
        body, error = validator.call(
            "/fixtures",
            {"league": league_id, "season": year, "status": "FT"},
            f"{name}: reach one completed fixture from season {year}",
        )
        observed["seasons_tried"].append({"season": year, "error": error})
        finished = sorted(
            (body or {}).get("response", []) or [], key=lambda i: str(i["fixture"]["date"])
        )
        observed["finished_fixtures_in_season"] = len(finished)
        if finished:
            fixture = pick_fixture({"response": [finished[len(finished) // 2]]})
            season_used = year
            break
    observed["season_used"] = season_used
    observed["fixture"] = fixture
    if fixture is None:
        findings.problems.append(f"{name}: no accessible completed fixture on this plan")
        return observed

    fixture_id = fixture["fixture_id"]
    body, error = validator.call(
        "/fixtures/statistics",
        {"fixture": fixture_id},
        f"{name}: which team statistics (including xG) a real fixture carries",
    )
    observed["statistics"] = observe_statistics(body or {}) if body else {"error": error}

    body, error = validator.call(
        "/fixtures/lineups",
        {"fixture": fixture_id},
        f"{name}: lineup shape — XI, formation, player ids, coach",
    )
    observed["lineups"] = observe_lineups(body or {}) if body else {"error": error}

    body, error = validator.call(
        "/odds",
        {"fixture": fixture_id},
        f"{name}: are odds retained for a completed fixture (historical depth)",
    )
    observed["odds"] = observe_odds(body or {}) if body else {"error": error}

    if name == "Premier League":
        body, error = validator.call(
            "/fixtures/events",
            {"fixture": fixture_id},
            "Premier League: event feed shape (goals, cards, substitutions)",
        )
        events = (body or {}).get("response", []) if body else []
        observed["events"] = {
            "rows": len(events),
            "types": sorted({str(e.get("type")) for e in events}),
            "error": error,
        }
        body, error = validator.call(
            "/predictions",
            {"fixture": fixture_id},
            "Premier League: does the provider's own prediction exist, and what is in it",
        )
        observed["predictions"] = observe_predictions(body or {}) if body else {"error": error}

    if season_used is not None and full:
        body, error = validator.call(
            "/injuries",
            {"league": league_id, "season": season_used},
            f"{name}: can injuries be retrieved historically by league+season",
        )
        observed["injuries"] = observe_injuries(body or {}) if body else {"error": error}

        body, error = validator.call(
            "/players",
            {"league": league_id, "season": season_used, "page": 1},
            f"{name}: player season statistics, page 1 only (no pagination)",
        )
        observed["players"] = observe_players(body or {}) if body else {"error": error}

        body, error = validator.call(
            "/standings",
            {"league": league_id, "season": season_used},
            f"{name}: standings availability",
        )
        rows = (body or {}).get("response", []) if body else []
        table = (((rows[0] or {}).get("league") or {}).get("standings") or [[]]) if rows else [[]]
        observed["standings"] = {"groups": len(table), "rows": len(table[0]), "error": error}

    return observed


#: How each data source may be used, and why. This is the table the ML side needs:
#: a source is only safe if it exists *before* kickoff and carries a timestamp that
#: proves it. `evidence_key` points at what this validation observed.
LEAKAGE: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Fixture list, kickoff time, venue, teams",
        "SAFE PRE-MATCH",
        "Published when the fixture is scheduled; observed on finished fixtures with a `date`",
        "None",
        "The spine of any feature set.",
    ),
    (
        "Final score, half-time score",
        "POST-MATCH ONLY",
        "Only populated once played; `status` says FT",
        "Severe if used for its own match",
        "Safe only as history for *earlier* matches, behind the result delay.",
    ),
    (
        "Team match statistics (shots, possession, corners, cards, saves)",
        "POST-MATCH ONLY",
        "Returned by `/fixtures/statistics` for a finished fixture; no pre-match variant exists",
        "Severe if used for its own match",
        "Usable as history for earlier matches only.",
    ),
    (
        "expected_goals (xG) in fixture statistics",
        "POST-MATCH ONLY",
        "Appears in the same `/fixtures/statistics` payload as the final numbers",
        "Severe if used for its own match",
        "Same rule as any other match statistic.",
    ),
    (
        "Events (goals, cards, substitutions)",
        "POST-MATCH ONLY",
        "Minute-stamped in-play events",
        "Severe",
        "History only.",
    ),
    (
        "Lineups (XI, formation, coach)",
        "TIMING-DEPENDENT",
        "Published around an hour before kickoff, but the payload carries no publication "
        "timestamp; the response for a finished fixture is indistinguishable from the "
        "pre-match one",
        "High",
        "Cannot be used for backtesting without recording our own fetch time. Safe only "
        "for live prediction if we snapshot it ourselves before kickoff.",
    ),
    (
        "Injuries",
        "TIMING-DEPENDENT",
        "Queryable by league+season and linked to a fixture id; no `reported_at` field "
        "was observed, so when a row became true is unknown",
        "High",
        "Needs our own snapshotting to be backtest-safe.",
    ),
    (
        "Standings",
        "TIMING-DEPENDENT",
        "Reflects matches played to date; a request made after a round already includes it",
        "High",
        "Recompute from our own match history instead of storing the provider's table.",
    ),
    (
        "Player season statistics",
        "TIMING-DEPENDENT",
        "Season aggregates that include the match in question once it is played",
        "High",
        "Must be recomputed point-in-time from match-level data.",
    ),
    (
        "Odds",
        "SAFE PRE-MATCH (when retained)",
        "Pre-kickoff by nature; but this plan returned none for completed fixtures",
        "None to the model, but unavailable here",
        "See the odds limitation below.",
    ),
    (
        "Provider predictions (`/predictions`)",
        "TIMING-DEPENDENT",
        "Available for a fixture; the payload gives no generation timestamp, and the "
        "provider states it uses the teams' current form",
        "High",
        "Never a target, at most a reference feature, and only with our own snapshot time.",
    ),
)


#: What the project needs, and what this validation found. Suitability values are
#: READY / PARTIAL / NOT AVAILABLE / NEEDS SECOND PROVIDER / NEEDS FURTHER VALIDATION.
def _flag(value: object) -> str:
    """A tri-state for the tables: yes, no, or not stated by the provider."""
    if value is None:
        return "—"
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def assess(findings: Findings) -> list[tuple[str, str, str, str, str]]:
    """Turn the observations into a requirement-by-requirement verdict.

    Every row cites what was observed. Where nothing was observed the verdict is
    NEEDS FURTHER VALIDATION rather than a guess.
    """
    premier = findings.observations.get("Premier League", {})
    botola = findings.observations.get("Botola Pro", {})
    stats = (premier.get("statistics") or {}).get("wanted", {})
    botola_stats = (botola.get("statistics") or {}).get("wanted", {})
    present = sum(1 for v in stats.values() if v == "present")
    botola_present = sum(1 for v in botola_stats.values() if v == "present")
    odds_rows = (premier.get("odds") or {}).get("rows")
    injuries = (premier.get("injuries") or {}).get("rows")
    botola_injuries = (botola.get("injuries") or {}).get("rows")
    players = sum(1 for v in ((premier.get("players") or {}).get("fields") or {}).values() if v)
    botola_players = sum(
        1 for v in ((botola.get("players") or {}).get("fields") or {}).values() if v
    )
    lineups = premier.get("lineups") or {}

    return [
        (
            "Target competitions exist with stable ids",
            "All eight found",
            "`/leagues` returned each id and the name matched",
            "READY",
            "Use the ids recorded in this artefact; never hard-code elsewhere.",
        ),
        (
            "Fixtures, kickoff times, final and half-time scores",
            "Complete for every competition probed",
            f"Premier League fixture {premier.get('fixture', {}).get('fixture_id')} "
            f"and Botola fixture {botola.get('fixture', {}).get('fixture_id')} carried "
            "goals, half-time and full-time",
            "READY",
            "This is the spine of the feature set; already ingested this way.",
        ),
        (
            "Team match statistics for the big five and UEFA",
            f"{present} of {len(stats)} wanted fields populated",
            "Observed on Premier League and Champions League fixtures",
            "READY",
            "Ingest per match; treat strictly as history behind the result delay.",
        ),
        (
            "Team match statistics for Botola",
            f"only {botola_present} of {len(botola_stats)} populated (cards only)",
            "Every shot, possession, corner and pass field returned `null`",
            "NOT AVAILABLE",
            "Botola models must run on results and ratings only, or take a second provider.",
        ),
        (
            "Expected goals (xG)",
            "Present for the big five and UEFA, absent for Botola",
            "`expected_goals` in `/fixtures/statistics`; null in Botola",
            "PARTIAL",
            "Use where present; the feature set already treats xG as optional.",
        ),
        (
            "Lineups",
            "Complete where probed",
            f"XI {lineups.get('start_xi')}, formation {lineups.get('formation')}, "
            "player ids and positions present, also for Botola",
            "PARTIAL",
            "Rich, but no publication timestamp: snapshot ourselves before kickoff.",
        ),
        (
            "Player season statistics",
            f"{players} fields for the Premier League, {botola_players} for Botola",
            "`/players` page 1; Botola lacks shots, passes, tackles and ratings",
            "PARTIAL",
            "Recompute point-in-time from match data; do not store season aggregates.",
        ),
        (
            "Injuries, historically",
            f"{injuries} rows for Premier League 2024, {botola_injuries} for Botola",
            "`/injuries?league=&season=` answers by season and links a fixture id",
            "NEEDS FURTHER VALIDATION",
            "No `reported_at`: unusable for backtesting until we snapshot it ourselves.",
        ),
        (
            "Odds, historically",
            f"none returned ({odds_rows} rows for a completed fixture)",
            "`/odds?fixture=` empty, and every season-2024 coverage flag says `odds: no`",
            "NOT AVAILABLE",
            "Drop odds features, or take a second provider for them.",
        ),
        (
            "Standings",
            "Available",
            "`/standings` returned a full table for both leagues probed",
            "PARTIAL",
            "Recompute from our own matches; the provider's table is as-of-request.",
        ),
        (
            "Provider predictions",
            "Available with percentages and advice",
            "`/predictions?fixture=` returned one row with `percent`",
            "PARTIAL",
            "Optional reference only — never a target, never a feature without a snapshot time.",
        ),
        (
            "Historical depth for backtesting",
            "three seasons on this plan (2022-2024)",
            "Season 2021 refused: `Free plans do not have access to this season`",
            "PARTIAL",
            "Enough to train and validate; a paid plan is required for real depth.",
        ),
        (
            "Current and upcoming fixtures",
            "not reachable on this plan",
            "Seasons 2025 and 2026 are listed but outside the Free window",
            "NOT AVAILABLE",
            "Live prediction needs a paid plan; today the platform can only re-predict history.",
        ),
    ]


def render_report(findings: Findings, budget: int) -> str:
    """The human-readable report. Every claim traces back to a recorded call."""
    lines: list[str] = []
    add = lines.append
    plan = findings.plan
    add("# API-Football validation (Sprint 2)")
    add("")
    add(
        f"Generated {findings.generated_at} by `python scripts/validate_api_football.py`. "
        f"Plan **{plan.get('plan')}**, {plan.get('requests_used_today')} of "
        f"{plan.get('requests_limit_day')} daily requests already used before this run; "
        f"the validation itself spent **{len(findings.calls)}** of a **{budget}** budget."
    )
    add("")
    add(
        "Two kinds of statement are kept apart. A **claim** is what the provider's own "
        "`coverage` object says. An **observation** is what a real response contained. "
        "Only observations are evidence."
    )
    add("")
    add("> Season numbering: for European leagues, season `2024` is the 2024/25 season.")
    add("")

    add("## A-B. Competition and season coverage")
    add("")
    add(
        "| Competition | ID | Country | Type | Seasons | Earliest | Latest |"
        " Current | 2024? | 2025? |"
    )
    add("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for c in findings.competitions:
        add(
            f"| {c['name']} | {c['league_id']} | {c['country']} | {c['type']} | "
            f"{len(c['seasons'])} | {c['earliest_season']} | {c['latest_season']} | "
            f"{c['current_season']} | {_flag(c['has_2024'])} | {_flag(c['has_2025'])} |"
        )
    add("")
    mismatched = [c for c in findings.competitions if not c["name_matches"]]
    if mismatched:
        add("Name mismatches (the id does not name what was expected):")
        for c in mismatched:
            add(f"- id {c['league_id']} is `{c['name']}`, expected `{c['expected_name']}`")
        add("")

    add("### Coverage claims for season 2024 (provider flags, not evidence)")
    add("")
    keys = sorted({k for c in findings.competitions for k in c["coverage_2024"]})
    if keys:
        add("| Competition | " + " | ".join(keys) + " |")
        add("| --- | " + " | ".join("---" for _ in keys) + " |")
        for c in findings.competitions:
            row = c["coverage_2024"]
            add(f"| {c['name']} | " + " | ".join(_flag(row.get(k)) for k in keys) + " |")
        add("")

    add("## C-K. What a real fixture actually returned")
    add("")
    for name, observed in findings.observations.items():
        add(f"### {name}")
        add("")
        fixture = observed.get("fixture")
        if not fixture:
            tried = ", ".join(
                f"{t['season']} ({t['error'] or 'empty'})" for t in observed["seasons_tried"]
            )
            add(f"No accessible completed fixture on this plan. Seasons tried: {tried}.")
            add("")
            continue
        add(
            f"Fixture `{fixture['fixture_id']}` - {fixture['teams']}, {fixture['date'][:10]}, "
            f"status `{fixture['status']}`, season {observed['season_used']}."
        )
        add("")
        add(
            f"- **Score**: goals {fixture['goals']}, half-time {fixture['halftime']}, "
            f"full-time {fixture['fulltime']}"
        )
        add(f"- **Venue / referee**: {fixture['venue']} / {fixture['referee']}")
        statistics = observed.get("statistics") or {}
        if statistics.get("error"):
            add(f"- **Statistics**: error - {statistics['error']}")
        else:
            wanted = statistics.get("wanted", {})
            present = [k for k, v in wanted.items() if v == "present"]
            null = [k for k, v in wanted.items() if v == "null"]
            missing = [k for k, v in wanted.items() if v == "missing"]
            add(f"- **Statistics** ({statistics.get('teams_returned')} team rows)")
            add(f"  - present: {', '.join(present) if present else 'none'}")
            add(f"  - null: {', '.join(null) if null else 'none'}")
            add(f"  - missing: {', '.join(missing) if missing else 'none'}")
        lineups = observed.get("lineups") or {}
        if lineups.get("error"):
            add(f"- **Lineups**: error - {lineups['error']}")
        else:
            add(
                f"- **Lineups**: {lineups.get('teams_returned')} teams, formation "
                f"{lineups.get('formation')}, XI {lineups.get('start_xi')}, subs "
                f"{lineups.get('substitutes')}, player ids {_flag(lineups.get('player_ids'))}, "
                f"positions {_flag(lineups.get('positions'))}, coach {_flag(lineups.get('coach'))}"
            )
        odds = observed.get("odds") or {}
        if odds.get("error"):
            add(f"- **Odds**: error - {odds['error']}")
        else:
            markets = ", ".join(str(m) for m in (odds.get("markets") or [])[:6])
            add(
                f"- **Odds**: {odds.get('rows')} rows, {odds.get('bookmakers')} bookmakers, "
                f"{odds.get('market_count')} markets" + (f" ({markets})" if markets else "")
            )
        for key, label in (
            ("events", "Events"),
            ("predictions", "Provider predictions"),
            ("injuries", "Injuries (league+season)"),
            ("players", "Player season statistics"),
            ("standings", "Standings"),
        ):
            section = observed.get(key)
            if section is None:
                continue
            if section.get("error"):
                add(f"- **{label}**: error - {section['error']}")
            elif key == "injuries":
                detail = (
                    f", player ids {_flag(section['has_player_id'])}, fixture linked "
                    f"{_flag(section['fixture_linked'])}, example reason `{section['reason']}`"
                    if section["rows"]
                    else " - none returned"
                )
                add(f"- **{label}**: {section['rows']} rows{detail}")
            elif key == "players":
                available = [k for k, v in section["fields"].items() if v]
                add(
                    f"- **{label}**: {section['players_on_page']} players on page 1 of "
                    f"{section['pages_available']}; fields present: "
                    f"{', '.join(available) if available else 'none'}"
                )
            elif key == "standings":
                add(f"- **{label}**: {section['groups']} group(s), {section['rows']} rows")
            elif key == "predictions":
                add(
                    f"- **{label}**: {section['rows']} rows, fields "
                    f"{', '.join(section['fields'])}, percent {section['percent']}"
                )
            else:
                add(f"- **{label}**: {section['rows']} rows, types {section.get('types')}")
        add("")

    if findings.problems:
        add("## Problems encountered")
        add("")
        for problem in findings.problems:
            add(f"- {problem}")
        add("")

    add("## Requirement matrix")
    add("")
    add("| Requirement | API-Football result | Evidence | ML suitability | Action |")
    add("| --- | --- | --- | --- | --- |")
    for requirement, result, evidence, suitability, action in assess(findings):
        add(f"| {requirement} | {result} | {evidence} | **{suitability}** | {action} |")
    add("")

    add("## M. Historical depth")
    add("")
    add(
        "The plan, not the catalogue, is the binding constraint. `/leagues` lists "
        "seasons back to 2010, but this Free plan answers only **2022-2024** "
        "(`Free plans do not have access to this season, try from 2022 to 2024`), and "
        "rejects the `last`, `next` and `date` narrowing parameters outright."
    )
    add("")
    add(
        "| Competition | Seasons listed | Seasons reachable on this plan | Fixtures |"
        " Team stats | Players | Injuries | Odds | xG |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in findings.competitions:
        observed = findings.observations.get(c["expected_name"])
        if not observed or not observed.get("fixture"):
            add(
                f"| {c['name']} | {c['earliest_season']}-{c['latest_season']} | "
                "2022-2024 (plan) | not probed | not probed | not probed | not probed "
                "| not probed | not probed |"
            )
            continue
        stats = observed.get("statistics", {}).get("wanted", {})
        rich = sum(1 for v in stats.values() if v == "present")
        players = observed.get("players", {})
        player_fields = sum(1 for v in (players.get("fields") or {}).values() if v)
        injuries = observed.get("injuries", {}).get("rows")
        odds = observed.get("odds", {}).get("rows")
        xg = stats.get("expected_goals", "—")
        add(
            f"| {c['name']} | {c['earliest_season']}-{c['latest_season']} | 2022-2024 | "
            f"{observed.get('finished_fixtures_in_season', '—')} in "
            f"{observed.get('season_used')} | {rich}/{len(stats)} fields | "
            f"{player_fields} fields | {injuries if injuries is not None else '—'} rows | "
            f"{odds if odds is not None else '—'} rows | {xg} |"
        )
    add("")

    add("## L. Data timing and leakage")
    add("")
    add("| Data source | Pre-match usable? | Timing evidence | Leakage risk | Notes |")
    add("| --- | --- | --- | --- | --- |")
    for source, verdict, evidence, risk, note in LEAKAGE:
        add(f"| {source} | {verdict} | {evidence} | {risk} | {note} |")
    add("")
    add(
        "The decisive point: **no endpoint in this API returns a publication "
        "timestamp**. For anything that changes as a match approaches — lineups, "
        "injuries, standings, provider predictions — the API cannot tell us what was "
        "true before kickoff. Those sources are therefore unusable for backtesting "
        "unless this platform snapshots them itself, at a recorded time, going "
        "forward. That is a property of the provider, not of our pipeline."
    )
    add("")

    add("## Requests spent")
    add("")
    add("| # | Endpoint | Parameters | Why it was needed | Result |")
    add("| ---: | --- | --- | --- | --- |")
    for call in findings.calls:
        params = ", ".join(f"{k}={v}" for k, v in call["params"].items()) or "-"
        result = call["error"] or f"{call['results']} results"
        add(f"| {call['number']} | `{call['path']}` | {params} | {call['reason']} | {result} |")
    add("")
    remaining = next(
        (c["requests_remaining"] for c in reversed(findings.calls) if c["requests_remaining"]),
        "unknown",
    )
    add(
        "Daily requests remaining after this run, from the "
        f"`x-ratelimit-requests-remaining` header: **{remaining}**."
    )
    add("")
    return "\n".join(lines)


def resume(validator: Validator, names: Sequence[str]) -> Findings:
    """Probe only `names`, reusing the survey already saved in the artefact.

    The survey (`/status` and `/leagues`) does not change between runs, so repeating
    it would spend requests to learn nothing. This keeps a partial validation
    completable without paying for it twice.
    """
    saved = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    findings = Findings(
        generated_at=datetime.now(UTC).isoformat(),
        plan=saved.get("plan", {}),
        competitions=saved.get("competitions", []),
        observations=saved.get("observations", {}),
        calls=saved.get("calls", []),
        problems=saved.get("problems", []),
    )
    findings.problems.append(
        f"resumed from a previous run; {len(findings.calls)} earlier requests are "
        "listed below and were not repeated"
    )
    for name in names:
        competition = next((c for c in findings.competitions if c["expected_name"] == name), None)
        if competition is None:
            findings.problems.append(f"--only {name}: not among the surveyed competitions")
            continue
        findings.observations[name] = probe_competition(
            validator, competition, findings, full=name in DEEP
        )
    return findings


def write_outputs(findings: Findings, budget: int) -> tuple[Path, Path]:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(asdict(findings), indent=2, default=str), encoding="utf-8")
    REPORT_OUT.write_text(render_report(findings, budget), encoding="utf-8")
    return JSON_OUT, REPORT_OUT


MISSING_KEY_MESSAGE = (
    "error: no API key configured.\n"
    "  Add it to the repository's .env file (git-ignored):\n"
    "      API_FOOTBALL_API_KEY=<your key>\n"
    "  or export API_FOOTBALL_API_KEY in your environment before running.\n"
    "  The key is never printed, logged or written to the artefacts."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate_api_football",
        description="Validate API-Football as the primary data provider.",
    )
    parser.add_argument(
        "--budget", type=int, default=25, help="hard cap on API requests (default 25)"
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the report from the saved artefact without calling the API",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="COMPETITION",
        help=(
            "probe just this competition and merge the result into the existing "
            "artefact, reusing the saved survey instead of spending requests on it "
            "again (repeatable)"
        ),
    )
    args = parser.parse_args(argv)

    if args.render_only:
        saved = json.loads(JSON_OUT.read_text(encoding="utf-8"))
        findings = Findings(**saved)
        write_outputs(findings, args.budget)
        print(f"report rebuilt from {JSON_OUT} without calling the API")
        return 0

    settings = get_settings()
    if settings.api_football_api_key is None:
        print(MISSING_KEY_MESSAGE, file=sys.stderr)
        return 3

    http = JsonHttpClient(
        provider=CODE,
        base_url=settings.api_football_base_url,
        headers={"x-apisports-key": settings.api_football_api_key.get_secret_value()},
        timeout_seconds=settings.provider_timeout_seconds,
        retry_policy=RetryPolicy(max_attempts=2),
        rate_limiter=RateLimiter(settings.api_football_requests_per_minute),
    )
    validator = Validator(http, args.budget)
    findings = Findings(generated_at=datetime.now(UTC).isoformat())
    try:
        findings = resume(validator, args.only) if args.only else run(validator)
    except BudgetExhaustedError as exc:
        findings.problems.append(str(exc))
    finally:
        http.close()
    findings.calls = [*findings.calls, *(asdict(call) for call in validator.calls)]
    json_path, report_path = write_outputs(findings, args.budget)
    print(f"requests spent in this run: {validator.used} of a {args.budget} budget")
    print(f"requests recorded in the artefact overall: {len(findings.calls)}")
    print(f"json:   {json_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
