"""Measure what API-Football actually returns for the competitions we care about.

    python -m app.providers.api_football.coverage [--budget 60] [--older-offset 5]

Needs API_FOOTBALL_API_KEY. It spends at most `--budget` requests (the free plan
allows 100/day) and writes every raw response plus `report.md` to
`reports/api_football_coverage/<timestamp>/`.

The report keeps two kinds of evidence apart:
- **flag**: what the provider *claims* in each season's `coverage` object;
- **observed**: what a real response for a sampled fixture actually contained.

Nothing is inferred. A cell is only marked available when a response showed it.
"""

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.providers.api_football.provider import CODE, check_envelope
from app.providers.errors import ProviderError
from app.providers.http import JsonHttpClient, RateLimiter, RetryPolicy

#: Target competitions whose API-Football IDs are widely published. Each is
#: verified at run time against the name the API returns.
KNOWN_TARGETS: tuple[tuple[int, str], ...] = (
    (39, "Premier League"),
    (140, "La Liga"),
    (135, "Serie A"),
    (78, "Bundesliga"),
    (61, "Ligue 1"),
    (2, "UEFA Champions League"),
)
#: Found by searching the country's leagues, not by a remembered ID.
BOTOLA_COUNTRY, BOTOLA_NAME = "Morocco", "botola"

#: Statistic `type` labels as documented for fixture statistics. The raw labels
#: seen are also reported, so a label mismatch is visible rather than hidden.
STAT_LABELS: dict[str, tuple[str, ...]] = {
    "shots": ("total shots",),
    "shots_on_target": ("shots on goal",),
    "possession": ("ball possession",),
    "corners": ("corner kicks",),
    "yellow_cards": ("yellow cards",),
    "red_cards": ("red cards",),
    "xg": ("expected_goals", "expected goals"),
}
CORE_STATS = ("shots", "shots_on_target", "possession", "corners", "yellow_cards")
SEASON_FLAG_ORDER = (
    "events",
    "lineups",
    "stats",
    "player_stats",
    "standings",
    "players",
    "injuries",
    "odds",
)
FINISHED = "FT-AET-PEN"
MAX_IDS_PER_REQUEST = 20


class BudgetExhaustedError(Exception):
    pass


@dataclass
class SeasonInfo:
    year: int
    start: date | None
    end: date | None
    current: bool
    flags: dict[str, bool]


@dataclass
class Competition:
    league_id: int
    name: str
    country: str | None
    seasons: list[SeasonInfo]


@dataclass
class FixtureObservation:
    fixture_id: int
    kickoff: str = ""
    teams: str = ""
    #: True once the fixture's detail response has been read.
    detailed: bool = False
    stat_labels: list[str] = field(default_factory=list)
    stats: dict[str, bool] = field(default_factory=dict)
    lineups: int = 0
    players: int = 0
    events: int = 0
    injuries: int | None = None
    odds_bookmakers: int | None = None
    odds_updated: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class SampleRow:
    competition: Competition
    season: SeasonInfo
    finished_fixtures: int | None = None
    fixture: FixtureObservation | None = None
    notes: list[str] = field(default_factory=list)
    upcoming: bool = False


class Probe:
    """Budgeted GETs that save every response and never raise provider errors."""

    def __init__(self, http: JsonHttpClient, budget: int, out_dir: Path) -> None:
        self.http = http
        self.budget = budget
        self.out_dir = out_dir
        self.log: list[str] = []
        #: Rate-limit headers from the latest response, to verify the plan's limits.
        self.rate_limit_headers: dict[str, str] = {}

    def get(self, path: str, params: Mapping[str, str | int]) -> tuple[dict[str, Any] | None, str]:
        """Return `(envelope, "")` on success or `(None, error text)` on failure."""
        if self.http.requests_sent >= self.budget:
            raise BudgetExhaustedError(f"request budget of {self.budget} reached")
        index = self.http.requests_sent + 1
        query = "&".join(f"{key}={value}" for key, value in params.items())
        body: dict[str, Any] | None
        try:
            body, error = self.http.get_json(path, params, check=check_envelope), ""
        except ProviderError as exc:
            body, error = None, f"{type(exc).__name__}: {exc}"
        headers = {k: v for k, v in self.http.last_headers.items() if "ratelimit" in k.lower()}
        if headers:
            self.rate_limit_headers = headers
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{path}_{query}").strip("_")[:80]
        saved = {"rate_limit_headers": headers, "body": body if body is not None else error}
        (self.out_dir / f"{index:03d}_{slug}.json").write_text(
            json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.log.append(f"{index:>3}  GET {path}?{query}  {'ERROR ' + error if error else 'ok'}")
        return body, error


def parse_competition(item: Mapping[str, Any]) -> Competition:
    seasons = []
    for raw in item.get("seasons") or []:
        coverage = raw.get("coverage") or {}
        fixtures = coverage.get("fixtures") or {}
        flags = {
            "events": bool(fixtures.get("events")),
            "lineups": bool(fixtures.get("lineups")),
            "stats": bool(fixtures.get("statistics_fixtures")),
            "player_stats": bool(fixtures.get("statistics_players")),
            "standings": bool(coverage.get("standings")),
            "players": bool(coverage.get("players")),
            "injuries": bool(coverage.get("injuries")),
            "odds": bool(coverage.get("odds")),
        }
        seasons.append(
            SeasonInfo(
                year=int(raw["year"]),
                start=date.fromisoformat(raw["start"]) if raw.get("start") else None,
                end=date.fromisoformat(raw["end"]) if raw.get("end") else None,
                current=bool(raw.get("current")),
                flags=flags,
            )
        )
    league, country = item["league"], item.get("country") or {}
    return Competition(
        league_id=int(league["id"]),
        name=str(league["name"]),
        country=country.get("name"),
        seasons=sorted(seasons, key=lambda s: s.year),
    )


def sample_seasons(competition: Competition, today: date, older_offset: int) -> list[SeasonInfo]:
    """The most recent completed season, plus one `older_offset` years earlier
    (or the earliest listed, if the history is shorter)."""
    completed = [s for s in competition.seasons if s.end is not None and s.end < today]
    if not completed:
        return []
    recent = completed[-1]
    older = next((s for s in completed if s.year == recent.year - older_offset), completed[0])
    return [recent] if older is recent else [recent, older]


def observe_fixture(item: Mapping[str, Any]) -> FixtureObservation:
    """Record which data a `/fixtures?ids=` item actually contains."""
    fixture, teams = item["fixture"], item.get("teams") or {}
    home, away = (teams.get("home") or {}).get("name"), (teams.get("away") or {}).get("name")
    observation = FixtureObservation(
        fixture_id=int(fixture["id"]),
        kickoff=str(fixture.get("date") or ""),
        teams=f"{home} v {away}",
        detailed=True,
    )
    values: dict[str, list[Any]] = {}
    for team_block in item.get("statistics") or []:
        for stat in team_block.get("statistics") or []:
            label = str(stat.get("type")).strip().lower()
            values.setdefault(label, []).append(stat.get("value"))
    observation.stat_labels = sorted(values)
    for key, labels in STAT_LABELS.items():
        seen = [value for label in labels for value in values.get(label, [])]
        observation.stats[key] = any(value is not None for value in seen)
    observation.lineups = len(item.get("lineups") or [])
    observation.players = sum(len(team.get("players") or []) for team in item.get("players") or [])
    observation.events = len(item.get("events") or [])
    return observation


def _mark(observed: bool | None, flag: bool | None) -> str:
    seen = "—" if observed is None else ("✓" if observed else "✗")
    if flag is None:
        return seen
    return f"{seen} (flag {'✓' if flag else '✗'})"


def _matrix_row(row: SampleRow) -> str:
    flags, obs = row.season.flags, row.fixture
    notes = list(row.notes)
    stats = xg = lineups = players = odds = None
    injuries = "—"
    if obs is not None:
        notes.insert(0, f"fixture {obs.fixture_id} {obs.teams} {obs.kickoff[:10]}".strip())
        if obs.detailed and not row.upcoming:
            stats = bool(obs.stat_labels) and all(obs.stats.get(k, False) for k in CORE_STATS)
            xg = obs.stats.get("xg", False)
            lineups, players = obs.lineups > 0, obs.players > 0
            absent = [k for k in CORE_STATS if not obs.stats.get(k, False)]
            if obs.stat_labels and absent:
                notes.append(f"missing stats: {', '.join(absent)}")
        if obs.injuries is not None:
            injuries = f"{obs.injuries} (flag {'✓' if flags['injuries'] else '✗'})"
        if obs.odds_bookmakers is not None:
            odds = obs.odds_bookmakers > 0
            notes.append(f"odds: {obs.odds_bookmakers} bookmaker(s)")
        if obs.odds_updated:
            notes.append(f"odds updated {obs.odds_updated}")
        notes.extend(obs.errors)
    cells = [
        row.competition.name,
        f"{row.season.year}{' (upcoming)' if row.upcoming else ''}",
        "—" if row.finished_fixtures is None else f"{row.finished_fixtures} finished",
        _mark(stats, flags["stats"]),
        _mark(xg, None),
        _mark(lineups, flags["lineups"]),
        injuries,
        _mark(players, flags["player_stats"]),
        _mark(odds, flags["odds"]),
        "; ".join(notes),
    ]
    return "| " + " | ".join(cells) + " |"


def render_report(
    competitions: Sequence[Competition],
    rows: Sequence[SampleRow],
    probe: Probe,
    status_before: str,
    status_after: str,
    problems: Sequence[str],
) -> str:
    lines = [
        "# API-Football coverage check",
        "",
        f"Run at {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Requests sent by this run: "
        f"**{probe.http.requests_sent}** (retries included).",
        "",
        f"- Account status before: {status_before}",
        f"- Account status after: {status_after}",
        "- Rate-limit headers on the last response: "
        + (", ".join(f"`{k}: {v}`" for k, v in sorted(probe.rate_limit_headers.items())) or "none"),
        "",
        "Cells read **observed (flag)**: *observed* is what a real response for the",
        "sampled fixture contained; *flag* is the season's `coverage` claim. `—` means",
        "not observed (request failed, not sampled, or not applicable). An injuries",
        "count of 0 cannot distinguish 'nobody injured' from 'not covered'.",
        "",
        "## Sampled fixtures",
        "",
        "| Competition | Season | Fixtures | Stats | xG | Lineups | Injuries | Players | Odds"
        " | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(_matrix_row(row) for row in rows),
        "",
        "## Statistic labels seen per sampled fixture",
        "",
    ]
    for row in rows:
        if row.fixture and row.fixture.stat_labels:
            labels = ", ".join(row.fixture.stat_labels)
            lines.append(f"- {row.competition.name} {row.season.year}: {labels}")
    lines += ["", "## Claimed coverage by season (provider flags, not verified)", ""]
    for comp in competitions:
        lines += [
            f"### {comp.name} (id {comp.league_id}, {comp.country})",
            "",
            "| Season | Start to end | Current | " + " | ".join(SEASON_FLAG_ORDER) + " |",
            "|---|---|---|" + "---|" * len(SEASON_FLAG_ORDER),
        ]
        for season in comp.seasons:
            flags = " | ".join("✓" if season.flags[k] else "✗" for k in SEASON_FLAG_ORDER)
            current = "yes" if season.current else ""
            lines.append(
                f"| {season.year} | {season.start} to {season.end} | {current} | {flags} |"
            )
        lines.append("")
    if problems:
        lines += ["## Problems", "", *(f"- {p}" for p in problems), ""]
    lines += ["## Request log", "", "```", *probe.log, "```", ""]
    return "\n".join(lines)


def _status(probe: Probe) -> str:
    try:
        body, error = probe.get("/status", {})
    except BudgetExhaustedError:
        return "not checked (budget reached)"
    if body is None:
        return f"unavailable ({error})"
    response = body.get("response") or {}
    if isinstance(response, list):  # an empty result may be a list
        response = response[0] if response else {}
    plan = (response.get("subscription") or {}).get("plan")
    requests = response.get("requests") or {}
    return f"plan={plan}, requests today={requests.get('current')}/{requests.get('limit_day')}"


def _read_odds(probe: Probe, observation: FixtureObservation) -> None:
    body, error = probe.get("/odds", {"fixture": observation.fixture_id})
    if body is None:
        observation.errors.append(f"odds: {error}")
        return
    items = body.get("response", [])
    observation.odds_bookmakers = sum(len(item.get("bookmakers") or []) for item in items)
    observation.odds_updated = next((item.get("update") for item in items), None)


def run(probe: Probe, today: date, older_offset: int) -> str:
    competitions: list[Competition] = []
    rows: list[SampleRow] = []
    problems: list[str] = []
    status_before = _status(probe)
    try:
        body, error = probe.get("/leagues", {"country": BOTOLA_COUNTRY})
        botola = [
            item
            for item in (body or {}).get("response", [])
            if BOTOLA_NAME in str((item.get("league") or {}).get("name", "")).lower()
        ]
        competitions.extend(parse_competition(item) for item in botola)
        if not botola:
            problems.append(f"no league named like 'Botola' in {BOTOLA_COUNTRY} {error}".strip())

        for league_id, expected in KNOWN_TARGETS:
            body, error = probe.get("/leagues", {"id": league_id})
            items = (body or {}).get("response", [])
            if not items:
                problems.append(f"league {league_id} ({expected}): {error or 'empty response'}")
                continue
            competition = parse_competition(items[0])
            if competition.name.lower() != expected.lower():
                problems.append(f"league {league_id} is '{competition.name}', not '{expected}'")
            competitions.append(competition)

        for competition in competitions:
            for season in sample_seasons(competition, today, older_offset):
                row = SampleRow(competition, season)
                rows.append(row)
                params = {"league": competition.league_id, "season": season.year}
                body, error = probe.get("/fixtures", {**params, "status": FINISHED})
                if body is None:
                    row.notes.append(error)
                    continue
                finished = sorted(body.get("response", []), key=lambda i: i["fixture"]["date"])
                row.finished_fixtures = len(finished)
                if finished:  # mid-season: least likely to have patchy coverage
                    middle = finished[len(finished) // 2]
                    row.fixture = FixtureObservation(fixture_id=int(middle["fixture"]["id"]))

        sampled = [row for row in rows if row.fixture is not None]
        for start in range(0, len(sampled), MAX_IDS_PER_REQUEST):
            batch = [row for row in sampled[start : start + MAX_IDS_PER_REQUEST] if row.fixture]
            ids = "-".join(str(row.fixture.fixture_id) for row in batch if row.fixture)
            body, error = probe.get("/fixtures", {"ids": ids})
            details = {int(i["fixture"]["id"]): i for i in (body or {}).get("response", [])}
            for row in batch:
                if row.fixture is None:
                    continue
                item = details.get(row.fixture.fixture_id)
                if item is None:
                    row.fixture.errors.append(f"details unavailable {error}".strip())
                else:
                    row.fixture = observe_fixture(item)

        for row in sampled:
            if row.fixture is None:
                continue
            body, error = probe.get("/injuries", {"fixture": row.fixture.fixture_id})
            if body is None:
                row.fixture.errors.append(f"injuries: {error}")
            else:
                row.fixture.injuries = len(body.get("response", []))
            _read_odds(probe, row.fixture)

        # One upcoming fixture, to see what pre-match odds and their timestamps look like.
        first = next((c for c in competitions if c.league_id == KNOWN_TARGETS[0][0]), None)
        if first is not None and first.seasons:
            body, error = probe.get("/fixtures", {"league": first.league_id, "next": 1})
            upcoming = (body or {}).get("response", [])
            if upcoming:
                current = next((s for s in first.seasons if s.current), first.seasons[-1])
                row = SampleRow(first, current, notes=["odds timing check"], upcoming=True)
                row.fixture = observe_fixture(upcoming[0])
                _read_odds(probe, row.fixture)
                rows.append(row)
            else:
                problems.append(f"no upcoming fixture found for odds timing {error}".strip())
    except BudgetExhaustedError as exc:
        problems.append(f"stopped early: {exc}")
    status_after = _status(probe)
    return render_report(competitions, rows, probe, status_before, status_after, problems)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.providers.api_football.coverage",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--budget", type=int, default=60, help="maximum requests (default 60)")
    parser.add_argument(
        "--older-offset", type=int, default=5, help="years back for the older sample (default 5)"
    )
    parser.add_argument("--out", type=Path, default=Path("reports/api_football_coverage"))
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.api_football_api_key is None:
        print("error: API_FOOTBALL_API_KEY is not set", file=sys.stderr)
        return 3
    out_dir = args.out / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    http = JsonHttpClient(
        provider=CODE,
        base_url=settings.api_football_base_url,
        headers={"x-apisports-key": settings.api_football_api_key.get_secret_value()},
        timeout_seconds=settings.provider_timeout_seconds,
        retry_policy=RetryPolicy(max_attempts=2),
        rate_limiter=RateLimiter(settings.api_football_requests_per_minute),
    )
    try:
        report = run(Probe(http, args.budget, out_dir), datetime.now(UTC).date(), args.older_offset)
    finally:
        http.close()
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(report)
    print(f"\nRaw responses and report: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
