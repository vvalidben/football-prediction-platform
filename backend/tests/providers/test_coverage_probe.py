"""The coverage probe reports only what responses contain, within its request budget.

All payloads here are synthetic, shaped like API-Football responses; they are not
claims about real coverage.
"""

from datetime import date
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from app.providers.api_football.coverage import (
    Competition,
    Probe,
    SeasonInfo,
    observe_fixture,
    run,
    sample_seasons,
)
from app.providers.http import JsonHttpClient, RetryPolicy
from tests.providers.fake_api import envelope

TODAY = date(2026, 9, 22)
ALL_FLAGS = {
    "fixtures": {
        "events": True,
        "lineups": True,
        "statistics_fixtures": True,
        "statistics_players": True,
    },
    "standings": True,
    "players": True,
    "injuries": True,
    "odds": True,
}


def season(
    year: int, *, current: bool = False, coverage: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "year": year,
        "start": f"{year}-08-01",
        "end": f"{year + 1}-05-31",
        "current": current,
        "coverage": ALL_FLAGS if coverage is None else coverage,
    }


def league(league_id: int, name: str, country: str, years: range) -> dict[str, Any]:
    return {
        "league": {"id": league_id, "name": name, "type": "League"},
        "country": {"name": country},
        "seasons": [season(y, current=(y == years[-1])) for y in years],
    }


def fixture_item(fixture_id: int, *, stats: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "fixture": {"id": fixture_id, "date": "2024-11-01T15:00:00+00:00"},
        "teams": {"home": {"name": "Home FC"}, "away": {"name": "Away FC"}},
        "statistics": [{"team": {"id": 1}, "statistics": stats or []}],
        "lineups": [{"team": {"id": 1}}, {"team": {"id": 2}}] if stats else [],
        "players": [{"players": [{}, {}]}] if stats else [],
        "events": [],
    }


FULL_STATS: list[dict[str, Any]] = [
    {"type": "Total Shots", "value": 14},
    {"type": "Shots on Goal", "value": 5},
    {"type": "Ball Possession", "value": "55%"},
    {"type": "Corner Kicks", "value": 6},
    {"type": "Yellow Cards", "value": 2},
    {"type": "Red Cards", "value": None},
    {"type": "expected_goals", "value": "1.73"},
]


class SyntheticApi:
    """Routes by path and query; records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        params = dict(request.url.params)
        path = request.url.path
        if path == "/status":
            return self._ok(
                {"subscription": {"plan": "Free"}, "requests": {"current": 3, "limit_day": 100}}
            )
        if path == "/leagues" and params == {"country": "Morocco"}:
            return self._ok([league(200, "Botola Pro", "Morocco", range(2020, 2027))])
        if path == "/leagues":
            league_id = int(params["id"])
            name = {39: "Premier League", 140: "La Liga"}.get(league_id)
            if name is None:
                return self._ok([])
            return self._ok([league(league_id, name, "X", range(2015, 2027))])
        if path == "/fixtures" and "ids" in params:
            ids = [int(i) for i in params["ids"].split("-")]
            # Fixture ids encode the season (see below); 2019 fixtures have no statistics.
            return self._ok(
                [fixture_item(i, stats=None if (i // 100) % 100 == 19 else FULL_STATS) for i in ids]
            )
        if path == "/fixtures" and "next" in params:
            return self._ok([fixture_item(999)])
        if path == "/fixtures":
            year = int(params["season"])
            if year < 2018:  # simulate a plan restriction on old seasons
                return self._ok(
                    [], errors={"plan": "Free plans do not have access to this season."}
                )
            # id = league * 10000 + (year % 100) * 100 + n
            base = int(params["league"]) * 10_000 + (year % 100) * 100
            return self._ok(
                [{"fixture": {"id": base + n, "date": f"2024-0{n}-01"}} for n in range(1, 4)]
            )
        if path == "/injuries":
            return self._ok([{"player": {"id": 1}}])
        if path == "/odds":
            fixture_id = int(params["fixture"])
            if fixture_id == 999:
                return self._ok(
                    [{"update": "2026-09-21T10:00:00+00:00", "bookmakers": [{"id": 8}]}]
                )
            return self._ok([])  # historical fixtures: no odds returned
        return httpx.Response(404)

    @staticmethod
    def _ok(response: Any, **overrides: Any) -> httpx.Response:
        headers = {"X-RateLimit-Limit": "10", "X-RateLimit-Remaining": "9"}
        return httpx.Response(200, json=envelope(response, **overrides), headers=headers)


def make_probe(tmp_path: Path, budget: int = 60) -> tuple[Probe, SyntheticApi]:
    api = SyntheticApi()
    http = JsonHttpClient(
        provider="api_football",
        base_url="https://api.test",
        retry_policy=RetryPolicy(max_attempts=1),
        transport=httpx.MockTransport(api.handle),
    )
    return Probe(http, budget, tmp_path), api


def test_report_separates_observations_from_flags(tmp_path: Path) -> None:
    probe, api = make_probe(tmp_path)

    report = run(probe, TODAY, older_offset=5)

    # Botola is discovered by search, not by a hard-coded id.
    assert "### Botola Pro (id 200, Morocco)" in report
    # Recent season: everything observed, xG present.
    assert "| Premier League | 2025 | 3 finished | ✓ (flag ✓) | ✓ | ✓ (flag ✓) |" in report
    assert "| Premier League | 2020 | 3 finished | ✓ (flag ✓) |" in report
    # Historical odds: flag claims coverage, response had none.
    assert "✗ (flag ✓)" in report
    # Upcoming fixture shows odds with their update timestamp.
    assert "odds updated 2026-09-21T10:00:00+00:00" in report
    # Unknown ids are reported, never guessed.
    assert "league 135 (Serie A): empty response" in report
    # Requests: every one counted, and fixture details fetched in one batched call.
    batched = [r for r in api.requests if "ids" in r.url.params]
    assert len(batched) == 1
    assert f"Requests sent by this run: **{len(api.requests)}**" in report
    assert len(list(tmp_path.glob("*.json"))) == len(api.requests)
    # Rate-limit headers are captured verbatim for checking the plan's limits.
    assert "`x-ratelimit-limit: 10`" in report


def test_fixture_without_statistics_is_reported_as_missing(tmp_path: Path) -> None:
    probe, _ = make_probe(tmp_path)

    report = run(probe, TODAY, older_offset=6)  # samples 2019, whose fixtures lack stats

    row = next(line for line in report.splitlines() if line.startswith("| La Liga | 2019 |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[2:6] == ["3 finished", "✗ (flag ✓)", "✗", "✗ (flag ✓)"]


def test_plan_restricted_season_is_reported_with_the_provider_message(tmp_path: Path) -> None:
    probe, _ = make_probe(tmp_path)

    report = run(probe, TODAY, older_offset=8)  # samples 2017, blocked by the plan

    row = next(line for line in report.splitlines() if line.startswith("| Premier League | 2017 |"))
    assert "| — | — (flag ✓) |" in row  # nothing observed, although flagged
    assert "Free plans do not have access to this season" in row


def test_budget_is_never_exceeded(tmp_path: Path) -> None:
    probe, api = make_probe(tmp_path, budget=5)

    report = run(probe, TODAY, older_offset=5)

    assert len(api.requests) == 5
    assert "stopped early: request budget of 5 reached" in report


class TestHelpers:
    def test_sample_seasons_picks_latest_completed_and_an_older_one(self) -> None:
        seasons = [
            SeasonInfo(y, date(y, 8, 1), date(y + 1, 5, 31), y == 2026, {})
            for y in range(2018, 2027)
        ]
        competition = Competition(1, "L", None, seasons)

        chosen = [s.year for s in sample_seasons(competition, TODAY, older_offset=5)]

        assert chosen == [2025, 2020]  # 2026 is still running

    def test_short_history_falls_back_to_the_earliest_season(self) -> None:
        seasons = [
            SeasonInfo(y, date(y, 8, 1), date(y + 1, 5, 31), False, {}) for y in (2023, 2024)
        ]

        chosen = sample_seasons(Competition(1, "L", None, seasons), TODAY, older_offset=5)

        assert [s.year for s in chosen] == [2024, 2023]

    def test_observe_fixture_detects_xg_and_null_values(self) -> None:
        observation = observe_fixture(fixture_item(1, stats=FULL_STATS))

        assert observation.stats["xg"] is True
        assert observation.stats["shots_on_target"] is True
        assert observation.stats["red_cards"] is False  # present but null
        assert "expected_goals" in observation.stat_labels

    def test_observe_fixture_without_statistics(self) -> None:
        observation = observe_fixture(fixture_item(1))

        assert observation.stat_labels == []
        assert not any(observation.stats.values())
        assert observation.lineups == 0


@pytest.mark.parametrize("missing", ["Total Shots", "expected_goals"])
def test_absent_statistics_are_not_marked_available(missing: str) -> None:
    stats = [s for s in FULL_STATS if s["type"] != missing]

    observation = observe_fixture(fixture_item(1, stats=stats))

    key = "shots" if missing == "Total Shots" else "xg"
    assert observation.stats[key] is False
