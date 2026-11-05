"""A fake football-data.org v4 server for tests, built on httpx's MockTransport.

The payload shapes were recorded from the real API, including the one that matters
most: `score.fullTime` of a shootout includes the penalties.
"""

from collections import deque
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.providers.football_data import FootballDataOrgProvider

BASE_URL = "https://football-data.test/v4"
PREMIER_LEAGUE = "2021"
SEASON_2025 = "2021:2025"

Responder = Callable[[httpx.Request], httpx.Response]


def score(
    full: tuple[int | None, int | None] = (None, None),
    half: tuple[int | None, int | None] = (None, None),
    *,
    duration: str = "REGULAR",
    penalties: tuple[int, int] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "winner": None,
        "duration": duration,
        "fullTime": {"home": full[0], "away": full[1]},
        "halfTime": {"home": half[0], "away": half[1]},
    }
    if penalties is not None:
        body["penalties"] = {"home": penalties[0], "away": penalties[1]}
    return body


def team_stub(team_id: int, name: str) -> dict[str, Any]:
    return {"id": team_id, "name": name, "shortName": name, "tla": name[:3].upper(), "crest": None}


def match_payload(match_id: int = 537785, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": match_id,
        "utcDate": "2025-08-15T19:00:00Z",
        "status": "FINISHED",
        "matchday": 1,
        "stage": "REGULAR_SEASON",
        "group": None,
        "homeTeam": team_stub(64, "Liverpool FC"),
        "awayTeam": team_stub(1044, "AFC Bournemouth"),
        "score": score((4, 2), (1, 0)),
        "odds": {"msg": "Activate Odds-Package in User-Panel to retrieve odds."},
        "referees": [
            {"id": 1, "name": "Some Assistant", "type": "ASSISTANT_REFEREE_N1"},
            {"id": 11580, "name": "Anthony Taylor", "type": "REFEREE"},
        ],
    }
    payload.update(overrides)
    return payload


def competition_payload() -> dict[str, Any]:
    return {
        "id": 2021,
        "name": "Premier League",
        "code": "PL",
        "type": "LEAGUE",
        "emblem": "https://crests.football-data.org/PL.png",
        "area": {"id": 2072, "name": "England", "code": "ENG"},
        "currentSeason": {"id": 2502, "startDate": "2026-08-21", "endDate": "2027-05-30"},
        "seasons": [
            {"id": 2502, "startDate": "2026-08-21", "endDate": "2027-05-30"},
            {"id": 2403, "startDate": "2025-08-15", "endDate": "2026-05-24"},
            {"id": 2287, "startDate": "2024-08-16", "endDate": "2025-05-25"},
            {"id": 2021, "startDate": "2023-08-11", "endDate": "2024-05-19"},
            {"id": 1490, "startDate": "2022-08-05", "endDate": "2023-05-28"},
        ],
    }


class FakeFootballData:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.failures: deque[Responder] = deque()
        self.competitions: list[dict[str, Any]] = [
            competition_payload(),
            {
                "id": 2000,
                "name": "FIFA World Cup",
                "code": "WC",
                "type": "CUP",
                "area": {"id": 2267, "name": "World", "code": "INT"},
            },
            {
                "id": 2001,
                "name": "UEFA Champions League",
                "code": "CL",
                "type": "CUP",
                "area": {"id": 2077, "name": "Europe", "code": "EUR"},
            },
            {"id": 9, "name": "Some Super Cup", "code": "SC", "type": "SUPER_CUP", "area": None},
        ]
        self.teams: list[dict[str, Any]] = [
            {
                **team_stub(64, "Liverpool FC"),
                "shortName": "Liverpool",
                "tla": "LIV",
                "crest": "https://crests.football-data.org/64.png",
                "area": {"id": 2072, "name": "England", "code": "ENG"},
            },
            {**team_stub(1044, "AFC Bournemouth"), "area": {"name": "England"}},
        ]
        self.matches: list[dict[str, Any]] = [match_payload()]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            return self.failures.popleft()(request)
        path = request.url.path.removeprefix("/v4")
        if path == "/competitions":
            return httpx.Response(
                200, json={"count": len(self.competitions), "competitions": self.competitions}
            )
        if path == f"/competitions/{PREMIER_LEAGUE}":
            return httpx.Response(200, json=competition_payload())
        if path == f"/competitions/{PREMIER_LEAGUE}/teams":
            return httpx.Response(200, json={"count": len(self.teams), "teams": self.teams})
        if path == f"/competitions/{PREMIER_LEAGUE}/matches":
            return httpx.Response(200, json={"matches": self.matches})
        return httpx.Response(404, json={"message": "no such route", "errorCode": 404})

    def provider(
        self, sleeps: list[float] | None = None, **settings: Any
    ) -> FootballDataOrgProvider:
        recorded = sleeps if sleeps is not None else []
        config = Settings(
            _env_file=None,
            postgres_user="unused",
            postgres_password=SecretStr("unused"),
            postgres_db="unused",
            football_data_api_key=SecretStr("test-token"),
            football_data_base_url=BASE_URL,
            football_data_requests_per_minute=6000,
            **settings,
        )
        return FootballDataOrgProvider.from_settings(
            config, transport=httpx.MockTransport(self.handle), sleep=recorded.append
        )
