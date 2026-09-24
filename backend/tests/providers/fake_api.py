"""A fake API-Football server for tests, built on httpx's MockTransport.

Routes are `(path, frozenset(query items))` → response factory. Unrouted requests
return 404. Failures can be queued to test retry behaviour.
"""

import json
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2 as httpx
from pydantic import SecretStr

from app.core.config import Settings
from app.providers.api_football import ApiFootballProvider

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "api_football"
BASE_URL = "https://api-football.test"

Responder = Callable[[httpx.Request], httpx.Response]


def load_fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


def envelope(response: list[Any], **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "get": "test",
        "parameters": {},
        "errors": [],
        "results": len(response),
        "paging": {"current": 1, "total": 1},
        "response": response,
    }
    body.update(overrides)
    return body


class FakeApiFootball:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, frozenset[tuple[str, str]]], Responder] = {}
        self.failures: deque[Responder] = deque()
        self.requests: list[httpx.Request] = []
        self.serve_fixture("/leagues", {}, "leagues.json")
        self.serve_fixture("/leagues", {"id": "39"}, "leagues_id_39.json")
        self.serve_fixture(
            "/teams", {"league": "39", "season": "2023"}, "teams_league_39_season_2023.json"
        )
        self.serve_fixture(
            "/fixtures", {"league": "39", "season": "2023"}, "fixtures_league_39_season_2023.json"
        )
        self.serve_fixture(
            "/teams", {"league": "39", "season": "2022"}, "teams_league_39_season_2022.json"
        )
        self.serve_fixture(
            "/fixtures", {"league": "39", "season": "2022"}, "fixtures_league_39_season_2022.json"
        )
        #: Fixture details served by `GET /fixtures?ids=a-b-c`, keyed by fixture id.
        self.fixture_details: dict[int, dict[str, Any]] = {
            item["fixture"]["id"]: item
            for item in load_fixture("fixture_details_league_39_season_2023.json")["response"]
        }

    # -- configuration ---------------------------------------------------------------

    def serve_json(self, path: str, params: dict[str, str], body: Any, status: int = 200) -> None:
        self.routes[(path, frozenset(params.items()))] = lambda _: httpx.Response(status, json=body)

    def serve_fixture(self, path: str, params: dict[str, str], name: str) -> None:
        self.serve_json(path, params, load_fixture(name))

    def fail_next(self, *responders: Responder) -> None:
        """Answer the next requests (any route) with these responders, in order."""
        self.failures.extend(responders)

    # -- transport -------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.failures:
            return self.failures.popleft()(request)
        if request.url.path == "/fixtures" and "ids" in request.url.params:
            ids = [int(i) for i in request.url.params["ids"].split("-")]
            if len(ids) > 20:
                return httpx.Response(
                    200, json=envelope([], errors={"ids": "Maximum of 20 ids allowed."})
                )
            found = [self.fixture_details[i] for i in ids if i in self.fixture_details]
            return httpx.Response(200, json=envelope(found))
        key = (request.url.path, frozenset(request.url.params.items()))
        responder = self.routes.get(key)
        if responder is None:
            return httpx.Response(404, json={"message": "no such route"})
        return responder(request)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def provider(self, sleeps: list[float] | None = None, **settings: Any) -> ApiFootballProvider:
        """An ApiFootballProvider talking to this fake; sleeps are recorded, not slept."""
        recorded = sleeps if sleeps is not None else []
        config = Settings(
            _env_file=None,
            postgres_user="unused",
            postgres_password=SecretStr("unused"),
            postgres_db="unused",
            api_football_api_key=SecretStr("test-key"),
            api_football_base_url=BASE_URL,
            api_football_requests_per_minute=6000,
            **settings,
        )
        return ApiFootballProvider.from_settings(
            config, transport=self.transport(), sleep=recorded.append
        )


def status(code: int, headers: dict[str, str] | None = None) -> Responder:
    return lambda _: httpx.Response(code, headers=headers, json={"message": "error"})


def body(payload: Any) -> Responder:
    return lambda _: httpx.Response(200, json=payload)


def raw(content: bytes) -> Responder:
    return lambda _: httpx.Response(200, content=content)


def timeout() -> Responder:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    return raise_timeout


def connection_error() -> Responder:
    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return raise_connect
