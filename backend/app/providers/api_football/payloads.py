"""API-Football v3 response shapes (https://www.api-football.com/documentation-v3).

Only the fields we use are declared; unknown fields are ignored so additive API
changes do not break parsing. Each item in a response is validated separately, so
one malformed record is rejected without failing the whole response.
"""

from datetime import date
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Paging(_Payload):
    current: int = 1
    total: int = 1


class Envelope(_Payload):
    """Every endpoint wraps its data like this. `errors` is `[]` or an object."""

    get: str | None = None
    errors: list[Any] | dict[str, Any] = []
    results: int | None = None
    paging: Paging = Paging()
    response: list[Any]


class League(_Payload):
    id: int
    name: str
    type: str  # "League" or "Cup"
    logo: str | None = None


class Country(_Payload):
    name: str | None = None  # "World" for international competitions
    code: str | None = None


class FixturesCoverage(_Payload):
    events: bool = False
    lineups: bool = False
    statistics_fixtures: bool = False
    statistics_players: bool = False


class Coverage(_Payload):
    fixtures: FixturesCoverage = FixturesCoverage()
    injuries: bool = False
    odds: bool = False


class Season(_Payload):
    year: int
    start: date | None = None
    end: date | None = None
    current: bool = False
    coverage: Coverage | None = None


class LeagueItem(_Payload):
    """An item of `GET /leagues`."""

    league: League
    country: Country = Country()
    seasons: list[Season] = []


class Team(_Payload):
    id: int
    name: str
    code: str | None = None
    country: str | None = None
    national: bool = False
    logo: str | None = None


class TeamItem(_Payload):
    """An item of `GET /teams`."""

    team: Team


class FixtureStatus(_Payload):
    short: str
    long: str | None = None


class FixtureVenue(_Payload):
    name: str | None = None
    city: str | None = None


class Fixture(_Payload):
    id: int
    referee: str | None = None
    date: AwareDatetime
    venue: FixtureVenue = FixtureVenue()
    status: FixtureStatus


class FixtureLeague(_Payload):
    id: int
    season: int
    round: str | None = None


class FixtureTeam(_Payload):
    id: int
    name: str | None = None


class FixtureTeams(_Payload):
    home: FixtureTeam
    away: FixtureTeam


class Goals(_Payload):
    home: int | None = None
    away: int | None = None


class Score(_Payload):
    halftime: Goals = Goals()
    fulltime: Goals = Goals()
    extratime: Goals = Goals()
    penalty: Goals = Goals()


class FixtureItem(_Payload):
    """An item of `GET /fixtures`. `goals` includes extra time, not penalties."""

    fixture: Fixture
    league: FixtureLeague
    teams: FixtureTeams
    goals: Goals = Goals()
    score: Score = Score()


class StatisticEntry(_Payload):
    type: str
    value: int | float | str | None = None


class TeamStatistics(_Payload):
    team: FixtureTeam
    statistics: list[StatisticEntry] = []


class FixtureDetailItem(_Payload):
    """An item of `GET /fixtures?ids=…`: the fixture plus embedded details."""

    fixture: Fixture
    teams: FixtureTeams
    statistics: list[TeamStatistics] = []
