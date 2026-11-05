"""football-data.org v4 response shapes (https://docs.football-data.org).

Only the fields we use are declared; unknown fields are ignored so additive API
changes do not break parsing. Each item in a response is validated separately, so
one malformed record is rejected without failing the whole response.
"""

from datetime import date

from pydantic import AwareDatetime, BaseModel, ConfigDict


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Area(_Payload):
    name: str | None = None


class Competition(_Payload):
    id: int
    name: str
    type: str | None = None  # LEAGUE, CUP, SUPER_CUP or PLAYOFFS
    emblem: str | None = None
    area: Area | None = None


class Season(_Payload):
    id: int
    startDate: date  # noqa: N815 - the API's own name
    endDate: date  # noqa: N815


class CompetitionDetail(Competition):
    """`GET /competitions/{id}`."""

    currentSeason: Season | None = None  # noqa: N815
    seasons: list[Season] = []


class Team(_Payload):
    id: int
    name: str
    shortName: str | None = None  # noqa: N815
    tla: str | None = None
    crest: str | None = None
    area: Area | None = None


class MatchTeam(_Payload):
    """Teams inside a match; `id` and `name` are null until a cup tie is decided."""

    id: int


class Goals(_Payload):
    home: int | None = None
    away: int | None = None


class Score(_Payload):
    fullTime: Goals = Goals()  # noqa: N815
    halfTime: Goals = Goals()  # noqa: N815
    penalties: Goals | None = None


class Referee(_Payload):
    name: str | None = None
    type: str | None = None


class Match(_Payload):
    """An item of `GET /competitions/{id}/matches`."""

    id: int
    utcDate: AwareDatetime  # noqa: N815
    status: str
    matchday: int | None = None
    stage: str | None = None
    venue: str | None = None
    homeTeam: MatchTeam  # noqa: N815
    awayTeam: MatchTeam  # noqa: N815
    score: Score = Score()
    referees: list[Referee] = []
