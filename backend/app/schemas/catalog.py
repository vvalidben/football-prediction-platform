"""Competitions, teams and matches as the API returns them."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Page(BaseModel):
    """Envelope for list endpoints."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(description="Rows matching the filters, ignoring limit and offset")
    limit: int
    offset: int


class CompetitionRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    type: str
    country: str | None
    current_season: str | None = Field(default=None, description="Name of the current season")


class CompetitionPage(Page):
    items: list[CompetitionRead]


class SeasonRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    competition_id: int
    competition: str | None
    name: str
    start_date: date | None
    end_date: date | None
    is_current: bool


class SeasonPage(Page):
    items: list[SeasonRead]


class TeamRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    short_name: str | None
    country: str | None


class TeamPage(Page):
    items: list[TeamRead]


class MatchTeam(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str


class MatchRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    kickoff_at: datetime
    status: str
    competition: MatchTeam = Field(description="Competition id and name")
    season: str | None
    matchday: int | None
    home_team: MatchTeam
    away_team: MatchTeam
    home_score: int | None
    away_score: int | None


class MatchDetail(MatchRead):
    venue: str | None
    referee: str | None
    half_time_score: dict[str, int] | None
    #: Predictions already stored for this match, newest first.
    predictions: list[dict[str, Any]] = Field(default_factory=list)


class MatchPage(Page):
    items: list[MatchRead]


class MatchFilters(BaseModel):
    """Query parameters shared by the match endpoints."""

    model_config = ConfigDict(frozen=True)

    competition_id: int | None = None
    team_id: int | None = None
    from_date: date | None = None
    to_date: date | None = None


class SideRecord(BaseModel):
    """A team's record at home or away."""

    model_config = ConfigDict(frozen=True)

    side: str
    played: int
    wins: int
    draws: int
    losses: int
    scored: int
    conceded: int
    clean_sheets: int
    goals_for_per_match: float | None
    goals_against_per_match: float | None


class TeamRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goals_for_per_match: float | None
    goals_against_per_match: float | None
    clean_sheets: int
    home: SideRecord
    away: SideRecord
    #: Most recent results first, e.g. ["W", "D", "L"].
    form: list[str]
    #: None when no imported match has expected goals.
    expected_goals_for_per_match: float | None
    expected_goals_against_per_match: float | None
    matches_with_expected_goals: int


class TeamMatchRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_id: int
    kickoff_at: datetime
    opponent: str
    is_home: bool
    goals_for: int
    goals_against: int
    result: str
    expected_goals_for: float | None
    rolling_goals_for: float
    rolling_goals_against: float


class TeamDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: TeamRead
    record: TeamRecord
    recent_matches: list[MatchRead]
    #: Oldest first, for the trend chart.
    trend: list[TeamMatchRead]
