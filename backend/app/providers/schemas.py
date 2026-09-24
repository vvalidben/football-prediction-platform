"""Provider-neutral records returned by every `FootballDataProvider`.

Adapters convert provider responses into these types; ingestion converts these into
database rows. Nothing downstream of the adapters (ingestion, and later features and
models) ever sees a provider's own format.

All identifiers here are the provider's *external* IDs, as strings. Resolving them to
internal rows is the ingestion layer's job.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)

from app.models.enums import (
    AbsenceType,
    AvailabilityStatus,
    CompetitionType,
    MatchEventType,
    MatchPeriod,
    MatchStatus,
    OddsMarket,
    PlayerPosition,
)

ExternalId = str


class ProviderRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ProviderCompetition(ProviderRecord):
    external_id: ExternalId = Field(min_length=1)
    name: str = Field(min_length=1)
    type: CompetitionType
    country: str | None = None  # None for international competitions
    logo_url: str | None = None


class ProviderSeason(ProviderRecord):
    external_id: ExternalId = Field(min_length=1)
    competition_external_id: ExternalId = Field(min_length=1)
    name: str = Field(min_length=1)  # e.g. "2023/2024"
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool = False
    #: The provider's own claim that match statistics exist for this season;
    #: None when the provider does not say.
    statistics_available: bool | None = None

    @model_validator(mode="after")
    def _dates_ordered(self) -> Self:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("season end_date is before start_date")
        return self


class ProviderTeam(ProviderRecord):
    external_id: ExternalId = Field(min_length=1)
    name: str = Field(min_length=1)
    short_name: str | None = None
    code: str | None = None
    country: str | None = None
    logo_url: str | None = None
    is_national_team: bool = False


class ProviderPlayer(ProviderRecord):
    external_id: ExternalId = Field(min_length=1)
    name: str = Field(min_length=1)
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    nationality: str | None = None
    position: PlayerPosition | None = None


class ScorePair(ProviderRecord):
    home: NonNegativeInt
    away: NonNegativeInt


class ProviderMatch(ProviderRecord):
    external_id: ExternalId = Field(min_length=1)
    competition_external_id: ExternalId = Field(min_length=1)
    season_external_id: ExternalId = Field(min_length=1)
    home_team_external_id: ExternalId = Field(min_length=1)
    away_team_external_id: ExternalId = Field(min_length=1)
    kickoff_at: AwareDatetime
    status: MatchStatus
    matchday: int | None = Field(default=None, gt=0)
    stage: str | None = None
    score: ScorePair | None = None  # after regular and extra time
    half_time_score: ScorePair | None = None
    penalty_score: ScorePair | None = None
    venue: str | None = None
    referee: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.home_team_external_id == self.away_team_external_id:
            raise ValueError("home and away team are the same")
        if self.status is MatchStatus.FINISHED and self.score is None:
            raise ValueError("finished match has no score")
        return self


class ProviderMatchEvent(ProviderRecord):
    external_id: ExternalId | None = None
    type: MatchEventType
    period: MatchPeriod | None = None
    minute: NonNegativeInt
    added_time: int | None = Field(default=None, ge=0)
    team_external_id: ExternalId | None = None
    player_external_id: ExternalId | None = None
    related_player_external_id: ExternalId | None = None
    detail: str | None = None


class ProviderMatchDetails(ProviderRecord):
    match: ProviderMatch
    events: tuple[ProviderMatchEvent, ...] = ()


Count = Annotated[int | None, Field(ge=0)]


class ProviderTeamMatchStatistics(ProviderRecord):
    """One team's statistics for one match. None means "not reported", not zero."""

    match_external_id: ExternalId = Field(min_length=1)
    team_external_id: ExternalId = Field(min_length=1)
    possession: Decimal | None = Field(default=None, ge=0, le=100)
    shots: Count = None
    shots_on_target: Count = None
    shots_off_target: Count = None
    blocked_shots: Count = None
    shots_inside_box: Count = None
    corners: Count = None
    fouls: Count = None
    offsides: Count = None
    yellow_cards: Count = None
    red_cards: Count = None
    saves: Count = None
    passes: Count = None
    passes_completed: Count = None
    expected_goals: Decimal | None = Field(default=None, ge=0)
    #: Provider statistics without a column of their own.
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def external_id(self) -> str:
        return f"{self.match_external_id}:{self.team_external_id}"

    def reported(self) -> dict[str, Any]:
        """The statistics the provider actually supplied."""
        ids = {"match_external_id", "team_external_id", "extra"}
        return {k: v for k, v in self.model_dump().items() if k not in ids and v is not None}


class ProviderPlayerMatchStatistics(ProviderRecord):
    match_external_id: ExternalId
    team_external_id: ExternalId
    player: ProviderPlayer
    minutes_played: int | None = Field(default=None, ge=0)
    goals: int | None = Field(default=None, ge=0)
    assists: int | None = Field(default=None, ge=0)
    shots: int | None = Field(default=None, ge=0)
    expected_goals: Decimal | None = Field(default=None, ge=0)
    rating: Decimal | None = Field(default=None, ge=0, le=10)
    extra: dict[str, Any] = Field(default_factory=dict)


class ProviderLineupPlayer(ProviderRecord):
    player: ProviderPlayer
    is_starter: bool
    position: str | None = None
    shirt_number: int | None = Field(default=None, ge=0, le=99)


class ProviderLineup(ProviderRecord):
    match_external_id: ExternalId
    team_external_id: ExternalId
    formation: str | None = None
    coach_name: str | None = None
    players: tuple[ProviderLineupPlayer, ...] = ()


class ProviderInjury(ProviderRecord):
    player: ProviderPlayer
    team_external_id: ExternalId | None = None
    match_external_id: ExternalId | None = None
    absence_type: AbsenceType
    status: AvailabilityStatus
    reason: str | None = None
    start_date: date | None = None


class ProviderOdds(ProviderRecord):
    match_external_id: ExternalId
    bookmaker: str = Field(min_length=1)
    market: OddsMarket
    selection: str = Field(min_length=1)
    line: Decimal | None = None
    price: Decimal = Field(gt=1)
    recorded_at: AwareDatetime
