"""Convert football-data.org payloads into provider-neutral records."""

from datetime import UTC

from app.models.enums import CompetitionType, MatchStatus
from app.providers.football_data import payloads
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ScorePair,
)

# https://docs.football-data.org/general/v4/match.html
STATUS_MAP: dict[str, MatchStatus] = {
    "SCHEDULED": MatchStatus.SCHEDULED,
    "TIMED": MatchStatus.SCHEDULED,  # scheduled, with the kickoff time confirmed
    "IN_PLAY": MatchStatus.LIVE,
    "PAUSED": MatchStatus.LIVE,  # half time
    "EXTRA_TIME": MatchStatus.LIVE,
    "PENALTY_SHOOTOUT": MatchStatus.LIVE,
    "LIVE": MatchStatus.LIVE,
    "FINISHED": MatchStatus.FINISHED,
    "AWARDED": MatchStatus.FINISHED,
    "SUSPENDED": MatchStatus.SUSPENDED,
    "POSTPONED": MatchStatus.POSTPONED,
    "CANCELLED": MatchStatus.CANCELLED,
}
COMPETITION_TYPES: dict[str, CompetitionType] = {
    "LEAGUE": CompetitionType.LEAGUE,
    "CUP": CompetitionType.CUP,
    "SUPER_CUP": CompetitionType.SUPER_CUP,
}
#: Areas that are continents or the world, so a competition there has no country.
INTERNATIONAL_AREAS = frozenset(
    {"World", "Europe", "South America", "North America", "Africa", "Asia", "Oceania"}
)


def season_external_id(competition_id: int | str, year: int) -> str:
    """The API addresses a season by its starting year, which is not unique across
    competitions, so the competition is part of our external ID: "2021:2025"."""
    return f"{competition_id}:{year}"


def parse_season_external_id(value: str) -> tuple[str, int]:
    competition, _, year = value.partition(":")
    if not (competition.isdigit() and year.isdigit()):
        raise ValueError(f"not a football-data.org season id: {value!r}")
    return competition, int(year)


def competition(raw: payloads.Competition) -> ProviderCompetition:
    area = raw.area.name if raw.area else None
    return ProviderCompetition(
        external_id=str(raw.id),
        name=raw.name,
        type=COMPETITION_TYPES.get(raw.type or "", CompetitionType.OTHER),
        country=None if area in INTERNATIONAL_AREAS else area,
        logo_url=raw.emblem,
    )


def season(competition_id: int, raw: payloads.Season, *, current_id: int | None) -> ProviderSeason:
    start, end = raw.startDate, raw.endDate
    name = f"{start.year}/{end.year}" if start.year != end.year else str(start.year)
    return ProviderSeason(
        external_id=season_external_id(competition_id, start.year),
        competition_external_id=str(competition_id),
        name=name,
        start_date=start,
        end_date=end,
        is_current=raw.id == current_id,
        statistics_available=False,  # this API has no per-match statistics on this adapter
    )


def team(raw: payloads.Team) -> ProviderTeam:
    country = raw.area.name if raw.area else None
    return ProviderTeam(
        external_id=str(raw.id),
        name=raw.name,
        short_name=raw.shortName,
        code=raw.tla,
        country=country,
        logo_url=raw.crest,
        is_national_team=raw.name == country,
    )


def _pair(goals: payloads.Goals | None) -> ScorePair | None:
    if goals is None or goals.home is None or goals.away is None:
        return None
    return ScorePair(home=goals.home, away=goals.away)


def match_score(score: payloads.Score) -> ScorePair | None:
    """The score after regular and extra time.

    `fullTime` is that score, except after a shootout, when the API adds the
    penalties to it: a match that ended 0-1 and went 1-4 on penalties reads 1-5.
    """
    full = _pair(score.fullTime)
    penalties = _pair(score.penalties)
    if full is None or penalties is None:
        return full
    if penalties.home > full.home or penalties.away > full.away:
        raise ValueError("penalties exceed the full-time score")
    return ScorePair(home=full.home - penalties.home, away=full.away - penalties.away)


def match(raw: payloads.Match, competition_id: str, season_id: str) -> ProviderMatch:
    status = STATUS_MAP.get(raw.status)
    if status is None:
        raise ValueError(f"unknown match status {raw.status!r}")
    referee = next((r.name for r in raw.referees if r.type == "REFEREE"), None)
    return ProviderMatch(
        external_id=str(raw.id),
        competition_external_id=competition_id,
        season_external_id=season_id,
        home_team_external_id=str(raw.homeTeam.id),
        away_team_external_id=str(raw.awayTeam.id),
        kickoff_at=raw.utcDate.astimezone(UTC),
        status=status,
        matchday=raw.matchday if raw.matchday and raw.matchday > 0 else None,
        stage=raw.stage.lower() if raw.stage else None,
        score=match_score(raw.score),
        half_time_score=_pair(raw.score.halfTime),
        penalty_score=_pair(raw.score.penalties),
        venue=raw.venue,
        referee=referee,
    )
