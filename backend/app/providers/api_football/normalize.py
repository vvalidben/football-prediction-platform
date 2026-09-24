"""Convert API-Football payloads into provider-neutral records."""

import re
from datetime import UTC
from decimal import Decimal, InvalidOperation
from typing import Any

from app.models.enums import CompetitionType, MatchStatus
from app.providers.api_football import payloads
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
    ScorePair,
)

# https://www.api-football.com/documentation-v3#tag/Fixtures (fixture.status.short)
STATUS_MAP: dict[str, MatchStatus] = {
    "TBD": MatchStatus.SCHEDULED,  # time to be defined
    "NS": MatchStatus.SCHEDULED,  # not started
    "1H": MatchStatus.LIVE,
    "HT": MatchStatus.LIVE,
    "2H": MatchStatus.LIVE,
    "ET": MatchStatus.LIVE,
    "BT": MatchStatus.LIVE,  # break before extra time
    "P": MatchStatus.LIVE,  # penalty shootout in progress
    "LIVE": MatchStatus.LIVE,
    "INT": MatchStatus.LIVE,  # interrupted, expected to resume shortly
    "SUSP": MatchStatus.SUSPENDED,
    "FT": MatchStatus.FINISHED,
    "AET": MatchStatus.FINISHED,
    "PEN": MatchStatus.FINISHED,
    "AWD": MatchStatus.FINISHED,  # technical loss / awarded
    "WO": MatchStatus.FINISHED,  # walkover
    "PST": MatchStatus.POSTPONED,
    "CANC": MatchStatus.CANCELLED,
    "ABD": MatchStatus.ABANDONED,
}

SUPER_CUP_MARKERS = ("super cup", "supercup", "supercopa", "supercoppa", "community shield")
ROUND_PATTERN = re.compile(r"^(?P<stage>.+?)\s+-\s+(?P<number>\d+)$")


def season_external_id(league_id: int, year: int) -> str:
    """API-Football identifies a season by year only, which is not unique across
    competitions, so the league is part of our external ID: "39:2023"."""
    return f"{league_id}:{year}"


def parse_season_external_id(value: str) -> tuple[int, int]:
    league, _, year = value.partition(":")
    if not (league.isdigit() and year.isdigit()):
        raise ValueError(f"not an API-Football season id: {value!r}")
    return int(league), int(year)


def competition_type(league: payloads.League) -> CompetitionType:
    name = league.name.lower()
    if "friendl" in name:
        return CompetitionType.FRIENDLY
    if any(marker in name for marker in SUPER_CUP_MARKERS):
        return CompetitionType.SUPER_CUP
    return {"league": CompetitionType.LEAGUE, "cup": CompetitionType.CUP}.get(
        league.type.lower(), CompetitionType.OTHER
    )


def competition(item: payloads.LeagueItem) -> ProviderCompetition:
    country = item.country.name
    return ProviderCompetition(
        external_id=str(item.league.id),
        name=item.league.name,
        type=competition_type(item.league),
        country=None if country in (None, "World") else country,
        logo_url=item.league.logo,
    )


def season_name(season: payloads.Season) -> str:
    if season.start and season.end and season.start.year != season.end.year:
        return f"{season.start.year}/{season.end.year}"
    return str(season.year)


def season(league_id: int, raw: payloads.Season) -> ProviderSeason:
    return ProviderSeason(
        external_id=season_external_id(league_id, raw.year),
        competition_external_id=str(league_id),
        name=season_name(raw),
        start_date=raw.start,
        end_date=raw.end,
        is_current=raw.current,
        statistics_available=(
            raw.coverage.fixtures.statistics_fixtures if raw.coverage is not None else None
        ),
    )


def team(item: payloads.TeamItem) -> ProviderTeam:
    raw = item.team
    return ProviderTeam(
        external_id=str(raw.id),
        name=raw.name,
        code=raw.code,
        country=raw.country,
        logo_url=raw.logo,
        is_national_team=raw.national,
    )


def _pair(goals: payloads.Goals) -> ScorePair | None:
    if goals.home is None or goals.away is None:
        return None
    return ScorePair(home=goals.home, away=goals.away)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_round(round_name: str | None) -> tuple[str | None, int | None]:
    """ "Regular Season - 12" -> ("regular_season", 12); "Final" -> ("final", None)."""
    if not round_name:
        return None, None
    match = ROUND_PATTERN.match(round_name.strip())
    if match:
        number = int(match["number"])
        return _slug(match["stage"]), number if number > 0 else None
    return _slug(round_name), None


def match(item: payloads.FixtureItem) -> ProviderMatch:
    status_code = item.fixture.status.short
    status = STATUS_MAP.get(status_code)
    if status is None:
        raise ValueError(f"unknown fixture status {status_code!r}")
    stage, matchday = parse_round(item.league.round)
    return ProviderMatch(
        external_id=str(item.fixture.id),
        competition_external_id=str(item.league.id),
        season_external_id=season_external_id(item.league.id, item.league.season),
        home_team_external_id=str(item.teams.home.id),
        away_team_external_id=str(item.teams.away.id),
        kickoff_at=item.fixture.date.astimezone(UTC),
        status=status,
        matchday=matchday,
        stage=stage,
        score=_pair(item.goals),
        half_time_score=_pair(item.score.halftime),
        penalty_score=_pair(item.score.penalty),
        venue=item.fixture.venue.name,
        referee=item.fixture.referee,
    )


# `type` labels of fixture statistics → our columns. Anything else goes to `extra`.
STATISTIC_COLUMNS: dict[str, str] = {
    "shots on goal": "shots_on_target",
    "shots off goal": "shots_off_target",
    "total shots": "shots",
    "blocked shots": "blocked_shots",
    "shots insidebox": "shots_inside_box",
    "fouls": "fouls",
    "corner kicks": "corners",
    "offsides": "offsides",
    "ball possession": "possession",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "goalkeeper saves": "saves",
    "total passes": "passes",
    "passes accurate": "passes_completed",
    "expected_goals": "expected_goals",
}
DECIMAL_COLUMNS = frozenset({"possession", "expected_goals"})


def _statistic_value(column: str | None, value: Any) -> Any:
    """ "55%" → Decimal("55"); "1.73" → Decimal("1.73"); "7" → 7; None stays None."""
    if value is None:
        return None
    if column is None:  # kept verbatim in `extra`
        return value
    text = str(value).strip().rstrip("%").strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"{column}: not a number: {value!r}") from None
    if column in DECIMAL_COLUMNS:
        return number
    if number != number.to_integral_value():
        raise ValueError(f"{column}: not a whole number: {value!r}")
    return int(number)


def team_statistics(item: payloads.FixtureDetailItem) -> list[ProviderTeamMatchStatistics]:
    """One record per team that has statistics; [] when the fixture has none."""
    records = []
    for block in item.statistics:
        if not block.statistics:
            continue
        values: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for entry in block.statistics:
            label = entry.type.strip().lower()
            column = STATISTIC_COLUMNS.get(label)
            if column is None:
                if entry.value is not None:
                    extra[_slug(label)] = entry.value
                continue
            values[column] = _statistic_value(column, entry.value)
        records.append(
            ProviderTeamMatchStatistics(
                match_external_id=str(item.fixture.id),
                team_external_id=str(block.team.id),
                extra=extra,
                **values,
            )
        )
    return records
