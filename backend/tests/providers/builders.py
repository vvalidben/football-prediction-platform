"""Provider-neutral records for tests of the mock provider and ingestion.

Values mirror the API-Football fixtures (Premier League 2023/24 matchday 1), so the
same real results appear whichever provider a test uses.
"""

from datetime import UTC, date, datetime
from typing import Any

from app.models.enums import CompetitionType, MatchStatus
from app.providers.mock import MockDataset
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ScorePair,
)

COMPETITION = "PL"
SEASON = "PL-2023"
PREVIOUS_SEASON = "PL-2022"


def competition(**overrides: Any) -> ProviderCompetition:
    values: dict[str, Any] = {
        "external_id": COMPETITION,
        "name": "Premier League",
        "type": CompetitionType.LEAGUE,
        "country": "England",
        "logo_url": "https://example.test/pl.png",
    }
    return ProviderCompetition(**{**values, **overrides})


def season(**overrides: Any) -> ProviderSeason:
    values: dict[str, Any] = {
        "external_id": SEASON,
        "competition_external_id": COMPETITION,
        "name": "2023/2024",
        "start_date": date(2023, 8, 11),
        "end_date": date(2024, 5, 19),
        "is_current": True,
    }
    return ProviderSeason(**{**values, **overrides})


def team(external_id: str, name: str, **overrides: Any) -> ProviderTeam:
    values: dict[str, Any] = {"country": "England", **overrides}
    return ProviderTeam(external_id=external_id, name=name, **values)


TEAMS = [
    team("44", "Burnley"),
    team("50", "Manchester City"),
    team("42", "Arsenal"),
    team("65", "Nottingham Forest"),
]


def match(**overrides: Any) -> ProviderMatch:
    values: dict[str, Any] = {
        "external_id": "1035037",
        "competition_external_id": COMPETITION,
        "season_external_id": SEASON,
        "home_team_external_id": "44",
        "away_team_external_id": "50",
        "kickoff_at": datetime(2023, 8, 11, 19, 0, tzinfo=UTC),
        "status": MatchStatus.FINISHED,
        "matchday": 1,
        "stage": "regular_season",
        "score": ScorePair(home=0, away=3),
        "half_time_score": ScorePair(home=0, away=2),
        "venue": "Turf Moor",
    }
    return ProviderMatch(**{**values, **overrides})


MATCHES = [
    match(),
    match(
        external_id="1035038",
        home_team_external_id="42",
        away_team_external_id="65",
        kickoff_at=datetime(2023, 8, 12, 11, 30, tzinfo=UTC),
        score=ScorePair(home=2, away=1),
        half_time_score=ScorePair(home=2, away=0),
        venue="Emirates Stadium",
    ),
]


def dataset() -> MockDataset:
    return MockDataset(
        competitions=[competition()],
        seasons=[
            season(
                external_id=PREVIOUS_SEASON,
                name="2022/2023",
                start_date=date(2022, 8, 5),
                end_date=date(2023, 5, 28),
                is_current=False,
            ),
            season(),
        ],
        teams={SEASON: list(TEAMS)},
        matches=list(MATCHES),
    )
