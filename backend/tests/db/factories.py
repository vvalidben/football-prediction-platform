"""Small builders for valid rows. Each call returns a new, unique, flushed object."""

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import count

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Competition,
    DataSource,
    Match,
    ModelVersion,
    Player,
    Prediction,
    Season,
    Team,
)
from app.models.enums import CompetitionType, MatchStatus

_sequence = count(1)
KICKOFF = datetime(2025, 8, 16, 14, 0, tzinfo=UTC)


def _next() -> int:
    return next(_sequence)


def data_source(session: Session, code: str | None = None) -> DataSource:
    n = _next()
    source = DataSource(code=code or f"provider_{n}", name=f"Provider {n}")
    session.add(source)
    session.flush()
    return source


def competition(session: Session) -> Competition:
    comp = Competition(name=f"League {_next()}", type=CompetitionType.LEAGUE, country="England")
    session.add(comp)
    session.flush()
    return comp


def season(session: Session, comp: Competition | None = None) -> Season:
    season_row = Season(
        competition=comp or competition(session),
        name=f"Season {_next()}",
        start_date=date(2025, 8, 1),
        end_date=date(2026, 5, 31),
    )
    session.add(season_row)
    session.flush()
    return season_row


def team(session: Session, name: str | None = None) -> Team:
    team_row = Team(name=name or f"Team {_next()}", country="England")
    session.add(team_row)
    session.flush()
    return team_row


def player(session: Session, current_team: Team | None = None) -> Player:
    player_row = Player(name=f"Player {_next()}", current_team=current_team)
    session.add(player_row)
    session.flush()
    return player_row


def match(
    session: Session,
    season_row: Season | None = None,
    home: Team | None = None,
    away: Team | None = None,
    **overrides: object,
) -> Match:
    season_row = season_row or season(session)
    values: dict[str, object] = {
        "competition": season_row.competition,
        "season": season_row,
        "home_team": home or team(session),
        "away_team": away or team(session),
        "kickoff_at": KICKOFF + timedelta(minutes=_next()),
        "status": MatchStatus.SCHEDULED,
        **overrides,
    }
    match_row = Match(**values)
    session.add(match_row)
    session.flush()
    return match_row


def model_version(session: Session, name: str = "poisson_baseline") -> ModelVersion:
    version = ModelVersion(
        name=name, version=f"1.0.{_next()}", algorithm="PoissonModel", dataset_version="ds-test"
    )
    session.add(version)
    session.flush()
    return version


def prediction(
    session: Session,
    match_row: Match | None = None,
    version: ModelVersion | None = None,
    **overrides: object,
) -> Prediction:
    values: dict[str, object] = {
        "match": match_row or match(session),
        "model_version": version or model_version(session),
        "predicted_at": KICKOFF - timedelta(days=1, minutes=_next()),
        "predicted_home_goals": 1,
        "predicted_away_goals": 1,
        "expected_home_goals": Decimal("1.420"),
        "expected_away_goals": Decimal("1.130"),
        "home_win_probability": Decimal("0.420000"),
        "draw_probability": Decimal("0.270000"),
        "away_win_probability": Decimal("0.310000"),
        "btts_probability": Decimal("0.520000"),
        **overrides,
    }
    prediction_row = Prediction(**values)
    session.add(prediction_row)
    session.flush()
    return prediction_row


def violated_constraint(session: Session, action: Callable[[], object]) -> str:
    """Run `action` and flush inside a savepoint; return the constraint that failed.

    Fails the test if no IntegrityError is raised. The savepoint is rolled back, so
    the session stays usable afterwards.
    """
    with pytest.raises(IntegrityError) as exc_info, session.begin_nested():
        action()
        session.flush()
    name: str = exc_info.value.orig.diag.constraint_name  # type: ignore[union-attr]
    return name


def assert_violates(session: Session, constraint: str, build: Callable[[], object]) -> None:
    """Assert that inserting the row returned by `build` fails on the named constraint.

    The row is built inside the savepoint: a row linked to persistent parents joins
    their relationship collections as soon as it is constructed, so building it
    earlier would leave an unflushable orphan behind after the rollback.
    """
    assert violated_constraint(session, lambda: session.add(build())) == constraint
