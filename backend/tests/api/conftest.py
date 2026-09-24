"""A running API backed by a real database, a real model and real features."""

import random
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis import Redis
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.features.__main__ import main as features_main
from app.features.config import FeatureConfig
from app.main import create_app
from app.models import Competition, Match, ModelVersion, Season, Team
from app.models.enums import CompetitionType, MatchStatus
from app.registry.loading import ModelCache
from app.registry.service import ModelRegistry, NewModelVersion
from app.services.predictions import PredictionService

pytestmark = pytest.mark.integration

SEASONS = (2023, 2024, 2025)
TEAMS = ("Arsenal", "Chelsea", "Liverpool", "Everton", "Fulham", "Brentford")


@pytest.fixture
def api_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        postgres_host="db.invalid",
        postgres_user="test_user",
        postgres_password=SecretStr("test-password"),
        postgres_db="test_db",
        redis_host="redis.invalid",
    )


@pytest.fixture
def league(session_factory: sessionmaker[Session]) -> dict[str, int]:
    """Three finished seasons plus a scheduled season, results driven by team strength."""
    rng = random.Random(11)
    ids: dict[str, int] = {}
    with session_factory() as session, session.begin():
        competition = Competition(
            name="Premier League", type=CompetitionType.LEAGUE, country="England"
        )
        cup = Competition(name="FA Cup", type=CompetitionType.CUP, country="England")
        teams = [Team(name=name, short_name=name[:3].upper(), country="England") for name in TEAMS]
        strength = {name: rng.gauss(0, 0.4) for name in TEAMS}
        session.add_all([competition, cup, *teams])
        session.flush()
        ids["competition"] = competition.id
        ids["cup"] = cup.id
        for team in teams:
            ids[team.name] = team.id
        for year in SEASONS:
            season = Season(
                competition=competition,
                name=f"{year}/{year + 1}",
                start_date=date(year, 8, 1),
                end_date=date(year + 1, 5, 31),
                is_current=year == SEASONS[-1],
            )
            kickoff = datetime(year, 8, 5, 15, tzinfo=UTC)
            for home in teams:
                for away in teams:
                    if home is away:
                        continue
                    gap = strength[home.name] - strength[away.name]
                    session.add(
                        Match(
                            competition=competition,
                            season=season,
                            home_team=home,
                            away_team=away,
                            kickoff_at=kickoff,
                            status=MatchStatus.FINISHED,
                            home_score=_poisson(rng, 1.5 * 2.718**gap),
                            away_score=_poisson(rng, 1.1 * 2.718**-gap),
                        )
                    )
                    kickoff += timedelta(days=3)
        # Two scheduled matches: one soon, one later, plus a cup tie for filtering.
        future = Season(
            competition=competition,
            name="2026/2027",
            start_date=date(2026, 8, 1),
            end_date=date(2027, 5, 31),
        )
        cup_season = Season(
            competition=cup,
            name="2026/2027",
            start_date=date(2026, 8, 1),
            end_date=date(2027, 5, 31),
        )
        session.add_all([future, cup_season])
        now = datetime.now(UTC)
        upcoming = [
            Match(
                competition=competition,
                season=future,
                home_team=teams[0],
                away_team=teams[1],
                kickoff_at=now + timedelta(days=3),
                status=MatchStatus.SCHEDULED,
                matchday=1,
                venue="Emirates Stadium",
            ),
            Match(
                competition=competition,
                season=future,
                home_team=teams[2],
                away_team=teams[3],
                kickoff_at=now + timedelta(days=10),
                status=MatchStatus.SCHEDULED,
            ),
            Match(
                competition=cup,
                season=cup_season,
                home_team=teams[0],
                away_team=teams[1],
                kickoff_at=now + timedelta(days=20),
                status=MatchStatus.SCHEDULED,
            ),
        ]
        session.add_all(upcoming)
        session.flush()
        ids["next_match"] = upcoming[0].id
        ids["later_match"] = upcoming[1].id
        ids["cup_match"] = upcoming[2].id
    with session_factory() as session:
        # The most recent finished match: it has earlier results behind it.
        finished = (
            session.query(Match)
            .filter(Match.status == MatchStatus.FINISHED)
            .order_by(Match.kickoff_at.desc())
            .first()
        )
        assert finished is not None
        ids["finished_match"] = finished.id
    features_main(["generate"], session_factory=session_factory)
    return ids


def _poisson(rng: random.Random, rate: float) -> int:
    count, total = 0, rng.expovariate(rate)
    while total < 1.0:
        count += 1
        total += rng.expovariate(rate)
    return count


@pytest.fixture
def active_model(
    league: dict[str, int], session_factory: sessionmaker[Session], tmp_path: Path
) -> ModelVersion:
    """A real Elo model, fitted on the league, saved, registered and activated."""
    from app.ml.dataset import load_dataset
    from app.ml.predictors import EloModel

    version = FeatureConfig().version()
    with session_factory() as session:
        data = load_dataset(session, version)
    model = EloModel().fit(data)
    directory = tmp_path / "elo-model"
    model.save(directory, {"feature_set_version": version})
    with session_factory() as session, session.begin():
        row = ModelRegistry(session).register(
            NewModelVersion(
                name="elo",
                version="v1",
                algorithm="EloModel",
                feature_set_version=version,
                dataset_version=data.fingerprint(),
                training_data_start=data.examples[0].kickoff_at.date(),
                training_data_end=data.examples[-1].kickoff_at.date(),
                competitions=[{"id": league["competition"], "name": "Premier League"}],
                hyperparameters={"feature": "elo_diff"},
                metrics={"validation": {"log_loss": 0.98, "accuracy": 0.52}},
                artifact_uri=directory.as_uri(),
                trained_at=datetime.now(UTC),
                description="test model",
            )
        )
        ModelRegistry(session).activate("elo", "v1", reason="tests")
        session.flush()
        model_id = row.id
    with session_factory() as session:
        registered = session.get(ModelVersion, model_id)
        assert registered is not None
        session.expunge(registered)
        return registered


@pytest.fixture
def client(
    api_settings: Settings, db_engine: Engine, session_factory: sessionmaker[Session]
) -> Iterator[TestClient]:
    """The real application, wired to the test database (no lifespan, no Redis)."""
    app = create_app(api_settings)
    app.state.settings = api_settings
    app.state.engine = db_engine
    app.state.session_factory = session_factory
    app.state.redis = MagicMock(spec=Redis)
    app.state.model_cache = ModelCache()
    app.state.predictions = PredictionService(session_factory, cache=app.state.model_cache)
    # Not used as a context manager on purpose: the lifespan would replace this
    # wiring with a real engine and Redis client.
    yield TestClient(app, raise_server_exceptions=False)
