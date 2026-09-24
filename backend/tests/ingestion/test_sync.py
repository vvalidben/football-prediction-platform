"""Synchronisation into PostgreSQL: idempotency, updates, failures and run logging."""

import threading
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.sync import ProviderSync
from app.models import (
    Competition,
    CompetitionExternalId,
    DataIngestionRun,
    DataSource,
    Match,
    MatchExternalId,
    Season,
    SeasonExternalId,
    Team,
    TeamExternalId,
)
from app.models.enums import IngestionStatus, MatchStatus
from app.providers.base import ProviderOperation, RejectedRecord
from app.providers.errors import ProviderUnavailableError, QuotaExceededError
from app.providers.mock import MockFootballDataProvider
from app.providers.schemas import ScorePair
from tests.providers import builders
from tests.providers.fake_api import FakeApiFootball, status, timeout

pytestmark = pytest.mark.integration

ENTITY_TABLES = (
    Competition,
    Season,
    Team,
    Match,
    CompetitionExternalId,
    SeasonExternalId,
    TeamExternalId,
    MatchExternalId,
)


def counts(factory: sessionmaker[Session]) -> dict[str, int]:
    with factory() as session:
        return {
            model.__tablename__: session.scalar(select(func.count()).select_from(model)) or 0
            for model in ENTITY_TABLES
        }


def runs(factory: sessionmaker[Session]) -> list[DataIngestionRun]:
    with factory() as session:
        return list(session.scalars(select(DataIngestionRun).order_by(DataIngestionRun.id)))


def outcome(run: Any) -> tuple[str, int, int, int, int, int]:
    return (run.job, run.fetched, run.inserted, run.updated, run.unchanged, run.failed)


@pytest.fixture
def mock_provider() -> MockFootballDataProvider:
    return MockFootballDataProvider(builders.dataset())


@pytest.fixture
def sync(
    mock_provider: MockFootballDataProvider, session_factory: sessionmaker[Session]
) -> ProviderSync:
    return ProviderSync(mock_provider, session_factory)


def stored_match(factory: sessionmaker[Session], external_id: str) -> Match:
    with factory() as session:
        return session.scalars(
            select(Match)
            .join(MatchExternalId, MatchExternalId.match_id == Match.id)
            .where(MatchExternalId.external_id == external_id)
        ).one()


class TestFirstSync:
    def test_inserts_competition_seasons_teams_and_matches(
        self, sync: ProviderSync, session_factory: sessionmaker[Session]
    ) -> None:
        results = sync.sync_competition(builders.COMPETITION)

        assert [outcome(r) for r in results] == [
            ("competitions", 1, 1, 0, 0, 0),
            ("seasons", 2, 2, 0, 0, 0),
            ("teams", 4, 4, 0, 0, 0),
            ("matches", 2, 2, 0, 0, 0),
        ]
        assert all(r.status is IngestionStatus.SUCCEEDED for r in results)
        assert counts(session_factory) == {
            "competitions": 1,
            "seasons": 2,
            "teams": 4,
            "matches": 2,
            "competition_external_ids": 1,
            "season_external_ids": 2,
            "team_external_ids": 4,
            "match_external_ids": 2,
        }

    def test_stores_normalised_values(
        self, sync: ProviderSync, session_factory: sessionmaker[Session]
    ) -> None:
        sync.sync_competition(builders.COMPETITION)

        match = stored_match(session_factory, "1035037")
        with session_factory() as session:
            match = session.merge(match)
            assert (match.home_team.name, match.away_team.name) == ("Burnley", "Manchester City")
            assert (match.home_score, match.away_score) == (0, 3)
            assert (match.home_half_time_score, match.away_half_time_score) == (0, 2)
            assert match.status is MatchStatus.FINISHED
            assert match.kickoff_at == datetime(2023, 8, 11, 19, 0, tzinfo=UTC)
            assert match.season.name == "2023/2024"
            assert match.season.is_current is True
            assert match.competition.name == "Premier League"
            source = session.scalars(select(DataSource)).one()
            assert (source.code, source.name) == ("mock", "Mock provider")

    def test_only_the_current_season_is_synced_by_default(
        self, sync: ProviderSync, mock_provider: MockFootballDataProvider
    ) -> None:
        sync.sync_competition(builders.COMPETITION)

        synced = {args for op, args in mock_provider.calls if op is ProviderOperation.MATCHES}
        assert synced == {(builders.COMPETITION, builders.SEASON)}


class TestIdempotency:
    def test_second_sync_creates_no_duplicates(
        self, sync: ProviderSync, session_factory: sessionmaker[Session]
    ) -> None:
        sync.sync_competition(builders.COMPETITION)
        before = counts(session_factory)

        second = sync.sync_competition(builders.COMPETITION)

        assert counts(session_factory) == before
        assert [outcome(r) for r in second] == [
            ("competitions", 1, 0, 0, 1, 0),
            ("seasons", 2, 0, 0, 2, 0),
            ("teams", 4, 0, 0, 4, 0),
            ("matches", 2, 0, 0, 2, 0),
        ]

    def test_new_sync_object_is_also_idempotent(
        self, mock_provider: MockFootballDataProvider, session_factory: sessionmaker[Session]
    ) -> None:
        ProviderSync(mock_provider, session_factory).sync_competition(builders.COMPETITION)
        before = counts(session_factory)

        ProviderSync(mock_provider, session_factory).sync_competition(builders.COMPETITION)

        assert counts(session_factory) == before
        with session_factory() as session:
            assert session.scalar(select(func.count()).select_from(DataSource)) == 1

    def test_api_football_adapter_end_to_end(self, session_factory: sessionmaker[Session]) -> None:
        provider = FakeApiFootball().provider()
        sync = ProviderSync(provider, session_factory)

        first = sync.sync_competition("39")
        after_first = counts(session_factory)
        second = sync.sync_competition("39")

        assert [outcome(r) for r in first] == [
            ("competitions", 1, 1, 0, 0, 0),
            ("seasons", 2, 2, 0, 0, 0),
            ("teams", 20, 20, 0, 0, 0),
            ("matches", 10, 10, 0, 0, 0),
        ]
        assert all(r.inserted == 0 and r.updated == 0 for r in second)
        assert counts(session_factory) == after_first
        assert after_first["matches"] == 10
        with session_factory() as session:
            newcastle_villa = session.merge(stored_match(session_factory, "1035043"))
            assert newcastle_villa.home_team.name == "Newcastle"
            assert (newcastle_villa.home_score, newcastle_villa.away_score) == (5, 1)
            assert newcastle_villa.venue == "St. James' Park"


class TestUpdates:
    def test_changed_match_is_updated_in_place(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        mock_provider.dataset.matches[0] = builders.match(
            status=MatchStatus.POSTPONED, score=None, half_time_score=None
        )
        sync.sync_competition(builders.COMPETITION)
        original = stored_match(session_factory, "1035037")

        rescheduled = datetime(2023, 9, 1, 19, 0, tzinfo=UTC)
        mock_provider.dataset.matches[0] = builders.match(kickoff_at=rescheduled)
        result = sync.sync_matches(builders.COMPETITION, builders.SEASON)

        assert outcome(result) == ("matches", 2, 0, 1, 1, 0)
        updated = stored_match(session_factory, "1035037")
        assert updated.id == original.id
        assert original.status is MatchStatus.POSTPONED
        assert updated.status is MatchStatus.FINISHED
        assert updated.kickoff_at == rescheduled
        assert (updated.home_score, updated.away_score) == (0, 3)

    def test_missing_values_do_not_erase_stored_data(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        sync.sync_competition(builders.COMPETITION)

        mock_provider.dataset.competitions[0] = builders.competition(logo_url=None)
        mock_provider.dataset.matches[0] = builders.match(venue=None, matchday=None)
        results = sync.sync_competition(builders.COMPETITION)

        assert all(r.updated == 0 for r in results)
        match = stored_match(session_factory, "1035037")
        assert (match.venue, match.matchday) == ("Turf Moor", 1)
        with session_factory() as session:
            assert session.scalars(select(Competition.logo_url)).one() == (
                "https://example.test/pl.png"
            )

    def test_current_season_flag_moves_without_conflict(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        sync.sync_competition(builders.COMPETITION)

        mock_provider.dataset.seasons = [
            builders.season(is_current=False),
            builders.season(
                external_id="PL-2024",
                name="2024/2025",
                start_date=date(2024, 8, 16),
                end_date=date(2025, 5, 25),
                is_current=True,
            ),
        ]
        result = sync.sync_seasons(builders.COMPETITION)

        assert result.status is IngestionStatus.SUCCEEDED
        with session_factory() as session:
            current = session.scalars(select(Season.name).where(Season.is_current)).all()
        assert current == ["2024/2025"]


class TestPartialFailures:
    def test_match_with_unknown_team_fails_alone(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        mock_provider.dataset.matches.append(
            builders.match(
                external_id="1035039",
                home_team_external_id="999",
                away_team_external_id="44",
                kickoff_at=datetime(2023, 8, 12, 14, 0, tzinfo=UTC),
                score=ScorePair(home=1, away=1),
            )
        )

        result = sync.sync_competition(builders.COMPETITION)[-1]

        assert result.status is IngestionStatus.PARTIAL
        assert outcome(result) == ("matches", 3, 2, 0, 0, 1)
        assert result.error_message is not None
        assert "[1035039] unknown team '999'" in result.error_message
        assert counts(session_factory)["matches"] == 2

    def test_malformed_provider_records_are_counted_as_failed(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        mock_provider.dataset.rejected[ProviderOperation.MATCHES] = [
            RejectedRecord("fixture.status.short: unknown status 'ZZZ'", "1035099")
        ]

        result = sync.sync_competition(builders.COMPETITION)[-1]

        assert result.status is IngestionStatus.PARTIAL
        assert outcome(result) == ("matches", 3, 2, 0, 0, 1)
        assert result.error_message is not None
        assert "malformed" in result.error_message
        stored = runs(session_factory)[-1]
        assert (stored.status, stored.records_failed) == (IngestionStatus.PARTIAL, 1)

    def test_error_message_is_bounded(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
    ) -> None:
        mock_provider.dataset.rejected[ProviderOperation.TEAMS] = [
            RejectedRecord("bad", str(n)) for n in range(25)
        ]

        result = sync.sync_competition(builders.COMPETITION)[2]

        assert result.failed == 25
        assert result.error_message is not None
        assert "and 15 more" in result.error_message


class TestProviderFailures:
    def test_failed_fetch_is_logged_and_writes_nothing(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        sync.sync_competitions()
        sync.sync_seasons(builders.COMPETITION)
        sync.sync_teams(builders.COMPETITION, builders.SEASON)
        mock_provider.fail_next(ProviderOperation.MATCHES, ProviderUnavailableError("gateway down"))

        with pytest.raises(ProviderUnavailableError):
            sync.sync_matches(builders.COMPETITION, builders.SEASON)

        failed = runs(session_factory)[-1]
        assert failed.job == "matches"
        assert failed.status is IngestionStatus.FAILED
        assert failed.error_message == "ProviderUnavailableError: gateway down"
        assert failed.finished_at is not None
        assert counts(session_factory)["matches"] == 0

    def test_failed_season_does_not_stop_the_next_one(
        self,
        sync: ProviderSync,
        mock_provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        mock_provider.fail_next(ProviderOperation.TEAMS, QuotaExceededError("daily quota"))

        results = sync.sync_competition(builders.COMPETITION, seasons="all")

        jobs = [(r.job, r.status) for r in results]
        assert jobs == [
            ("competitions", IngestionStatus.SUCCEEDED),
            ("seasons", IngestionStatus.SUCCEEDED),
            # 2022/2023 teams failed, so its matches were skipped …
            ("teams", IngestionStatus.SUCCEEDED),
            ("matches", IngestionStatus.SUCCEEDED),
        ]
        logged = [(r.job, r.status) for r in runs(session_factory)]
        assert ("teams", IngestionStatus.FAILED) in logged
        assert counts(session_factory)["matches"] == 2

    def test_retries_through_the_full_stack(self, session_factory: sessionmaker[Session]) -> None:
        api = FakeApiFootball()
        sleeps: list[float] = []
        provider = api.provider(sleeps=sleeps)
        api.fail_next(timeout(), status(503))

        result = ProviderSync(provider, session_factory).sync_competitions(only=["39"])

        assert result.status is IngestionStatus.SUCCEEDED
        assert result.inserted == 1
        assert len(sleeps) == 2
        assert len(api.requests) == 3


class TestRunLog:
    def test_every_operation_records_timing_provider_and_counts(
        self, sync: ProviderSync, session_factory: sessionmaker[Session]
    ) -> None:
        sync.sync_competition(builders.COMPETITION)

        logged = runs(session_factory)
        assert [r.job for r in logged] == ["competitions", "seasons", "teams", "matches"]
        with session_factory() as session:
            source_id = session.scalars(select(DataSource.id)).one()
        for run in logged:
            assert run.data_source_id == source_id
            assert run.status is IngestionStatus.SUCCEEDED
            assert run.finished_at is not None
            assert run.started_at <= run.finished_at
            assert run.error_message is None
        matches_run = logged[-1]
        assert matches_run.parameters == {
            "competition": builders.COMPETITION,
            "season": builders.SEASON,
        }
        assert (matches_run.records_fetched, matches_run.records_inserted) == (2, 2)


def test_existing_fixture_from_another_provider_is_linked_not_duplicated(
    sync: ProviderSync, session_factory: sessionmaker[Session]
) -> None:
    sync.sync_competition(builders.COMPETITION)
    other = MockFootballDataProvider(builders.dataset(), code="other_provider")
    other_sync = ProviderSync(other, session_factory)
    # Pretend the second provider's competition, seasons and teams were already
    # matched to the same canonical rows (cross-provider matching is future work).
    with session_factory() as session, session.begin():
        other_source = DataSource(code="other_provider", name="Other")
        session.add(other_source)
        session.flush()
        for model, fk in (
            (CompetitionExternalId, "competition_id"),
            (SeasonExternalId, "season_id"),
            (TeamExternalId, "team_id"),
        ):
            for mapping in session.scalars(select(model)).all():
                session.add(
                    model(
                        **{fk: getattr(mapping, fk)},
                        data_source_id=other_source.id,
                        external_id=mapping.external_id,
                    )
                )

    result = other_sync.sync_matches(builders.COMPETITION, builders.SEASON)

    assert (result.inserted, result.updated) == (0, 2)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Match)) == 2
        assert session.scalar(select(func.count()).select_from(MatchExternalId)) == 4


def test_fixture_stored_under_another_season_is_a_conflict(
    sync: ProviderSync,
    mock_provider: MockFootballDataProvider,
    session_factory: sessionmaker[Session],
) -> None:
    sync.sync_competition(builders.COMPETITION, seasons="all")
    # The provider now lists the same fixture (same teams and kickoff) under a
    # different external id and season: never silently move the stored match.
    mock_provider.dataset.matches = [
        builders.match(external_id="new-id", season_external_id=builders.PREVIOUS_SEASON)
    ]
    mock_provider.dataset.teams[builders.PREVIOUS_SEASON] = list(builders.TEAMS)

    result = sync.sync_matches(builders.COMPETITION, builders.PREVIOUS_SEASON)

    assert (result.status, result.failed) == (IngestionStatus.PARTIAL, 1)
    assert result.error_message is not None
    assert "another season" in result.error_message
    assert counts(session_factory)["matches"] == 2


def test_concurrent_syncs_of_one_provider_do_not_duplicate(db_engine: Engine) -> None:
    """Two syncs racing on separate connections: the advisory lock serialises them.

    Uses committed transactions (not the rolled-back fixture) so the race is real,
    then deletes what it created.
    """
    factory = sessionmaker(bind=db_engine, expire_on_commit=False)
    code = "concurrency_test"
    dataset = builders.dataset()
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            sync = ProviderSync(MockFootballDataProvider(dataset, code=code), factory)
            sync.sync_competitions()
            sync.sync_seasons(builders.COMPETITION)
            barrier.wait(timeout=10)
            sync.sync_teams(builders.COMPETITION, builders.SEASON)
        except BaseException as exc:  # surfaced to the main thread below
            errors.append(exc)

    # Both workers must use the same data source row; create it up front.
    ProviderSync(MockFootballDataProvider(dataset, code=code), factory).data_source_id()
    threads = [threading.Thread(target=worker) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert errors == []
        with factory() as session:
            source_id = session.scalars(select(DataSource.id).where(DataSource.code == code)).one()
            team_mappings = session.scalar(
                select(func.count())
                .select_from(TeamExternalId)
                .where(TeamExternalId.data_source_id == source_id)
            )
            team_runs = session.scalars(
                select(DataIngestionRun).where(
                    DataIngestionRun.data_source_id == source_id, DataIngestionRun.job == "teams"
                )
            ).all()
        assert team_mappings == len(builders.TEAMS)
        assert sorted(run.records_inserted for run in team_runs) == [0, len(builders.TEAMS)]
        assert [(run.status, run.records_failed) for run in team_runs] == [
            (IngestionStatus.SUCCEEDED, 0),
            (IngestionStatus.SUCCEEDED, 0),
        ]
    finally:
        _delete_data_source(factory, code)


def _delete_data_source(factory: sessionmaker[Session], code: str) -> None:
    with factory() as session, session.begin():
        source_id = session.scalars(select(DataSource.id).where(DataSource.code == code)).first()
        if source_id is None:
            return
        for mapping, entity, fk in (
            (TeamExternalId, Team, TeamExternalId.team_id),
            (SeasonExternalId, Season, SeasonExternalId.season_id),
            (CompetitionExternalId, Competition, CompetitionExternalId.competition_id),
        ):
            ids = select(fk).where(mapping.data_source_id == source_id).scalar_subquery()
            session.execute(delete(entity).where(entity.id.in_(ids)))
        session.execute(
            delete(DataIngestionRun).where(DataIngestionRun.data_source_id == source_id)
        )
        session.execute(delete(DataSource).where(DataSource.id == source_id))
