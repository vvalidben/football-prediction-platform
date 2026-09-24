"""Data-quality issues raised while syncing: detected, recorded, never silent."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.sync import ProviderSync
from app.models import DataQualityIssue, Match, Season, TeamMatchStatistics
from app.models.enums import IssueEntity, IssueSeverity, IssueType, MatchStatus
from app.providers.base import FetchResult, ProviderOperation, RejectedRecord
from app.providers.mock import MockFootballDataProvider
from app.providers.schemas import ProviderMatch, ProviderTeamMatchStatistics, ScorePair
from tests.providers import builders

pytestmark = pytest.mark.integration


def issues(
    factory: sessionmaker[Session], *, open_only: bool = True, **filters: object
) -> list[DataQualityIssue]:
    query = select(DataQualityIssue).order_by(DataQualityIssue.id)
    if open_only:
        query = query.where(DataQualityIssue.resolved_at.is_(None))
    for name, value in filters.items():
        query = query.where(getattr(DataQualityIssue, name) == value)
    with factory() as session:
        return list(session.scalars(query))


def count(factory: sessionmaker[Session], model: type) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def provider() -> MockFootballDataProvider:
    return MockFootballDataProvider(builders.dataset())


@pytest.fixture
def sync(
    provider: MockFootballDataProvider, session_factory: sessionmaker[Session]
) -> ProviderSync:
    return ProviderSync(provider, session_factory, clock=lambda: datetime(2026, 9, 22, tzinfo=UTC))


def test_clean_sync_records_no_issues(
    sync: ProviderSync, session_factory: sessionmaker[Session]
) -> None:
    sync.sync_competition(builders.COMPETITION)

    assert issues(session_factory) == []


class TestDetection:
    def test_missing_team(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches.append(
            builders.match(
                external_id="m-unknown",
                home_team_external_id="999",
                kickoff_at=datetime(2023, 8, 20, 14, tzinfo=UTC),
            )
        )

        sync.sync_competition(builders.COMPETITION)

        (issue,) = issues(session_factory, issue_type=IssueType.MISSING_TEAM)
        assert (issue.entity_type, issue.external_id) == (IssueEntity.MATCH, "m-unknown")
        assert issue.severity is IssueSeverity.ERROR
        assert issue.details == {"team_external_id": "999"}
        assert issue.detector == "ingestion:matches"
        assert issue.ingestion_run_id is not None

    def test_impossible_score_is_stored_and_flagged(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches[0] = builders.match(
            score=ScorePair(home=1, away=0), half_time_score=ScorePair(home=2, away=0)
        )

        sync.sync_competition(builders.COMPETITION)

        (issue,) = issues(session_factory, issue_type=IssueType.IMPOSSIBLE_SCORE)
        assert issue.severity is IssueSeverity.ERROR
        assert issue.entity_id is not None
        with session_factory() as session:
            stored = session.get_one(Match, issue.entity_id)  # stored, not dropped
            assert (stored.home_half_time_score, stored.home_score) == (2, 1)

    def test_corrupt_kickoff_is_not_stored_but_fully_recorded(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches.append(
            builders.match(
                external_id="m-1700",
                home_team_external_id="42",
                away_team_external_id="65",
                kickoff_at=datetime(1700, 1, 1, tzinfo=UTC),
            )
        )

        result = sync.sync_competition(builders.COMPETITION)[-1]

        assert result.failed == 1
        assert count(session_factory, Match) == 2
        (issue,) = issues(session_factory, issue_type=IssueType.INVALID_DATE)
        assert issue.external_id == "m-1700"
        assert issue.details["provider_record"]["kickoff_at"].startswith("1700-01-01")

    def test_kickoff_outside_season_is_stored_with_a_warning(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches[0] = builders.match(kickoff_at=datetime(2024, 10, 1, tzinfo=UTC))

        sync.sync_competition(builders.COMPETITION)

        (issue,) = issues(session_factory, issue_type=IssueType.INVALID_DATE)
        assert (issue.severity, issue.external_id) == (IssueSeverity.WARNING, "1035037")
        assert count(session_factory, Match) == 2

    def test_match_from_another_season_is_inconsistent(
        self, session_factory: sessionmaker[Session]
    ) -> None:
        # A provider that ignores the season filter and returns a 2022 fixture
        # when asked for 2023.
        class LeakyProvider(MockFootballDataProvider):
            def get_matches(
                self, competition_id: str, season_id: str
            ) -> FetchResult[ProviderMatch]:
                return FetchResult(list(self.dataset.matches))

        provider = LeakyProvider(builders.dataset())
        provider.dataset.matches.append(
            builders.match(external_id="m-other", season_external_id=builders.PREVIOUS_SEASON)
        )
        sync = ProviderSync(provider, session_factory)

        result = sync.sync_competition(builders.COMPETITION)[-1]

        assert (result.inserted, result.failed) == (2, 1)
        (issue,) = issues(session_factory, issue_type=IssueType.INCONSISTENT_COMPETITION)
        assert issue.external_id == "m-other"
        assert issue.details["requested"] == {
            "competition": builders.COMPETITION,
            "season": builders.SEASON,
        }

    def test_duplicate_provider_id_in_one_response(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.teams[builders.SEASON].append(builders.team("44", "Burnley FC"))

        result = sync.sync_competition(builders.COMPETITION)[2]

        assert (result.inserted, result.failed) == (4, 1)
        (issue,) = issues(session_factory, issue_type=IssueType.DUPLICATE_PROVIDER_ID)
        assert issue.details["kept"]["name"] == "Burnley"
        assert issue.details["skipped"]["name"] == "Burnley FC"
        assert issue.details["identical"] is False

    def test_malformed_provider_record(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.rejected[ProviderOperation.MATCHES] = [
            RejectedRecord("fixture.status.short: unknown status 'ZZZ'", "1035099")
        ]

        sync.sync_competition(builders.COMPETITION)

        (issue,) = issues(session_factory, issue_type=IssueType.MALFORMED_RECORD)
        assert (issue.external_id, issue.entity_type) == ("1035099", IssueEntity.MATCH)
        assert "ZZZ" in issue.message

    def test_unrecognised_season_name_is_kept_and_flagged(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.seasons[1] = builders.season(name="Apertura 2023")

        sync.sync_competition(builders.COMPETITION)

        with session_factory() as session:
            assert "Apertura 2023" in session.scalars(select(Season.name)).all()
        (issue,) = issues(session_factory, issue_type=IssueType.INCOMPLETE_RECORD)
        assert issue.severity is IssueSeverity.WARNING


class TestStatistics:
    def sync_with_statistics(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        stats: list[ProviderTeamMatchStatistics],
    ) -> None:
        provider.dataset.team_statistics["1035037"] = stats
        sync.sync_competition(builders.COMPETITION)
        sync.sync_team_statistics(builders.COMPETITION, builders.SEASON)

    def test_statistics_for_a_team_not_in_the_match(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.sync_with_statistics(
            sync,
            provider,
            [ProviderTeamMatchStatistics(match_external_id="1035037", team_external_id="42")],
        )

        (issue,) = issues(session_factory, issue_type=IssueType.INCONSISTENT_STATISTICS)
        assert issue.external_id == "1035037:42"
        assert count(session_factory, TeamMatchStatistics) == 0

    def test_contradictory_statistics_are_kept_in_the_issue(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.sync_with_statistics(
            sync,
            provider,
            [
                ProviderTeamMatchStatistics(
                    match_external_id="1035037", team_external_id="44", shots=2, shots_on_target=5
                )
            ],
        )

        (issue,) = issues(session_factory, issue_type=IssueType.INCONSISTENT_STATISTICS)
        assert issue.details["provider_record"]["shots_on_target"] == 5
        assert count(session_factory, TeamMatchStatistics) == 0


class TestLifecycle:
    def test_repeat_detection_updates_instead_of_duplicating(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches[0] = builders.match(
            score=ScorePair(home=20, away=0), half_time_score=ScorePair(home=11, away=0)
        )

        for _ in range(3):
            sync.sync_competition(builders.COMPETITION)

        (issue,) = issues(session_factory, issue_type=IssueType.IMPOSSIBLE_SCORE)
        assert issue.occurrences == 3
        assert issue.last_seen_at >= issue.first_seen_at

    def test_fixed_problems_are_resolved_not_deleted(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        provider.dataset.matches.append(
            builders.match(
                external_id="m-late",
                home_team_external_id="999",
                away_team_external_id="44",
                kickoff_at=datetime(2023, 9, 1, 19, tzinfo=UTC),
                score=ScorePair(home=1, away=1),
                status=MatchStatus.FINISHED,
            )
        )
        sync.sync_competition(builders.COMPETITION)
        assert len(issues(session_factory, issue_type=IssueType.MISSING_TEAM)) == 1

        # The provider now lists the team, so the match can be stored.
        provider.dataset.teams[builders.SEASON].append(builders.team("999", "Luton"))
        sync.sync_competition(builders.COMPETITION)

        assert issues(session_factory, issue_type=IssueType.MISSING_TEAM) == []
        (history,) = issues(session_factory, open_only=False, issue_type=IssueType.MISSING_TEAM)
        assert history.resolved_at is not None
        assert count(session_factory, Match) == 3

    def test_records_a_provider_drops_are_kept(
        self,
        sync: ProviderSync,
        provider: MockFootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        sync.sync_competition(builders.COMPETITION)
        provider.dataset.matches = provider.dataset.matches[:1]

        sync.sync_competition(builders.COMPETITION)

        assert count(session_factory, Match) == 2  # never deleted
