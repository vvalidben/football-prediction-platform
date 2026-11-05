"""Linking a second provider onto stored rows, so its data extends them, not copies them."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.linking import link_competition
from app.ingestion.sync import ProviderSync
from app.models import Competition, Match, Team
from app.models.enums import CompetitionType, MatchStatus
from app.providers.mock import MockDataset, MockFootballDataProvider
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ScorePair,
)

FIRST_KICKOFF = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
SEASON = "first:2024"
#: Same clubs, spelled the way each provider spells them.
OLD_NAMES = ["Arsenal", "Wolves", "Spurs", "Leeds"]
NEW_NAMES = {
    "n1": "Arsenal FC",
    "n2": "Wolverhampton Wanderers FC",
    "n3": "Tottenham Hotspur FC",
}
#: (home, away, home goals, away goals), each pairing exactly as both providers report it.
RESULTS = [
    (0, 1, 2, 0),
    (1, 2, 1, 1),
    (2, 0, 0, 3),
    (1, 0, 2, 2),
    (2, 1, 4, 1),
    (0, 2, 1, 0),
]


def _season(external_id: str, competition: str, name: str, *, current: bool) -> ProviderSeason:
    start_year = int(name.split("/")[0])
    return ProviderSeason(
        external_id=external_id,
        competition_external_id=competition,
        name=name,
        start_date=date(start_year, 8, 1),
        end_date=date(start_year + 1, 5, 31),
        is_current=current,
    )


def _old_dataset() -> MockDataset:
    teams = [ProviderTeam(external_id=f"o{i}", name=n) for i, n in enumerate(OLD_NAMES)]
    matches = [
        ProviderMatch(
            external_id=f"old{i}",
            competition_external_id="first",
            season_external_id=SEASON,
            home_team_external_id=f"o{home}",
            away_team_external_id=f"o{away}",
            kickoff_at=FIRST_KICKOFF + timedelta(days=7 * i),
            status=MatchStatus.FINISHED,
            score=ScorePair(home=h, away=a),
        )
        for i, (home, away, h, a) in enumerate(RESULTS)
    ]
    # Leeds appears only in a fixture the new provider cannot see, so nothing but the
    # name can connect it to the side the new provider calls "Leeds United FC".
    matches.append(
        ProviderMatch(
            external_id="old_leeds",
            competition_external_id="first",
            season_external_id=SEASON,
            home_team_external_id="o3",
            away_team_external_id="o0",
            kickoff_at=FIRST_KICKOFF - timedelta(days=30),
            status=MatchStatus.FINISHED,
            score=ScorePair(home=1, away=1),
        )
    )
    return MockDataset(
        competitions=[
            ProviderCompetition(
                external_id="first", name="Premier League", type=CompetitionType.LEAGUE
            )
        ],
        seasons=[_season(SEASON, "first", "2024/2025", current=False)],
        teams={SEASON: teams},
        matches=matches,
    )


def _new_dataset() -> MockDataset:
    season = "second:2024"
    next_season = "second:2025"
    teams = [ProviderTeam(external_id=k, name=v) for k, v in NEW_NAMES.items()]
    teams_next = [
        *teams,
        ProviderTeam(external_id="n4", name="Leeds United FC"),
        ProviderTeam(external_id="n5", name="Coventry City FC"),
    ]
    ids = list(NEW_NAMES)
    matches = [
        ProviderMatch(
            external_id=f"new{i}",
            competition_external_id="second",
            season_external_id=season,
            home_team_external_id=ids[home],
            away_team_external_id=ids[away],
            kickoff_at=FIRST_KICKOFF + timedelta(days=7 * i),
            status=MatchStatus.FINISHED,
            score=ScorePair(home=h, away=a),
        )
        for i, (home, away, h, a) in enumerate(RESULTS)
    ]
    upcoming = ProviderMatch(
        external_id="new_upcoming",
        competition_external_id="second",
        season_external_id=next_season,
        home_team_external_id="n4",
        away_team_external_id="n5",
        kickoff_at=FIRST_KICKOFF + timedelta(days=400),
        status=MatchStatus.SCHEDULED,
    )
    return MockDataset(
        competitions=[
            ProviderCompetition(
                external_id="second", name="Premier League", type=CompetitionType.LEAGUE
            )
        ],
        seasons=[
            _season(season, "second", "2024/2025", current=False),
            _season(next_season, "second", "2025/2026", current=True),
        ],
        teams={season: teams, next_season: teams_next},
        matches=[*matches, upcoming],
    )


@pytest.fixture
def linked_setup(session_factory: sessionmaker[Session]) -> tuple[ProviderSync, int]:
    ProviderSync(
        MockFootballDataProvider(_old_dataset(), code="old_provider"), session_factory
    ).sync_competition("first", "all")
    new_sync = ProviderSync(
        MockFootballDataProvider(_new_dataset(), code="new_provider"), session_factory
    )
    with session_factory() as session:
        competition_id = session.scalars(select(Competition.id)).one()
    return new_sync, competition_id


def _count(factory: sessionmaker[Session], entity: type) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(entity)) or 0


class TestDeciding:
    def test_differently_named_teams_are_matched_from_shared_fixtures(
        self, linked_setup: tuple[ProviderSync, int]
    ) -> None:
        sync, competition_id = linked_setup

        report = link_competition(sync, "second", competition_id)

        assert report.by_fixtures == {
            "Arsenal FC": "Arsenal",
            "Wolverhampton Wanderers FC": "Wolves",
            "Tottenham Hotspur FC": "Spurs",
        }

    def test_a_team_with_no_shared_fixture_falls_back_to_its_name(
        self, linked_setup: tuple[ProviderSync, int]
    ) -> None:
        sync, competition_id = linked_setup

        report = link_competition(sync, "second", competition_id)

        assert report.by_name == {"Leeds United FC": "Leeds"}

    def test_a_team_nobody_has_seen_stays_new(self, linked_setup: tuple[ProviderSync, int]) -> None:
        sync, competition_id = linked_setup

        report = link_competition(sync, "second", competition_id)

        assert report.new_teams == ["Coventry City FC"]


class TestTheSyncAfterLinking:
    def test_it_extends_stored_rows_instead_of_copying_them(
        self, linked_setup: tuple[ProviderSync, int], session_factory: sessionmaker[Session]
    ) -> None:
        sync, competition_id = linked_setup
        matches_before = _count(session_factory, Match)
        link_competition(sync, "second", competition_id)

        sync.sync_competition("second", "all")

        assert _count(session_factory, Competition) == 1
        # Four stored teams, plus Coventry, who are genuinely new.
        assert _count(session_factory, Team) == 5
        # Every finished fixture was already stored: only the upcoming one is added.
        assert _count(session_factory, Match) == matches_before + 1

    def test_without_linking_the_same_sync_would_have_copied_everything(
        self, linked_setup: tuple[ProviderSync, int], session_factory: sessionmaker[Session]
    ) -> None:
        """Why this module exists."""
        sync, _ = linked_setup

        sync.sync_competition("second", "all")

        assert _count(session_factory, Competition) == 2


class TestKickoffDisagreements:
    """Recorded from the real providers: four matches on 2024-10-27, the day the clocks
    changed, are an hour apart between them."""

    def _sync_with_kickoff_moved(
        self, sync: ProviderSync, competition_id: int, moved: timedelta
    ) -> None:
        link_competition(sync, "second", competition_id)
        dataset = sync.provider.dataset  # type: ignore[attr-defined]
        first = dataset.matches[0]
        dataset.matches[0] = first.model_copy(update={"kickoff_at": first.kickoff_at + moved})
        sync.sync_competition("second", "all")

    def test_an_hour_apart_is_still_the_same_fixture(
        self, linked_setup: tuple[ProviderSync, int], session_factory: sessionmaker[Session]
    ) -> None:
        sync, competition_id = linked_setup
        matches_before = _count(session_factory, Match)

        self._sync_with_kickoff_moved(sync, competition_id, timedelta(hours=1))

        assert _count(session_factory, Match) == matches_before + 1  # only the upcoming one

    def test_days_apart_is_a_different_fixture(
        self, linked_setup: tuple[ProviderSync, int], session_factory: sessionmaker[Session]
    ) -> None:
        """Scottish-style leagues really do play the same home tie twice in a season."""
        sync, competition_id = linked_setup
        matches_before = _count(session_factory, Match)

        self._sync_with_kickoff_moved(sync, competition_id, timedelta(days=3))

        assert _count(session_factory, Match) == matches_before + 2


class TestSafety:
    def test_a_dry_run_writes_nothing(
        self, linked_setup: tuple[ProviderSync, int], session_factory: sessionmaker[Session]
    ) -> None:
        sync, competition_id = linked_setup

        report = link_competition(sync, "second", competition_id, dry_run=True)
        sync.sync_competition("second", "all")

        assert report.by_fixtures
        assert report.mappings_written == 0
        assert _count(session_factory, Competition) == 2

    def test_linking_twice_changes_nothing(self, linked_setup: tuple[ProviderSync, int]) -> None:
        sync, competition_id = linked_setup

        first = link_competition(sync, "second", competition_id)
        second = link_competition(sync, "second", competition_id)

        assert first.mappings_written == 5  # the competition and four teams
        assert second.mappings_written == 0

    def test_it_refuses_a_competition_that_is_not_stored(
        self, linked_setup: tuple[ProviderSync, int]
    ) -> None:
        sync, _ = linked_setup

        with pytest.raises(ValueError, match="no stored competition"):
            link_competition(sync, "second", 999_999)
