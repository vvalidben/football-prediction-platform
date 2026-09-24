"""Database constraints: duplicate provider IDs, natural keys and domain checks."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    Bookmaker,
    CompetitionExternalId,
    DataSource,
    Injury,
    MatchEvent,
    MatchExternalId,
    Odds,
    PlayerExternalId,
    SeasonExternalId,
    Standing,
    TeamExternalId,
    TeamMatchStatistics,
)
from app.models.enums import (
    AbsenceType,
    AvailabilityStatus,
    MatchEventType,
    MatchStatus,
    OddsMarket,
)
from app.models.mixins import ExternalIdMixin
from tests.db import factories
from tests.db.factories import assert_violates

pytestmark = pytest.mark.integration

# (mapping model, entity attribute, factory building a fresh entity)
MappingCase = tuple[type[ExternalIdMixin], str, Callable[[Session], object]]
MAPPING_CASES: list[MappingCase] = [
    (CompetitionExternalId, "competition", factories.competition),
    (SeasonExternalId, "season", factories.season),
    (TeamExternalId, "team", factories.team),
    (PlayerExternalId, "player", factories.player),
    (MatchExternalId, "match", factories.match),
]
MAPPING_IDS = [case[0].__tablename__ for case in MAPPING_CASES]  # type: ignore[attr-defined]


def _mapping(model: type[ExternalIdMixin], attr: str, entity: object, **kw: object) -> object:
    return model(**{attr: entity}, **kw)


@pytest.mark.parametrize(("model", "attr", "make"), MAPPING_CASES, ids=MAPPING_IDS)
def test_provider_id_cannot_map_to_two_entities(
    session: Session, model: type[ExternalIdMixin], attr: str, make: Callable[[Session], object]
) -> None:
    source = factories.data_source(session)
    session.add(_mapping(model, attr, make(session), data_source=source, external_id="123"))
    session.flush()

    table = model.__tablename__  # type: ignore[attr-defined]
    assert_violates(
        session,
        f"uq_{table}_data_source_id_external_id",
        lambda: _mapping(model, attr, make(session), data_source=source, external_id="123"),
    )


@pytest.mark.parametrize(("model", "attr", "make"), MAPPING_CASES, ids=MAPPING_IDS)
def test_entity_has_one_id_per_provider(
    session: Session, model: type[ExternalIdMixin], attr: str, make: Callable[[Session], object]
) -> None:
    source = factories.data_source(session)
    entity = make(session)
    session.add(_mapping(model, attr, entity, data_source=source, external_id="1"))
    session.flush()

    table = model.__tablename__  # type: ignore[attr-defined]
    assert_violates(
        session,
        f"uq_{table}_{attr}_id_data_source_id",
        lambda: _mapping(model, attr, entity, data_source=source, external_id="2"),
    )


@pytest.mark.parametrize(("model", "attr", "make"), MAPPING_CASES, ids=MAPPING_IDS)
def test_same_external_id_is_allowed_at_different_providers(
    session: Session, model: type[ExternalIdMixin], attr: str, make: Callable[[Session], object]
) -> None:
    first, second = factories.data_source(session), factories.data_source(session)
    entity_a, entity_b = make(session), make(session)

    session.add_all(
        [
            _mapping(model, attr, entity_a, data_source=first, external_id="7"),
            _mapping(model, attr, entity_b, data_source=second, external_id="7"),
        ]
    )
    session.flush()  # no IntegrityError


def test_duplicate_provider_event_id_is_rejected(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)

    def event() -> MatchEvent:
        return MatchEvent(match=match, data_source=source, external_id="e1", type="goal", minute=10)

    session.add(event())
    session.flush()

    assert_violates(session, "uq_match_events_data_source_id_external_id", event)


def test_observations_without_external_id_are_not_unique_on_it(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)

    session.add_all(
        MatchEvent(match=match, data_source=source, type=MatchEventType.OTHER, minute=m)
        for m in (1, 2)
    )
    session.flush()  # NULL external_ids never collide


def test_one_statistics_row_per_match_team_and_provider(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)
    session.add(TeamMatchStatistics(data_source=source, match=match, team=match.home_team))
    session.flush()

    assert_violates(
        session,
        "uq_team_match_statistics_match_id_team_id_data_source_id",
        lambda: TeamMatchStatistics(data_source=source, match=match, team=match.home_team),
    )
    # Another provider may report the same match.
    session.add(
        TeamMatchStatistics(
            data_source=factories.data_source(session), match=match, team=match.home_team
        )
    )
    session.flush()


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        ({"possession": Decimal("100.5")}, "ck_team_match_statistics_possession_percentage"),
        (
            {"shots": 3, "shots_on_target": 4},
            "ck_team_match_statistics_shots_on_target_within_shots",
        ),
        ({"corners": -1}, "ck_team_match_statistics_counts_non_negative"),
    ],
)
def test_team_statistics_checks(
    session: Session, overrides: dict[str, object], constraint: str
) -> None:
    match = factories.match(session)
    source = factories.data_source(session)

    assert_violates(
        session,
        constraint,
        lambda: TeamMatchStatistics(
            data_source=source, match=match, team=match.home_team, **overrides
        ),
    )


class TestMatchConstraints:
    def test_team_cannot_play_itself(self, session: Session) -> None:
        team = factories.team(session)
        season = factories.season(session)

        with pytest.raises(IntegrityError, match="ck_matches_distinct_teams"):
            factories.match(session, season, team, team)

    def test_finished_match_requires_a_score(self, session: Session) -> None:
        with pytest.raises(IntegrityError, match="ck_matches_finished_has_score"):
            factories.match(session, status=MatchStatus.FINISHED)

    def test_scores_must_be_paired_and_non_negative(self, session: Session) -> None:
        with pytest.raises(IntegrityError, match="ck_matches_scores_paired"):
            factories.match(session, home_score=1)
        session.rollback()
        with pytest.raises(IntegrityError, match="ck_matches_scores_non_negative"):
            factories.match(session, home_score=-1, away_score=0)

    def test_competition_must_match_the_season(self, session: Session) -> None:
        season = factories.season(session)
        other_competition = factories.competition(session)

        with pytest.raises(IntegrityError, match="fk_matches_season_id_competition_id_seasons"):
            factories.match(session, season, competition=other_competition)

    def test_same_fixture_cannot_be_stored_twice(self, session: Session) -> None:
        first = factories.match(session)

        with pytest.raises(IntegrityError, match="uq_matches_home_team_id_away_team_id_kickoff_at"):
            factories.match(
                session,
                first.season,
                first.home_team,
                first.away_team,
                kickoff_at=first.kickoff_at,
            )

    def test_unknown_status_is_rejected_by_the_database(self, session: Session) -> None:
        match = factories.match(session)

        with pytest.raises(IntegrityError, match="ck_matches_match_status"):
            session.execute(
                text("UPDATE matches SET status = 'halftime' WHERE id = :id"), {"id": match.id}
            )


def test_only_one_current_season_per_competition(session: Session) -> None:
    competition = factories.competition(session)
    factories.season(session, competition).is_current = True
    session.flush()

    second = factories.season(session, competition)
    with pytest.raises(IntegrityError, match="uq_seasons_one_current_per_competition"):
        second.is_current = True
        session.flush()


def test_standings_played_must_equal_results(session: Session) -> None:
    match = factories.match(session)
    source = factories.data_source(session)

    def row() -> Standing:
        return Standing(
            data_source=source,
            season=match.season,
            team=match.home_team,
            as_of_date=date(2025, 9, 1),
            position=1,
            played=5,
            won=2,
            drawn=1,
            lost=1,
            goals_for=5,
            goals_against=3,
            points=7,
        )

    assert_violates(session, "ck_standings_played_equals_results", row)


def test_standings_snapshot_is_unique_even_without_a_group(session: Session) -> None:
    match = factories.match(session)
    source = factories.data_source(session)

    def row() -> Standing:
        return Standing(
            data_source=source,
            season=match.season,
            team=match.home_team,
            as_of_date=date(2025, 9, 1),
            position=1,
            played=0,
            won=0,
            drawn=0,
            lost=0,
            goals_for=0,
            goals_against=0,
            points=0,
        )

    session.add(row())
    session.flush()

    # group_name is NULL in both rows; NULLS NOT DISTINCT still treats them as equal.
    assert_violates(session, "uq_standings_natural_key", row)


class TestOddsConstraints:
    RECORDED = datetime(2025, 8, 15, 12, tzinfo=UTC)

    def _odds(self, session: Session, **overrides: object) -> Odds:
        values: dict[str, object] = {
            "data_source": factories.data_source(session),
            "match": factories.match(session),
            "bookmaker": Bookmaker(name=f"Book {id(overrides)}"),
            "market": OddsMarket.MATCH_WINNER,
            "selection": "home",
            "price": Decimal("2.10"),
            "recorded_at": self.RECORDED,
            **overrides,
        }
        return Odds(**values)

    def test_price_must_exceed_one(self, session: Session) -> None:
        assert_violates(
            session,
            "ck_odds_price_above_one",
            lambda: self._odds(session, price=Decimal("1.00")),
        )

    def test_over_under_requires_a_line(self, session: Session) -> None:
        assert_violates(
            session,
            "ck_odds_over_under_selection",
            lambda: self._odds(session, market=OddsMarket.OVER_UNDER, selection="over"),
        )

    def test_match_winner_selection_is_validated(self, session: Session) -> None:
        assert_violates(
            session,
            "ck_odds_match_winner_selection",
            lambda: self._odds(session, selection="1"),
        )

    def test_duplicate_snapshot_is_rejected(self, session: Session) -> None:
        first = self._odds(session)
        session.add(first)
        session.flush()

        def duplicate() -> Odds:
            return Odds(
                data_source=first.data_source,
                match=first.match,
                bookmaker=first.bookmaker,
                market=first.market,
                selection=first.selection,
                price=Decimal("2.20"),
                recorded_at=first.recorded_at,
            )

        # line is NULL in both rows and is still compared (NULLS NOT DISTINCT).
        assert_violates(session, "uq_odds_natural_key", duplicate)


def test_injury_needs_a_match_or_a_start_date(session: Session) -> None:
    source, player = factories.data_source(session), factories.player(session)

    assert_violates(
        session,
        "ck_injuries_match_or_start_date",
        lambda: Injury(
            data_source=source,
            player=player,
            absence_type=AbsenceType.SUSPENSION,
            status=AvailabilityStatus.OUT,
        ),
    )


def test_data_source_code_must_be_a_slug(session: Session) -> None:
    assert_violates(
        session,
        "ck_data_sources_code_format",
        lambda: DataSource(code="Football Data", name="x"),
    )
