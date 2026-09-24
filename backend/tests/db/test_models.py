"""Creating entities and navigating their relationships against the real schema."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Bookmaker,
    CompetitionExternalId,
    DataIngestionRun,
    FeatureSnapshot,
    Injury,
    Lineup,
    LineupPlayer,
    Match,
    MatchEvent,
    MatchExternalId,
    Odds,
    PlayerExternalId,
    PlayerMatchStatistics,
    SeasonExternalId,
    Standing,
    Team,
    TeamExternalId,
    TeamMatchStatistics,
)
from app.models.enums import (
    AbsenceType,
    AvailabilityStatus,
    IngestionStatus,
    MatchEventType,
    MatchPeriod,
    MatchStatus,
    OddsMarket,
)
from tests.db import factories
from tests.db.factories import violated_constraint

pytestmark = pytest.mark.integration


def test_competition_season_and_match_graph(session: Session) -> None:
    source = factories.data_source(session, code="football_data_org")
    season = factories.season(session)
    season.competition.external_ids.append(
        CompetitionExternalId(data_source=source, external_id="PL")
    )
    season.external_ids.append(SeasonExternalId(data_source=source, external_id="2025"))
    home, away = factories.team(session, "Arsenal"), factories.team(session, "Chelsea")
    match = factories.match(
        session,
        season,
        home,
        away,
        status=MatchStatus.FINISHED,
        home_score=2,
        away_score=1,
        home_half_time_score=1,
        away_half_time_score=0,
        venue="Emirates Stadium",
        referee="Michael Oliver",
        external_ids=[MatchExternalId(data_source=source, external_id="537785")],
    )
    session.commit()
    session.expire_all()

    loaded = session.get_one(Match, match.id)
    assert loaded.home_team.name == "Arsenal"
    assert loaded.away_team.name == "Chelsea"
    assert loaded.season.competition_id == loaded.competition_id
    assert loaded.status is MatchStatus.FINISHED
    assert (loaded.home_score, loaded.away_score) == (2, 1)
    assert loaded.external_ids[0].data_source.code == "football_data_org"
    assert loaded.season.external_ids[0].external_id == "2025"
    assert loaded.competition.external_ids[0].external_id == "PL"
    assert loaded in loaded.season.matches
    assert loaded.created_at is not None
    assert loaded.updated_at is not None


def test_team_can_carry_ids_from_several_providers(session: Session) -> None:
    team = factories.team(session)
    first, second = factories.data_source(session), factories.data_source(session)
    team.external_ids.extend(
        [
            TeamExternalId(data_source=first, external_id="57"),
            TeamExternalId(data_source=second, external_id="42"),
        ]
    )
    session.flush()

    found = session.scalars(
        select(Team)
        .join(TeamExternalId)
        .where(TeamExternalId.data_source_id == second.id, TeamExternalId.external_id == "42")
    ).one()
    assert found is team
    assert {ext.external_id for ext in found.external_ids} == {"57", "42"}


def test_match_events_lineups_and_player_links(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)
    scorer = factories.player(session, current_team=match.home_team)
    assister = factories.player(session, current_team=match.home_team)
    scorer.external_ids.append(PlayerExternalId(data_source=source, external_id="p-1"))

    match.events.append(
        MatchEvent(
            data_source=source,
            external_id="evt-1",
            type=MatchEventType.GOAL,
            period=MatchPeriod.SECOND_HALF,
            minute=90,
            added_time=3,
            team=match.home_team,
            player=scorer,
            related_player=assister,
            detail="Header",
        )
    )
    match.lineups.append(
        Lineup(
            data_source=source,
            team=match.home_team,
            formation="4-3-3",
            players=[
                LineupPlayer(player=scorer, is_starter=True, position="ST", shirt_number=9),
                LineupPlayer(player=assister, is_starter=False, shirt_number=17),
            ],
        )
    )
    session.commit()
    session.expire_all()

    loaded = session.get_one(Match, match.id)
    goal = loaded.events[0]
    assert (goal.minute, goal.added_time, goal.type) == (90, 3, MatchEventType.GOAL)
    assert goal.player is not None and goal.player.id == scorer.id
    assert goal.related_player is not None and goal.related_player.id == assister.id
    lineup = loaded.lineups[0]
    assert lineup.formation == "4-3-3"
    assert {(lp.player_id, lp.is_starter) for lp in lineup.players} == {
        (scorer.id, True),
        (assister.id, False),
    }
    assert scorer.current_team is match.home_team


def test_statistics_store_partial_and_provider_specific_values(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)
    home_stats = TeamMatchStatistics(
        data_source=source,
        match=match,
        team=match.home_team,
        possession=Decimal("58.40"),
        shots=15,
        shots_on_target=6,
        corners=7,
        fouls=10,
        yellow_cards=2,
        red_cards=0,
        expected_goals=Decimal("1.874"),
        extra={"big_chances": 3, "touches_in_box": 31},
    )
    # A provider that only reports a few numbers: the rest stay NULL, not 0.
    away_stats = TeamMatchStatistics(
        data_source=source,
        match=match,
        team=match.away_team,
        shots=8,
        expected_goals=Decimal("0.91"),
    )
    player = factories.player(session, current_team=match.home_team)
    player_stats = PlayerMatchStatistics(
        data_source=source,
        match=match,
        player=player,
        team=match.home_team,
        minutes_played=90,
        goals=1,
        expected_goals=Decimal("0.640"),
        expected_assists=Decimal("0.120"),
        rating=Decimal("7.80"),
        extra={"aerials_won": 4},
    )
    session.add_all([home_stats, away_stats, player_stats])
    session.commit()
    session.expire_all()

    stats = {
        row.team_id: row
        for row in session.scalars(
            select(TeamMatchStatistics).where(TeamMatchStatistics.match_id == match.id)
        )
    }
    home, away = stats[match.home_team_id], stats[match.away_team_id]
    assert home.possession == Decimal("58.40")
    assert home.extra == {"big_chances": 3, "touches_in_box": 31}
    assert away.corners is None
    assert away.possession is None
    assert away.extra == {}
    # xGA is derived from the opponent's row rather than stored.
    assert away.expected_goals == Decimal("0.910")
    assert session.get_one(PlayerMatchStatistics, player_stats.id).rating == Decimal("7.80")


def test_injury_odds_standing_and_ingestion_run(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)
    player = factories.player(session, current_team=match.home_team)
    bookmaker = Bookmaker(name="Pinnacle")
    recorded = datetime(2025, 8, 15, 12, tzinfo=UTC)

    session.add_all(
        [
            Injury(
                data_source=source,
                player=player,
                team=match.home_team,
                absence_type=AbsenceType.INJURY,
                status=AvailabilityStatus.OUT,
                reason="Hamstring",
                start_date=date(2025, 8, 1),
                expected_return_date=date(2025, 9, 1),
            ),
            Odds(
                data_source=source,
                match=match,
                bookmaker=bookmaker,
                market=OddsMarket.MATCH_WINNER,
                selection="home",
                price=Decimal("2.150"),
                recorded_at=recorded,
            ),
            Odds(
                data_source=source,
                match=match,
                bookmaker=bookmaker,
                market=OddsMarket.OVER_UNDER,
                selection="over",
                line=Decimal("2.5"),
                price=Decimal("1.910"),
                recorded_at=recorded,
            ),
            Standing(
                data_source=source,
                season=match.season,
                team=match.home_team,
                as_of_date=date(2025, 10, 1),
                position=1,
                played=7,
                won=5,
                drawn=1,
                lost=1,
                goals_for=15,
                goals_against=6,
                points=16,
                form="WWDWL",
            ),
            DataIngestionRun(
                data_source=source,
                job="fixtures",
                parameters={"competition": "PL", "season": "2025"},
                started_at=recorded,
            ),
            FeatureSnapshot(
                match=match,
                feature_set_version="v1",
                data_cutoff_at=recorded,
                features={"home_elo": 1850.5, "away_elo": 1790.0},
            ),
        ]
    )
    session.commit()
    session.expire_all()

    standing = session.scalars(select(Standing).where(Standing.team_id == match.home_team_id)).one()
    assert standing.goal_difference == 9  # computed by the database
    assert standing in standing.season.standings
    run = session.scalars(select(DataIngestionRun)).one()
    assert run.status is IngestionStatus.RUNNING
    assert run.records_fetched == 0
    assert run in source.ingestion_runs
    odds_count = session.scalar(
        select(func.count()).select_from(Odds).where(Odds.match_id == match.id)
    )
    assert odds_count == 2


def test_deleting_a_match_cascades_to_its_provider_data(session: Session) -> None:
    source = factories.data_source(session)
    match = factories.match(session)
    player = factories.player(session)
    match.external_ids.append(MatchExternalId(data_source=source, external_id="m-1"))
    match.events.append(
        MatchEvent(data_source=source, type=MatchEventType.YELLOW_CARD, minute=12, player=player)
    )
    session.add(TeamMatchStatistics(data_source=source, match=match, team=match.home_team, shots=3))
    session.flush()
    match_id = match.id

    session.delete(match)
    session.flush()

    for model in (MatchEvent, MatchExternalId, TeamMatchStatistics):
        assert session.scalar(select(func.count()).where(model.match_id == match_id)) == 0
    assert session.get(type(player), player.id) is not None  # players survive


def test_referenced_team_cannot_be_deleted(session: Session) -> None:
    match = factories.match(session)

    violated = violated_constraint(session, lambda: session.delete(match.home_team))

    assert violated == "fk_matches_home_team_id_teams"


def test_data_source_in_use_cannot_be_deleted(session: Session) -> None:
    team = factories.team(session)
    source = factories.data_source(session)
    session.add(TeamExternalId(team=team, data_source=source, external_id="1"))
    session.flush()

    violated = violated_constraint(session, lambda: session.delete(source))

    assert violated == "fk_team_external_ids_data_source_id_data_sources"
