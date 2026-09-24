"""Feature values checked against hand calculations on the seeded league.

Arsenal (home) v Brighton (away) is the target, one week after round 6:

    round  results
    0      Arsenal 2-0 Chelsea (xG 1.8-0.6)   Brighton 1-1 Derby
    1      Chelsea 0-3 Brighton (xG 0.9-2.1)  Derby 1-2 Arsenal
    2      Arsenal 1-1 Brighton (xG 1.2-1.0)  Chelsea 2-2 Derby
    3      Brighton 2-1 Chelsea               Arsenal 3-0 Derby (xG 2.4-0.3)
    4      Derby 0-1 Brighton                 Chelsea 1-1 Arsenal
    5      Brighton 0-2 Arsenal (xG 0.7-1.9)  Derby 1-0 Chelsea
"""

from datetime import timedelta
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.features.builder import FeatureBuilder
from app.features.config import FeatureConfig
from app.features.registry import FEATURE_NAMES
from app.features.timeline import load_targets
from app.models import Match, Player, PlayerMatchStatistics
from tests.features.seed import KICKOFF, League, build_league

pytestmark = pytest.mark.integration


@pytest.fixture
def league(session_factory: sessionmaker[Session]) -> League:
    return build_league(session_factory)


@pytest.fixture
def features(league: League) -> dict[str, float | int | None]:
    return league.features()


def test_every_registered_feature_is_present_in_order(features: dict[str, object]) -> None:
    assert list(features) == list(FEATURE_NAMES)


def test_form(features: dict[str, object]) -> None:
    # Arsenal's last three: 3-0 W, 1-1 D, 2-0 W.
    assert features["home_form3_matches"] == 3
    assert features["home_form3_goals_for"] == pytest.approx(2.0)
    assert features["home_form3_goals_against"] == pytest.approx(1 / 3, abs=1e-6)
    assert features["home_form3_points"] == pytest.approx(7 / 3, abs=1e-6)
    assert (features["home_form3_win_rate"], features["home_form3_loss_rate"]) == (
        pytest.approx(2 / 3, abs=1e-6),
        0.0,
    )
    # Last five: goals 2,1,3,1,2 for; 1,1,0,1,0 against; points 3,1,3,1,3.
    assert features["home_form5_goals_for"] == pytest.approx(1.8)
    assert features["home_form5_goals_against"] == pytest.approx(0.6)
    assert features["home_form5_points"] == pytest.approx(2.2)
    # Only six matches exist: the 10-match window says so.
    assert features["home_form10_matches"] == 6
    # Brighton's last three: 2-1 W, 1-0 W, 0-2 L.
    assert features["away_form3_goals_for"] == pytest.approx(1.0)
    assert features["away_form3_points"] == pytest.approx(2.0)


def test_xg_and_xga_use_only_matches_with_xg(features: dict[str, object]) -> None:
    # Arsenal's last three: xG known for 3-0 (2.4 / 0.3) and 2-0 away (1.9 / 0.7).
    assert features["home_form3_xg_matches"] == 2
    assert features["home_form3_xg_for"] == pytest.approx(2.15)
    assert features["home_form3_xg_against"] == pytest.approx(0.5)
    assert features["home_form10_xg_matches"] == 4


def test_rest_and_congestion(features: dict[str, object]) -> None:
    assert features["home_rest_days"] == pytest.approx(7.0)
    assert features["away_rest_days"] == pytest.approx(7.0)
    assert features["rest_days_diff"] == pytest.approx(0.0)
    assert features["home_matches_last_7d"] == 1
    assert features["home_matches_last_14d"] == 2
    assert features["home_matches_last_30d"] == 4


def test_competition_context(features: dict[str, object]) -> None:
    assert features["competition_matches"] == 12
    assert features["competition_home_win_rate"] == pytest.approx(4 / 12, abs=1e-6)
    assert features["competition_draw_rate"] == pytest.approx(4 / 12, abs=1e-6)
    assert features["competition_home_goals"] == pytest.approx(14 / 12, abs=1e-6)
    assert features["competition_is_cup"] == 0
    # 2024-03-02 in a season from 2023-08-01 to 2024-05-31: day 214 of 304.
    assert features["season_progress"] == pytest.approx(214 / 304, abs=1e-6)
    assert features["home_season_matches"] == 6
    assert features["matchday"] is None


def test_venue_strengths_are_shrunk_toward_the_average(features: dict[str, object]) -> None:
    # Arsenal at home: 6 scored, 1 conceded in 3 (k = 3). League baselines: 14 home
    # and 14 away goals in 12 matches, shrunk toward 1.5 / 1.2 with weight 20.
    home_baseline = (14 + 20 * 1.5) / 32
    away_baseline = (14 + 20 * 1.2) / 32
    assert features["home_venue_matches"] == 3
    assert features["home_attack_strength"] == pytest.approx(
        (6 + 3 * home_baseline) / 6 / home_baseline, abs=1e-6
    )
    assert features["home_defence_weakness"] == pytest.approx(
        (1 + 3 * away_baseline) / 6 / away_baseline, abs=1e-6
    )


def test_goalless_side_history_does_not_break_baselines(
    session_factory: sessionmaker[Session],
) -> None:
    """Regression (found on real data): after one 1-0, away goals per match is 0."""
    league = build_league(session_factory)
    with session_factory() as session, session.begin():
        session.query(Match).filter(Match.id != league.target_id).delete()
    league.add_match("Chelsea", "Derby", KICKOFF - timedelta(days=3), (1, 0))

    features = league.features()

    assert features["away_attack_strength"] is not None
    assert features["competition_away_goals"] == 0.0


def test_head_to_head_is_time_decayed(features: dict[str, object]) -> None:
    # Meetings 28 days ago (1-1 at Arsenal) and 7 days ago (Arsenal won 2-0 away).
    half_life = FeatureConfig().h2h_half_life_days
    old, recent = 0.5 ** (28 / half_life), 0.5 ** (7 / half_life)
    assert features["h2h_matches"] == 2
    assert features["h2h_weight"] == pytest.approx(old + recent, abs=1e-6)
    assert features["h2h_weighted_goal_diff"] == pytest.approx(
        (old * 0 + recent * 2) / (old + recent), abs=1e-6
    )
    assert features["h2h_weighted_points"] == pytest.approx(
        (old * 1 + recent * 3) / (old + recent), abs=1e-6
    )
    assert features["h2h_days_since_last"] == pytest.approx(7.0)


def test_old_meetings_count_less_than_recent_ones(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    # A heavy Brighton win four years ago: weight 1/16, margin capped at 3.
    league.add_match("Brighton", "Arsenal", KICKOFF - timedelta(days=4 * 365), (5, 0))

    features = league.features()

    weights = [0.5 ** (days / 365) for days in (4 * 365, 28, 7)]
    expected = (weights[0] * -3 + weights[1] * 0 + weights[2] * 2) / sum(weights)
    assert features["h2h_matches"] == 3
    assert features["h2h_weighted_goal_diff"] == pytest.approx(expected, abs=1e-6)
    assert features["h2h_weighted_goal_diff"] > 0.8  # type: ignore[operator]


def test_ratings(features: dict[str, object]) -> None:
    assert features["home_rated_matches"] == 6
    assert features["home_elo"] > features["away_elo"]  # type: ignore[operator]
    assert 0 < features["elo_home_win_expectancy"] < 1  # type: ignore[operator]
    assert features["home_attack_rating"] > 0  # type: ignore[operator]
    assert features["rating_expected_home_goals"] > features["rating_expected_away_goals"]  # type: ignore[operator]


def test_optional_sources_are_none_without_data(features: dict[str, object]) -> None:
    for name in FEATURE_NAMES:
        if name.startswith(("odds_", "home_absen", "away_absen")):
            assert features[name] is None, name


def test_odds_are_margin_free_probabilities(league: League) -> None:
    league.add_odds(KICKOFF - timedelta(days=2), "2.00", "3.50", "4.00", book="A")
    league.add_odds(KICKOFF - timedelta(days=2), "2.10", "3.40", "3.80", book="B")

    features = league.features()

    def implied(home: float, draw: float, away: float) -> tuple[float, float]:
        total = 1 / home + 1 / draw + 1 / away
        return (1 / home) / total, total - 1

    (a_home, a_margin), (b_home, b_margin) = implied(2.0, 3.5, 4.0), implied(2.1, 3.4, 3.8)
    assert features["odds_bookmakers"] == 2
    assert features["odds_latest_home_prob"] == pytest.approx((a_home + b_home) / 2, abs=1e-6)
    assert features["odds_latest_overround"] == pytest.approx((a_margin + b_margin) / 2, abs=1e-6)
    probs = [features[f"odds_latest_{s}_prob"] for s in ("home", "draw", "away")]
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)  # type: ignore[arg-type]


def test_absent_players_contribution(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    league.add_injury("Arsenal", start=(KICKOFF - timedelta(days=10)).date())
    with session_factory() as session, session.begin():
        injured = session.query(Player).filter(Player.name.like("Player Arsenal%")).one()
        fit = Player(name="Fit striker", current_team_id=league.teams["Arsenal"])
        session.add(fit)
        session.flush()
        past = session.query(Match).filter(Match.kickoff_at < KICKOFF).order_by(Match.id).first()
        assert past is not None
        for player, goals, assists in ((injured, 2, 1), (fit, 3, 0)):
            session.add(
                PlayerMatchStatistics(
                    match_id=past.id,
                    player_id=player.id,
                    team_id=league.teams["Arsenal"],
                    data_source_id=league.source_id,
                    goals=goals,
                    assists=assists,
                    rating=Decimal("7.0"),
                )
            )

    features = league.features()

    assert features["home_absences_out"] == 1
    assert features["home_absent_contribution"] == pytest.approx(0.5)  # 3 of 6
    assert features["away_absences_out"] is None  # no absence data for Brighton at all


def test_feature_values_are_json_friendly(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    builder = FeatureBuilder(session_factory)
    with session_factory() as session:
        (row,) = builder.build(load_targets(session, match_ids=[league.target_id]))

    assert all(v is None or isinstance(v, int | float) for v in row.values.values())
    assert row.latest_information_at is not None
    assert row.latest_information_at <= row.target.cutoff
