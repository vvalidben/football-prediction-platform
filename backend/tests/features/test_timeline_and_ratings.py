"""Unit tests for the leakage guards and the rating engine (no database)."""

from datetime import UTC, datetime, timedelta

import pytest

from app.features.config import FeatureConfig
from app.features.history import TeamIndex
from app.features.ratings import RatingEngine
from app.features.timeline import LeakageError, PastMatch, Target, Timeline
from app.models.enums import CompetitionType

T0 = datetime(2024, 1, 1, 15, tzinfo=UTC)
DELAY = timedelta(hours=3)


def past(
    match_id: int,
    kickoff: datetime,
    home: int = 1,
    away: int = 2,
    score: tuple[int, int] = (1, 0),
    season: int = 1,
    delay: timedelta = DELAY,
) -> PastMatch:
    return PastMatch(
        match_id=match_id,
        competition_id=1,
        season_id=season,
        kickoff_at=kickoff,
        available_at=kickoff + delay,
        home_team_id=home,
        away_team_id=away,
        home_goals=score[0],
        away_goals=score[1],
        home_xg=None,
        away_xg=None,
    )


def target(match_id: int, kickoff: datetime, cutoff: datetime | None = None) -> Target:
    return Target(
        match_id=match_id,
        competition_id=1,
        competition_type=CompetitionType.LEAGUE,
        season_id=1,
        season_start=None,
        season_end=None,
        kickoff_at=kickoff,
        cutoff=cutoff or kickoff,
        home_team_id=1,
        away_team_id=2,
        matchday=None,
    )


class TestTimeline:
    def test_only_results_known_by_the_cutoff_are_fed(self) -> None:
        index = TeamIndex()
        matches = [
            past(1, T0 - timedelta(days=7)),  # known long before
            past(2, T0 - timedelta(hours=4)),  # finished 1h before the cutoff: known
            past(3, T0 - timedelta(hours=2)),  # still being played: not known
            past(4, T0 + timedelta(days=7)),  # future
        ]
        timeline = Timeline(matches, [index])

        timeline.advance(T0)

        assert [m.match_id for m in index.matches(1)] == [1, 2]
        assert timeline.latest_available is not None
        assert timeline.latest_available <= T0

    def test_cannot_move_backwards(self) -> None:
        timeline = Timeline([], [])
        timeline.advance(T0)

        with pytest.raises(LeakageError, match="cannot move back"):
            timeline.advance(T0 - timedelta(seconds=1))

    def test_refuses_when_the_targets_own_result_is_known(self) -> None:
        # With no result delay the target's result would count as known at kickoff.
        timeline = Timeline([past(99, T0, delay=timedelta(0))], [])

        with pytest.raises(LeakageError, match="own result"):
            timeline.prepare(target(99, T0))

    def test_cutoff_after_kickoff_is_rejected(self) -> None:
        with pytest.raises(LeakageError, match="after kickoff"):
            target(1, T0, cutoff=T0 + timedelta(minutes=1))


class TestRatings:
    def test_winner_gains_what_loser_loses(self) -> None:
        engine = RatingEngine(FeatureConfig())

        engine.add(past(1, T0, score=(2, 0)))

        home, away = engine.rating(1), engine.rating(2)
        assert home.elo > 1500 > away.elo
        assert home.elo + away.elo == pytest.approx(3000)

    def test_expected_home_win_includes_home_advantage(self) -> None:
        engine = RatingEngine(FeatureConfig())

        assert engine.expected_score(1, 2) > 0.5  # equal ratings, home side favoured

    def test_bigger_margins_move_ratings_more(self) -> None:
        narrow, wide = RatingEngine(FeatureConfig()), RatingEngine(FeatureConfig())

        narrow.add(past(1, T0, score=(1, 0)))
        wide.add(past(1, T0, score=(4, 0)))

        assert wide.rating(1).elo > narrow.rating(1).elo

    def test_attack_rises_and_opponent_defence_falls_after_many_goals(self) -> None:
        engine = RatingEngine(FeatureConfig())

        engine.add(past(1, T0, score=(5, 0)))

        assert engine.rating(1).attack > 0 > engine.rating(2).defence
        home_goals, away_goals = engine.expected_goals(1, 2)
        assert home_goals > away_goals

    def test_new_season_pulls_ratings_toward_the_mean(self) -> None:
        engine = RatingEngine(FeatureConfig(elo_season_regression=0.5))
        for n in range(5):
            engine.add(past(n, T0 + timedelta(days=7 * n), score=(3, 0)))
        before = engine.rating(1).elo

        engine.add(past(99, T0 + timedelta(days=200), season=2, score=(1, 1)))

        after = engine.rating(1).elo
        assert 1500 < after < before  # regressed halfway, then a draw vs a weaker side
        assert after == pytest.approx(1500 + 0.5 * (before - 1500), abs=15)

    def test_switching_competition_is_not_a_new_season(self) -> None:
        """Season ids are per competition, so a side alternating between its league and
        Europe changes season id every other match. Pulling it back each time collapsed
        the ratings of exactly the best teams."""

        def final_rating(*seasons: int) -> float:
            engine = RatingEngine(FeatureConfig(elo_season_regression=0.5))
            for n in range(20):
                engine.add(
                    past(
                        n,
                        T0 + timedelta(days=7 * n),
                        season=seasons[n % len(seasons)],
                        score=(3, 0),
                    )
                )
            return engine.rating(1).elo

        assert final_rating(1, 7) == pytest.approx(final_rating(1))

    def test_a_new_year_pulls_back_once_across_competitions(self) -> None:
        engine = RatingEngine(FeatureConfig(elo_season_regression=0.5))
        for n in range(5):
            engine.add(past(n, T0 + timedelta(days=7 * n), score=(3, 0)))
        before = engine.rating(1).elo

        engine.add(past(50, T0 + timedelta(days=365), season=2, score=(1, 1)))
        once = engine.rating(1).elo
        engine.add(past(51, T0 + timedelta(days=372), season=8, score=(1, 1)))
        engine.add(past(52, T0 + timedelta(days=379), season=2, score=(1, 1)))

        assert once == pytest.approx(1500 + 0.5 * (before - 1500), abs=15)
        assert engine.rating(1).elo == pytest.approx(once, abs=20)


def test_version_changes_with_any_parameter() -> None:
    base = FeatureConfig()

    assert base.version() == FeatureConfig().version()
    assert base.version() != FeatureConfig(elo_k=21).version()
    assert base.version() != FeatureConfig(result_delay=timedelta(hours=2)).version()
    assert base.version().startswith("v1-")
