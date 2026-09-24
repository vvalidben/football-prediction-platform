"""MANDATORY: features for a match at time T use only information available before T.

Every test computes the target's features, then changes something the features
must NOT depend on (future matches, future statistics, the target's own result and
statistics, odds or absences after the cutoff) and requires the features to be
byte-for-byte identical. Sanity tests confirm that features *do* react to
legitimate past information, so an identical result proves exclusion, not
insensitivity.
"""

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.features.builder import FeatureBuilder
from app.features.config import FeatureConfig
from app.features.timeline import LeakageError, load_targets
from tests.features.seed import KICKOFF, League, build_league

pytestmark = pytest.mark.integration

DAY = timedelta(days=1)


@pytest.fixture
def league(session_factory: sessionmaker[Session]) -> League:
    return build_league(session_factory)


class TestFutureMatchesAreNotUsed:
    def test_matches_after_kickoff_change_nothing(self, league: League) -> None:
        before = league.features()

        # Extreme future results for both teams, days and months later.
        league.add_match("Arsenal", "Chelsea", KICKOFF + 3 * DAY, (9, 0))
        league.add_match("Derby", "Brighton", KICKOFF + 10 * DAY, (0, 9))
        league.add_match("Brighton", "Arsenal", KICKOFF + 90 * DAY, (7, 0))

        assert league.features() == before

    def test_match_being_played_at_the_cutoff_is_not_used(self, league: League) -> None:
        before = league.features()

        # Kicked off 2 hours before the target: not finished when the target starts.
        league.add_match("Chelsea", "Derby", KICKOFF - timedelta(hours=2), (5, 5))

        assert league.features() == before

    def test_result_known_before_the_cutoff_is_used(self, league: League) -> None:
        """Sanity: a match that finished before the cutoff does change features."""
        before = league.features()

        league.add_match("Chelsea", "Derby", KICKOFF - 2 * DAY, (5, 5))

        after = league.features()
        assert after != before
        assert after["competition_matches"] == before["competition_matches"] + 1  # type: ignore[operator]

    def test_with_a_lead_time_matches_between_cutoff_and_kickoff_are_excluded(
        self, league: League
    ) -> None:
        lead = timedelta(hours=48)
        before = league.features(lead=lead)

        # Played after a 48-hour-early cutoff, but before kickoff.
        league.add_match("Arsenal", "Derby", KICKOFF - DAY, (4, 0))

        assert league.features(lead=lead) == before
        assert league.features() != before  # with cutoff = kickoff it is known


class TestFutureStatisticsAreNotUsed:
    def test_statistics_of_future_matches_change_nothing(self, league: League) -> None:
        future = league.add_match("Arsenal", "Chelsea", KICKOFF + 7 * DAY, (1, 0))
        before = league.features()

        league.add_xg(future, "Arsenal", "Chelsea", (9.9, 0.1))

        assert league.features() == before

    def test_statistics_of_past_matches_are_used(self, league: League) -> None:
        """Sanity: xG of an already-known match does change xG features."""
        past = league.add_match("Arsenal", "Chelsea", KICKOFF - 2 * DAY, (1, 0))
        before = league.features()

        league.add_xg(past, "Arsenal", "Chelsea", (3.0, 0.2))

        after = league.features()
        assert after["home_form3_xg_for"] != before["home_form3_xg_for"]


class TestPostMatchInformationOfTheTargetIsNotUsed:
    def test_the_targets_own_result_changes_nothing(self, league: League) -> None:
        before = league.features()

        league.finish_target((6, 0))

        assert league.features() == before

    def test_the_targets_own_statistics_change_nothing(self, league: League) -> None:
        before = league.features()

        league.finish_target((1, 0))
        league.add_xg(league.target_id, "Arsenal", "Brighton", (4.5, 0.1))

        assert league.features() == before

    def test_zero_result_delay_would_leak_and_is_refused(
        self, league: League, session_factory: sessionmaker[Session]
    ) -> None:
        """If results counted as known at kickoff, the target's own result would enter
        the history; the timeline detects this instead of silently using it."""
        league.finish_target((6, 0))
        builder = FeatureBuilder(session_factory, FeatureConfig(result_delay=timedelta(0)))
        with session_factory() as session:
            targets = load_targets(session, match_ids=[league.target_id])

        with pytest.raises(LeakageError, match="own result"):
            builder.build(targets)


class TestOddsAndAbsencesRespectTheCutoff:
    def test_odds_recorded_after_the_cutoff_are_ignored(self, league: League) -> None:
        league.add_odds(KICKOFF - 3 * DAY, "2.00", "3.50", "4.00")
        before = league.features()

        league.add_odds(KICKOFF + timedelta(minutes=30), "1.10", "9.00", "20.00")  # in-play
        league.add_odds(KICKOFF, "1.20", "7.00", "15.00")  # exactly at the cutoff

        assert league.features() == before
        assert before["odds_bookmakers"] == 1

    def test_later_odds_before_the_cutoff_update_latest_not_opening(self, league: League) -> None:
        league.add_odds(KICKOFF - 3 * DAY, "2.00", "3.50", "4.00")
        opening_only = league.features()

        league.add_odds(KICKOFF - DAY, "1.80", "3.60", "4.50")

        both = league.features()
        assert both["odds_opening_home_prob"] == opening_only["odds_opening_home_prob"]
        assert both["odds_latest_home_prob"] != opening_only["odds_latest_home_prob"]

    def test_absences_starting_after_the_cutoff_are_ignored(self, league: League) -> None:
        league.add_injury("Arsenal", start=(KICKOFF - 10 * DAY).date())
        before = league.features()

        league.add_injury("Arsenal", start=(KICKOFF + DAY).date())  # injured later
        league.add_injury("Arsenal", start=KICKOFF.date())  # injured on the day

        assert league.features() == before
        assert before["home_absences_out"] == 1

    def test_fixture_absence_lists_ingested_after_the_cutoff_are_ignored(
        self, league: League
    ) -> None:
        # Ingested today, long after the 2024 cutoff: we cannot prove it was known.
        league.add_injury("Brighton", start=None, for_target=True)

        conservative = league.features()
        trusting = league.features(FeatureConfig(trust_provider_absence_timing=True))

        assert conservative["away_absences_out"] is None
        assert trusting["away_absences_out"] == 1


class TestWholeDatasetCanary:
    def test_appending_a_future_season_changes_no_existing_features(
        self, league: League, session_factory: sessionmaker[Session]
    ) -> None:
        builder = FeatureBuilder(session_factory)
        with session_factory() as session:
            targets = load_targets(session)
        before = {row.target.match_id: row.values for row in builder.build(targets)}

        league.finish_target((3, 3))
        for week in range(1, 11):
            kickoff = KICKOFF + week * 7 * DAY
            match_id = league.add_match("Brighton", "Arsenal", kickoff, (week % 4, 1))
            league.add_xg(match_id, "Brighton", "Arsenal", (2.0, 1.0))
            league.add_match("Derby", "Chelsea", kickoff, (0, week % 3))

        after = {row.target.match_id: row.values for row in builder.build(targets)}
        assert after == before

    def test_batch_and_single_match_generation_agree(
        self, league: League, session_factory: sessionmaker[Session]
    ) -> None:
        builder = FeatureBuilder(session_factory)
        with session_factory() as session:
            targets = load_targets(session)

        batch = {row.target.match_id: row.values for row in builder.build(targets)}
        single = {target.match_id: builder.build([target])[0].values for target in targets}

        assert batch == single
        assert len(batch) == 13
