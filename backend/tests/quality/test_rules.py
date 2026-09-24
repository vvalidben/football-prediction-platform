from datetime import UTC, date, datetime

import pytest

from app.models.enums import IssueSeverity, IssueType, MatchStatus
from app.providers.schemas import ProviderMatch, ProviderTeamMatchStatistics, ScorePair
from app.quality.issues import IssueDraft
from app.quality.rules import kickoff_is_corrupt, match_warnings, statistics_violations
from tests.providers import builders

NOW = datetime(2026, 9, 22, tzinfo=UTC)
SEASON_START, SEASON_END = date(2023, 8, 11), date(2024, 5, 19)


def warnings(record: ProviderMatch) -> list[IssueDraft]:
    return match_warnings(record, season_start=SEASON_START, season_end=SEASON_END, now=NOW)


def keys(record: ProviderMatch) -> set[str]:
    return {issue.key for issue in warnings(record)}


def test_clean_match_has_no_issues() -> None:
    assert warnings(builders.match()) == []


@pytest.mark.parametrize(
    ("overrides", "key", "severity"),
    [
        (
            {"score": ScorePair(home=1, away=0), "half_time_score": ScorePair(home=2, away=0)},
            "half_time_exceeds_full_time",
            IssueSeverity.ERROR,
        ),
        ({"score": ScorePair(home=21, away=0)}, "implausible_goals", IssueSeverity.WARNING),
        (
            {"score": ScorePair(home=2, away=1), "penalty_score": ScorePair(home=4, away=3)},
            "shootout_without_draw",
            IssueSeverity.ERROR,
        ),
        (
            {"score": ScorePair(home=1, away=1), "penalty_score": ScorePair(home=3, away=3)},
            "shootout_tied",
            IssueSeverity.ERROR,
        ),
        (
            {"status": MatchStatus.POSTPONED, "score": ScorePair(home=1, away=0)},
            "score_for_unplayed_match",
            IssueSeverity.WARNING,
        ),
        (
            {"kickoff_at": datetime(2023, 3, 1, tzinfo=UTC)},
            "before_season",
            IssueSeverity.WARNING,
        ),
        (
            {"kickoff_at": datetime(2024, 9, 1, tzinfo=UTC)},
            "after_season",
            IssueSeverity.WARNING,
        ),
        (
            {"kickoff_at": datetime(2027, 1, 1, tzinfo=UTC)},
            "finished_in_future",
            IssueSeverity.ERROR,
        ),
    ],
)
def test_questionable_matches_are_flagged(
    overrides: dict[str, object], key: str, severity: IssueSeverity
) -> None:
    issues = warnings(builders.match(**overrides))

    flagged = {issue.key: issue for issue in issues}
    assert key in flagged
    assert flagged[key].severity is severity
    assert flagged[key].external_id == "1035037"


def test_kickoff_within_tolerance_of_season_is_fine() -> None:
    assert keys(builders.match(kickoff_at=datetime(2023, 7, 15, tzinfo=UTC))) == set()


def test_season_without_dates_skips_date_checks() -> None:
    record = builders.match(kickoff_at=datetime(2019, 1, 1, tzinfo=UTC))

    assert match_warnings(record, season_start=None, season_end=None, now=NOW) == []


@pytest.mark.parametrize(
    "kickoff", [datetime(1700, 1, 1, tzinfo=UTC), datetime(2035, 1, 1, tzinfo=UTC)]
)
def test_corrupt_kickoffs_keep_the_whole_record(kickoff: datetime) -> None:
    issue = kickoff_is_corrupt(builders.match(kickoff_at=kickoff), NOW)

    assert issue is not None
    assert issue.issue_type is IssueType.INVALID_DATE
    assert issue.severity is IssueSeverity.ERROR
    assert issue.details["provider_record"]["external_id"] == "1035037"


def test_plausible_kickoff_is_not_corrupt() -> None:
    assert kickoff_is_corrupt(builders.match(), NOW) is None


def test_contradictory_statistics_are_violations() -> None:
    record = ProviderTeamMatchStatistics(
        match_external_id="1", team_external_id="2", shots=3, shots_on_target=5, passes=10
    )

    (violation,) = statistics_violations(record)

    assert violation.key == "sot_gt_shots"
    assert violation.details["provider_record"]["shots_on_target"] == 5


def test_partial_statistics_are_not_violations() -> None:
    record = ProviderTeamMatchStatistics(match_external_id="1", team_external_id="2", shots=3)

    assert statistics_violations(record) == []
