"""Enumerations stored as CHECK-constrained text columns (see `mixins.str_enum`)."""

from enum import StrEnum


class CompetitionType(StrEnum):
    LEAGUE = "league"
    CUP = "cup"
    SUPER_CUP = "super_cup"
    FRIENDLY = "friendly"
    OTHER = "other"


class MatchStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    POSTPONED = "postponed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


class PlayerPosition(StrEnum):
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"


class MatchPeriod(StrEnum):
    FIRST_HALF = "first_half"
    SECOND_HALF = "second_half"
    EXTRA_TIME_FIRST_HALF = "extra_time_first_half"
    EXTRA_TIME_SECOND_HALF = "extra_time_second_half"
    PENALTY_SHOOTOUT = "penalty_shootout"


class MatchEventType(StrEnum):
    GOAL = "goal"
    OWN_GOAL = "own_goal"
    PENALTY_GOAL = "penalty_goal"
    PENALTY_MISSED = "penalty_missed"
    YELLOW_CARD = "yellow_card"
    SECOND_YELLOW_CARD = "second_yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    VAR_DECISION = "var_decision"
    OTHER = "other"


class AvailabilityStatus(StrEnum):
    """How likely an injured or suspended player is to miss a match."""

    OUT = "out"
    DOUBTFUL = "doubtful"


class AbsenceType(StrEnum):
    INJURY = "injury"
    ILLNESS = "illness"
    SUSPENSION = "suspension"
    OTHER = "other"


class OddsMarket(StrEnum):
    MATCH_WINNER = "match_winner"
    DOUBLE_CHANCE = "double_chance"
    OVER_UNDER = "over_under"
    BOTH_TEAMS_TO_SCORE = "both_teams_to_score"
    CORRECT_SCORE = "correct_score"
    ASIAN_HANDICAP = "asian_handicap"


class JobStatus(StrEnum):
    """Outcome of one scheduled background job."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"  # nothing to do, or a precondition was not met
    FAILED = "failed"


class IngestionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class IssueSeverity(StrEnum):
    ERROR = "error"  # wrong or unusable; exclude from modelling until resolved
    WARNING = "warning"  # suspicious; review
    INFO = "info"  # expected gap, e.g. statistics a provider does not offer


class IssueType(StrEnum):
    MALFORMED_RECORD = "malformed_record"
    MISSING_TEAM = "missing_team"
    DUPLICATE_MATCH = "duplicate_match"
    DUPLICATE_PROVIDER_ID = "duplicate_provider_id"
    POSSIBLE_DUPLICATE_TEAM = "possible_duplicate_team"
    IMPOSSIBLE_SCORE = "impossible_score"
    INVALID_DATE = "invalid_date"
    MISSING_STATISTICS = "missing_statistics"
    INCONSISTENT_STATISTICS = "inconsistent_statistics"
    INCONSISTENT_COMPETITION = "inconsistent_competition"
    INCOMPLETE_RECORD = "incomplete_record"


class IssueEntity(StrEnum):
    COMPETITION = "competition"
    SEASON = "season"
    TEAM = "team"
    MATCH = "match"
    TEAM_MATCH_STATISTICS = "team_match_statistics"
