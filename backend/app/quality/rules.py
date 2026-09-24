"""Validation rules applied to provider records during ingestion.

Policy: a record is **stored and flagged** when it is questionable but storable
(e.g. an implausible score). It is **not stored** only when storing it would break
the schema or the timeline (e.g. no timezone, a kickoff decades away); then the
issue keeps the full record in `details`, so nothing disappears silently.
"""

from datetime import date, datetime, timedelta

from app.models.enums import IssueEntity, IssueSeverity, IssueType, MatchStatus
from app.providers.schemas import ProviderMatch, ProviderTeamMatchStatistics
from app.quality.issues import IssueDraft

#: Kickoffs outside this range are treated as corrupt, not merely odd.
EARLIEST_PLAUSIBLE_KICKOFF = datetime.fromisoformat("1860-01-01T00:00:00+00:00")
MAX_YEARS_AHEAD = 3
#: A kickoff this far outside its season's dates is flagged.
SEASON_DATE_TOLERANCE = timedelta(days=45)
#: More goals than this for one team in one match is flagged as implausible.
MAX_PLAUSIBLE_GOALS = 15
NOT_PLAYED = frozenset({MatchStatus.SCHEDULED, MatchStatus.POSTPONED, MatchStatus.CANCELLED})


def _issue(
    record: ProviderMatch,
    issue_type: IssueType,
    severity: IssueSeverity,
    key: str,
    message: str,
    **details: object,
) -> IssueDraft:
    return IssueDraft(
        issue_type=issue_type,
        severity=severity,
        entity_type=IssueEntity.MATCH,
        external_id=record.external_id,
        key=key,
        message=message,
        details={"kickoff_at": record.kickoff_at.isoformat(), **details},
    )


def kickoff_is_corrupt(record: ProviderMatch, now: datetime) -> IssueDraft | None:
    """A kickoff so far from reality that storing it would poison the timeline."""
    latest = now + timedelta(days=365 * MAX_YEARS_AHEAD)
    if EARLIEST_PLAUSIBLE_KICKOFF <= record.kickoff_at <= latest:
        return None
    return _issue(
        record,
        IssueType.INVALID_DATE,
        IssueSeverity.ERROR,
        "kickoff_out_of_range",
        f"kickoff {record.kickoff_at.isoformat()} is outside the plausible range; not stored",
        provider_record=record.model_dump(mode="json"),
    )


def match_warnings(
    record: ProviderMatch,
    *,
    season_start: date | None,
    season_end: date | None,
    now: datetime,
) -> list[IssueDraft]:
    """Checks for a storable match. Every returned issue is attached to the stored row."""
    issues: list[IssueDraft] = []
    kickoff_day = record.kickoff_at.date()
    if season_start and kickoff_day < season_start - SEASON_DATE_TOLERANCE:
        issues.append(
            _issue(
                record,
                IssueType.INVALID_DATE,
                IssueSeverity.WARNING,
                "before_season",
                f"kickoff {kickoff_day} is well before the season start {season_start}",
                season_start=season_start.isoformat(),
            )
        )
    if season_end and kickoff_day > season_end + SEASON_DATE_TOLERANCE:
        issues.append(
            _issue(
                record,
                IssueType.INVALID_DATE,
                IssueSeverity.WARNING,
                "after_season",
                f"kickoff {kickoff_day} is well after the season end {season_end}",
                season_end=season_end.isoformat(),
            )
        )
    if record.status is MatchStatus.FINISHED and record.kickoff_at > now:
        issues.append(
            _issue(
                record,
                IssueType.INVALID_DATE,
                IssueSeverity.ERROR,
                "finished_in_future",
                "match is marked finished but kicks off in the future",
            )
        )

    score, half, pens = record.score, record.half_time_score, record.penalty_score
    if score and max(score.home, score.away) > MAX_PLAUSIBLE_GOALS:
        issues.append(
            _issue(
                record,
                IssueType.IMPOSSIBLE_SCORE,
                IssueSeverity.WARNING,
                "implausible_goals",
                f"implausible score {score.home}-{score.away}",
            )
        )
    if score and half and (half.home > score.home or half.away > score.away):
        issues.append(
            _issue(
                record,
                IssueType.IMPOSSIBLE_SCORE,
                IssueSeverity.ERROR,
                "half_time_exceeds_full_time",
                f"half-time {half.home}-{half.away} exceeds final score {score.home}-{score.away}",
            )
        )
    if pens and score and score.home != score.away:
        issues.append(
            _issue(
                record,
                IssueType.IMPOSSIBLE_SCORE,
                IssueSeverity.ERROR,
                "shootout_without_draw",
                f"penalty shoot-out recorded after a decisive score {score.home}-{score.away}",
            )
        )
    if pens and pens.home == pens.away:
        issues.append(
            _issue(
                record,
                IssueType.IMPOSSIBLE_SCORE,
                IssueSeverity.ERROR,
                "shootout_tied",
                f"penalty shoot-out cannot end level ({pens.home}-{pens.away})",
            )
        )
    if record.status in NOT_PLAYED and score is not None:
        issues.append(
            _issue(
                record,
                IssueType.IMPOSSIBLE_SCORE,
                IssueSeverity.WARNING,
                "score_for_unplayed_match",
                f"{record.status.value} match has a score {score.home}-{score.away}",
            )
        )
    return issues


def statistics_violations(record: ProviderTeamMatchStatistics) -> list[IssueDraft]:
    """Contradictions the schema's CHECK constraints would reject. A record with any
    of these is not stored; the issue carries the record instead."""
    issues = []

    def check(key: str, part: int | None, whole: int | None, label: str) -> None:
        if part is not None and whole is not None and part > whole:
            issues.append(
                IssueDraft(
                    issue_type=IssueType.INCONSISTENT_STATISTICS,
                    severity=IssueSeverity.ERROR,
                    entity_type=IssueEntity.TEAM_MATCH_STATISTICS,
                    external_id=record.external_id,
                    key=key,
                    message=f"{label} ({part} > {whole}); not stored",
                    details={"provider_record": record.model_dump(mode="json")},
                )
            )

    check("sot_gt_shots", record.shots_on_target, record.shots, "shots on target exceed shots")
    check(
        "completed_gt_passes",
        record.passes_completed,
        record.passes,
        "completed passes exceed passes",
    )
    return issues
