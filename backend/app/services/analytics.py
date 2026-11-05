"""Aggregates behind the team, model-performance and data-status pages.

Everything here is computed from what the database actually holds. Where a figure
cannot be computed (no statistics imported, nothing evaluated yet), the field is
`None` or the list is empty, so the interface can say so instead of inventing it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, case, func, or_, select
from sqlalchemy.orm import Session

from app.features.config import FeatureConfig
from app.models import (
    Competition,
    DataIngestionRun,
    DataQualityIssue,
    DataSource,
    FeatureSnapshot,
    JobRun,
    Match,
    ModelVersion,
    Prediction,
    PredictionResult,
    Season,
    TeamMatchStatistics,
)
from app.models.enums import IngestionStatus, IssueSeverity, MatchStatus

#: Reliability bins for the calibration chart.
CALIBRATION_BINS = 10


@dataclass(frozen=True)
class TeamMatchRow:
    match: Match
    is_home: bool
    goals_for: int
    goals_against: int
    result: str  # "W", "D" or "L"
    expected_goals_for: float | None
    expected_goals_against: float | None


def team_matches(session: Session, team_id: int, *, limit: int = 20) -> list[TeamMatchRow]:
    """The team's most recent finished matches, newest first, with its own xG."""
    rows = session.scalars(
        select(Match)
        .where(
            or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            Match.status == MatchStatus.FINISHED,
            Match.home_score.is_not(None),
        )
        .order_by(Match.kickoff_at.desc())
        .limit(limit)
    ).all()
    statistics = {
        (row.match_id, row.team_id): row
        for row in session.scalars(
            select(TeamMatchStatistics).where(
                TeamMatchStatistics.match_id.in_([match.id for match in rows])
            )
        )
    }
    result = []
    for match in rows:
        is_home = match.home_team_id == team_id
        goals_for = (match.home_score if is_home else match.away_score) or 0
        goals_against = (match.away_score if is_home else match.home_score) or 0
        opponent = match.away_team_id if is_home else match.home_team_id
        own = statistics.get((match.id, team_id))
        other = statistics.get((match.id, opponent))
        result.append(
            TeamMatchRow(
                match=match,
                is_home=is_home,
                goals_for=goals_for,
                goals_against=goals_against,
                result="W"
                if goals_for > goals_against
                else "D"
                if goals_for == goals_against
                else "L",
                expected_goals_for=float(own.expected_goals)
                if own and own.expected_goals
                else None,
                expected_goals_against=(
                    float(other.expected_goals) if other and other.expected_goals else None
                ),
            )
        )
    return result


def team_summary(session: Session, team_id: int, *, window: int = 20) -> dict[str, Any]:
    """Scoring, conceding and home/away splits, over the team's whole history."""
    home, away = (
        _side_totals(session, team_id, home=True),
        _side_totals(session, team_id, home=False),
    )
    played = home["played"] + away["played"]
    scored = home["scored"] + away["scored"]
    conceded = home["conceded"] + away["conceded"]
    recent = team_matches(session, team_id, limit=window)
    with_xg = [row for row in recent if row.expected_goals_for is not None]
    return {
        "played": played,
        "wins": home["wins"] + away["wins"],
        "draws": home["draws"] + away["draws"],
        "losses": home["losses"] + away["losses"],
        "goals_for": scored,
        "goals_against": conceded,
        "goals_for_per_match": round(scored / played, 3) if played else None,
        "goals_against_per_match": round(conceded / played, 3) if played else None,
        "clean_sheets": home["clean_sheets"] + away["clean_sheets"],
        "home": home,
        "away": away,
        "form": [row.result for row in recent[:5]],
        "expected_goals_for_per_match": (
            round(sum(r.expected_goals_for or 0 for r in with_xg) / len(with_xg), 3)
            if with_xg
            else None
        ),
        "expected_goals_against_per_match": (
            round(
                sum(r.expected_goals_against or 0 for r in with_xg if r.expected_goals_against)
                / len(with_xg),
                3,
            )
            if with_xg
            else None
        ),
        "matches_with_expected_goals": len(with_xg),
    }


def _side_totals(session: Session, team_id: int, *, home: bool) -> dict[str, Any]:
    scored_column = Match.home_score if home else Match.away_score
    conceded_column = Match.away_score if home else Match.home_score
    side = (Match.home_team_id if home else Match.away_team_id) == team_id
    row = session.execute(
        select(
            func.count(),
            func.coalesce(func.sum(scored_column), 0),
            func.coalesce(func.sum(conceded_column), 0),
            func.count(case((scored_column > conceded_column, 1))),
            func.count(case((scored_column == conceded_column, 1))),
            func.count(case((scored_column < conceded_column, 1))),
            func.count(case((conceded_column == 0, 1))),
        ).where(side, Match.status == MatchStatus.FINISHED, Match.home_score.is_not(None))
    ).one()
    played, scored, conceded, wins, draws, losses, clean_sheets = row
    return {
        "side": "home" if home else "away",
        "played": int(played),
        "scored": int(scored),
        "conceded": int(conceded),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "clean_sheets": int(clean_sheets),
        "goals_for_per_match": round(int(scored) / played, 3) if played else None,
        "goals_against_per_match": round(int(conceded) / played, 3) if played else None,
    }


def team_trend(rows: Sequence[TeamMatchRow]) -> list[dict[str, Any]]:
    """Oldest first, with a five-match rolling average of goals for and against."""
    ordered = list(reversed(rows))
    trend = []
    for index, row in enumerate(ordered):
        window = ordered[max(0, index - 4) : index + 1]
        trend.append(
            {
                "match_id": row.match.id,
                "kickoff_at": row.match.kickoff_at,
                "opponent": (row.match.away_team if row.is_home else row.match.home_team).name,
                "is_home": row.is_home,
                "goals_for": row.goals_for,
                "goals_against": row.goals_against,
                "result": row.result,
                "expected_goals_for": row.expected_goals_for,
                "rolling_goals_for": round(sum(r.goals_for for r in window) / len(window), 3),
                "rolling_goals_against": round(
                    sum(r.goals_against for r in window) / len(window), 3
                ),
            }
        )
    return trend


def model_performance(session: Session, model_version_id: int) -> dict[str, Any]:
    """Accuracy, scores, calibration and a monthly series from evaluated predictions."""
    rows = session.execute(
        select(Prediction, PredictionResult, Match)
        .join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
        .join(Match, Match.id == Prediction.match_id)
        .where(Prediction.model_version_id == model_version_id)
        .order_by(Match.kickoff_at)
    ).all()
    if not rows:
        return {
            "evaluated": 0,
            "accuracy": None,
            "log_loss": None,
            "brier": None,
            "ranked_probability_score": None,
            "exact_score_accuracy": None,
            "calibration": [],
            "monthly": [],
            "first_match": None,
            "last_match": None,
        }
    correct = sum(1 for _, result, _ in rows if result.outcome_correct)
    exact = sum(1 for _, result, _ in rows if result.exact_score_correct)
    monthly: dict[str, dict[str, float]] = {}
    bins: list[dict[str, float]] = [
        {
            "low": index / CALIBRATION_BINS,
            "high": (index + 1) / CALIBRATION_BINS,
            "count": 0.0,
            "predicted": 0.0,
            "observed": 0.0,
        }
        for index in range(CALIBRATION_BINS)
    ]
    for prediction, result, match in rows:
        month = f"{match.kickoff_at:%Y-%m}"
        entry = monthly.setdefault(month, {"matches": 0.0, "correct": 0.0, "log_loss": 0.0})
        entry["matches"] += 1
        entry["correct"] += 1 if result.outcome_correct else 0
        entry["log_loss"] += float(result.log_loss)
        # Reliability over all three outcomes: each prediction contributes three points.
        outcome = _outcome(match)
        for index, probability in enumerate(
            (
                float(prediction.home_win_probability),
                float(prediction.draw_probability),
                float(prediction.away_win_probability),
            )
        ):
            bucket = bins[min(int(probability * CALIBRATION_BINS), CALIBRATION_BINS - 1)]
            bucket["count"] += 1
            bucket["predicted"] += probability
            bucket["observed"] += 1.0 if index == outcome else 0.0
    calibration = [
        {
            "low": bucket["low"],
            "high": bucket["high"],
            "count": int(bucket["count"]),
            "mean_predicted": round(bucket["predicted"] / bucket["count"], 4),
            "observed": round(bucket["observed"] / bucket["count"], 4),
        }
        for bucket in bins
        if bucket["count"]
    ]
    return {
        "evaluated": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "log_loss": round(sum(float(r.log_loss) for _, r, _ in rows) / len(rows), 4),
        "brier": round(sum(float(r.brier_score) for _, r, _ in rows) / len(rows), 4),
        "ranked_probability_score": round(
            sum(float(r.ranked_probability_score) for _, r, _ in rows) / len(rows), 4
        ),
        "exact_score_accuracy": round(exact / len(rows), 4),
        "calibration": calibration,
        "monthly": [
            {
                "month": month,
                "matches": int(entry["matches"]),
                "accuracy": round(entry["correct"] / entry["matches"], 4),
                "log_loss": round(entry["log_loss"] / entry["matches"], 4),
            }
            for month, entry in sorted(monthly.items())
        ],
        "first_match": rows[0][2].kickoff_at,
        "last_match": rows[-1][2].kickoff_at,
    }


def _outcome(match: Match) -> int:
    home, away = match.home_score or 0, match.away_score or 0
    return 0 if home > away else 1 if home == away else 2


def data_status(session: Session, *, now: datetime | None = None) -> dict[str, Any]:
    """Synchronisation, coverage, gaps and provider state, for the data page."""
    now = now or datetime.now(UTC)
    version = FeatureConfig().version()
    ingestion = [
        {
            "job": job,
            "status": status.value,
            "last_run_at": last,
            "records_inserted": int(inserted or 0),
        }
        for job, status, last, inserted in session.execute(
            select(
                DataIngestionRun.job,
                DataIngestionRun.status,
                func.max(DataIngestionRun.started_at),
                func.sum(DataIngestionRun.records_inserted),
            ).group_by(DataIngestionRun.job, DataIngestionRun.status)
        ).all()
    ]
    jobs = [
        {"task": task, "status": status.value, "last_run_at": last}
        for task, status, last in session.execute(
            select(JobRun.task, JobRun.status, func.max(JobRun.started_at)).group_by(
                JobRun.task, JobRun.status
            )
        ).all()
    ]
    coverage = [
        {
            "competition_id": competition_id,
            "competition": name,
            "seasons": int(seasons or 0),
            "matches": int(matches or 0),
            "finished": int(finished or 0),
            "with_statistics": 0,
            "with_features": 0,
        }
        for competition_id, name, seasons, matches, finished in session.execute(
            select(
                Competition.id,
                Competition.name,
                func.count(func.distinct(Season.id)),
                func.count(func.distinct(Match.id)),
                func.count(func.distinct(case((Match.status == MatchStatus.FINISHED, Match.id)))),
            )
            .select_from(Competition)
            .outerjoin(Season, Season.competition_id == Competition.id)
            .outerjoin(Match, Match.season_id == Season.id)
            .group_by(Competition.id, Competition.name)
            .order_by(Competition.name)
        ).all()
    ]
    statistics_by_competition: dict[int, int] = {
        competition_id: int(count)
        for competition_id, count in session.execute(
            select(Match.competition_id, func.count(func.distinct(Match.id)))
            .join(TeamMatchStatistics, TeamMatchStatistics.match_id == Match.id)
            .group_by(Match.competition_id)
        ).all()
    }
    features_by_competition: dict[int, int] = {
        competition_id: int(count)
        for competition_id, count in session.execute(
            select(Match.competition_id, func.count(func.distinct(Match.id)))
            .join(
                FeatureSnapshot,
                and_(
                    FeatureSnapshot.match_id == Match.id,
                    FeatureSnapshot.feature_set_version == version,
                ),
            )
            .where(Match.status == MatchStatus.FINISHED)
            .group_by(Match.competition_id)
        ).all()
    }
    for entry in coverage:
        entry["with_statistics"] = int(statistics_by_competition.get(entry["competition_id"], 0))
        entry["with_features"] = int(features_by_competition.get(entry["competition_id"], 0))
    issues = [
        {"severity": severity.value, "issue_type": issue_type.value, "count": int(count)}
        for severity, issue_type, count in session.execute(
            select(DataQualityIssue.severity, DataQualityIssue.issue_type, func.count())
            .where(DataQualityIssue.resolved_at.is_(None))
            .group_by(DataQualityIssue.severity, DataQualityIssue.issue_type)
            .order_by(func.count().desc())
        ).all()
    ]
    latest_result = session.scalar(
        select(func.max(Match.kickoff_at)).where(Match.status == MatchStatus.FINISHED)
    )
    latest_snapshot = session.scalar(
        select(func.max(FeatureSnapshot.created_at)).where(
            FeatureSnapshot.feature_set_version == version
        )
    )
    last_success = session.scalar(
        select(func.max(DataIngestionRun.finished_at)).where(
            DataIngestionRun.status == IngestionStatus.SUCCEEDED
        )
    )
    return {
        "generated_at": now,
        "feature_set_version": version,
        "ingestion": sorted(ingestion, key=lambda row: str(row["job"])),
        "jobs": sorted(jobs, key=lambda row: str(row["task"])),
        "coverage": coverage,
        "issues": issues,
        "open_error_issues": sum(
            row["count"] for row in issues if row["severity"] == IssueSeverity.ERROR.value
        ),
        "last_successful_ingestion_at": last_success,
        "latest_finished_match_at": latest_result,
        "latest_feature_snapshot_at": latest_snapshot,
        "matches_missing_features": session.scalar(
            select(func.count())
            .select_from(Match)
            .where(
                Match.status == MatchStatus.FINISHED,
                ~select(FeatureSnapshot.id)
                .where(
                    FeatureSnapshot.match_id == Match.id,
                    FeatureSnapshot.feature_set_version == version,
                )
                .exists(),
            )
        ),
        "providers": [
            {
                "code": code,
                "name": name,
                "is_active": is_active,
                "last_run_at": last_run,
                "runs": int(runs or 0),
            }
            for code, name, is_active, last_run, runs in session.execute(
                select(
                    DataSource.code,
                    DataSource.name,
                    DataSource.is_active,
                    func.max(DataIngestionRun.started_at),
                    func.count(DataIngestionRun.id),
                )
                .outerjoin(DataIngestionRun, DataIngestionRun.data_source_id == DataSource.id)
                .group_by(DataSource.code, DataSource.name, DataSource.is_active)
                .order_by(DataSource.code)
            ).all()
        ],
    }


def freshness(session: Session, *, now: datetime | None = None) -> dict[str, Any]:
    """The small freshness panel on the dashboard."""
    now = now or datetime.now(UTC)
    version = FeatureConfig().version()
    latest_result = session.scalar(
        select(func.max(Match.kickoff_at)).where(Match.status == MatchStatus.FINISHED)
    )
    last_success = session.scalar(
        select(func.max(DataIngestionRun.finished_at)).where(
            DataIngestionRun.status == IngestionStatus.SUCCEEDED
        )
    )
    last_job = session.scalar(
        select(func.max(JobRun.finished_at)).where(JobRun.status == "succeeded")
    )
    snapshot = session.scalar(
        select(func.max(FeatureSnapshot.created_at)).where(
            FeatureSnapshot.feature_set_version == version
        )
    )
    return {
        "generated_at": now,
        "feature_set_version": version,
        "last_successful_ingestion_at": last_success,
        "last_successful_job_at": last_job,
        "latest_finished_match_at": latest_result,
        "latest_feature_snapshot_at": snapshot,
        "upcoming_matches": session.scalar(
            select(func.count())
            .select_from(Match)
            .where(Match.kickoff_at >= now, Match.status == MatchStatus.SCHEDULED)
        ),
        "predictions_last_24h": session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.predicted_at >= now - timedelta(hours=24))
        ),
        "active_models": session.scalar(
            select(func.count()).select_from(ModelVersion).where(ModelVersion.is_active)
        ),
    }
