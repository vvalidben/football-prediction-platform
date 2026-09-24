"""Read queries behind the competition, team and match endpoints."""

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.models import Competition, Match, Prediction, Season, Team
from app.models.enums import MatchStatus
from app.schemas.catalog import (
    CompetitionRead,
    MatchDetail,
    MatchRead,
    MatchTeam,
    SeasonRead,
    TeamDetail,
    TeamMatchRead,
    TeamRead,
    TeamRecord,
)

#: Matches that have not been played yet.
UPCOMING_STATUSES = (MatchStatus.SCHEDULED, MatchStatus.POSTPONED)


def count(session: Session, query: Select[Any]) -> int:
    return session.scalar(select(func.count()).select_from(query.subquery())) or 0


def competitions(
    session: Session, *, country: str | None, limit: int, offset: int
) -> tuple[int, list[CompetitionRead]]:
    query = select(Competition).order_by(Competition.name, Competition.id)
    if country:
        query = query.where(func.lower(Competition.country) == country.lower())
    total = count(session, query)
    rows = list(session.scalars(query.limit(limit).offset(offset)))
    current = {
        season.competition_id: season.name
        for season in session.scalars(
            select(Season).where(Season.competition_id.in_([c.id for c in rows]), Season.is_current)
        )
    }
    return total, [
        CompetitionRead(
            id=row.id,
            name=row.name,
            type=row.type.value,
            country=row.country,
            current_season=current.get(row.id),
        )
        for row in rows
    ]


def seasons(
    session: Session, *, competition_id: int | None, limit: int, offset: int
) -> tuple[int, list[SeasonRead]]:
    """Seasons, newest first, so a caller can offer them as a filter."""
    query = (
        select(Season, Competition)
        .join(Competition, Competition.id == Season.competition_id)
        .order_by(Season.start_date.desc().nullslast(), Season.name.desc(), Season.id.desc())
    )
    if competition_id is not None:
        query = query.where(Season.competition_id == competition_id)
    total = count(session, query)
    rows = session.execute(query.limit(limit).offset(offset)).all()
    return total, [
        SeasonRead(
            id=season.id,
            competition_id=season.competition_id,
            competition=competition.name,
            name=season.name,
            start_date=season.start_date,
            end_date=season.end_date,
            is_current=season.is_current,
        )
        for season, competition in rows
    ]


def teams(
    session: Session,
    *,
    search: str | None,
    competition_id: int | None,
    limit: int,
    offset: int,
) -> tuple[int, list[TeamRead]]:
    query = select(Team).order_by(Team.name, Team.id)
    if search:
        pattern = f"%{search.lower()}%"
        query = query.where(
            or_(func.lower(Team.name).like(pattern), func.lower(Team.short_name).like(pattern))
        )
    if competition_id is not None:
        playing = (
            select(Match.home_team_id)
            .where(Match.competition_id == competition_id)
            .union(select(Match.away_team_id).where(Match.competition_id == competition_id))
        )
        query = query.where(Team.id.in_(playing))
    total = count(session, query)
    rows = session.scalars(query.limit(limit).offset(offset))
    return total, [
        TeamRead(id=row.id, name=row.name, short_name=row.short_name, country=row.country)
        for row in rows
    ]


def _match_query(
    *,
    competition_id: int | None = None,
    team_id: int | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
) -> Select[tuple[Match]]:
    # `match_read` names the competition, the season and both teams, so they are
    # loaded with the matches rather than one query per row.
    query = select(Match).options(
        joinedload(Match.competition),
        joinedload(Match.season),
        joinedload(Match.home_team),
        joinedload(Match.away_team),
    )
    if competition_id is not None:
        query = query.where(Match.competition_id == competition_id)
    if team_id is not None:
        query = query.where(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
    if from_date is not None:
        query = query.where(
            Match.kickoff_at >= datetime.combine(from_date, datetime.min.time(), tzinfo=UTC)
        )
    if to_date is not None:
        query = query.where(
            Match.kickoff_at
            < datetime.combine(to_date, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
        )
    return query


def upcoming_matches(
    session: Session,
    *,
    competition_id: int | None,
    team_id: int | None,
    from_date: date | None,
    to_date: date | None,
    limit: int,
    offset: int,
    now: datetime,
) -> tuple[int, list[MatchRead]]:
    query = _match_query(
        competition_id=competition_id, team_id=team_id, from_date=from_date, to_date=to_date
    ).where(Match.kickoff_at >= now, Match.status.in_(UPCOMING_STATUSES))
    total = count(session, query)
    rows = session.scalars(query.order_by(Match.kickoff_at, Match.id).limit(limit).offset(offset))
    return total, [match_read(row) for row in rows]


def match_read(row: Match) -> MatchRead:
    return MatchRead(
        id=row.id,
        kickoff_at=row.kickoff_at,
        status=row.status.value,
        competition=MatchTeam(id=row.competition.id, name=row.competition.name),
        season=row.season.name if row.season else None,
        matchday=row.matchday,
        home_team=MatchTeam(id=row.home_team.id, name=row.home_team.name),
        away_team=MatchTeam(id=row.away_team.id, name=row.away_team.name),
        home_score=row.home_score,
        away_score=row.away_score,
    )


def find_match(session: Session, match_id: int) -> Match | None:
    """One match with everything `match_read` names, in a single query."""
    return session.scalars(
        select(Match)
        .options(
            joinedload(Match.competition),
            joinedload(Match.season),
            joinedload(Match.home_team),
            joinedload(Match.away_team),
        )
        .where(Match.id == match_id)
    ).one_or_none()


def match_detail(session: Session, row: Match) -> MatchDetail:
    stored = session.scalars(
        select(Prediction)
        .options(joinedload(Prediction.model_version))  # each one names its version
        .where(Prediction.match_id == row.id)
        .order_by(Prediction.predicted_at.desc())
        .limit(10)
    )
    half_time = (
        {"home": row.home_half_time_score, "away": row.away_half_time_score}
        if row.home_half_time_score is not None and row.away_half_time_score is not None
        else None
    )
    return MatchDetail(
        **match_read(row).model_dump(),
        venue=row.venue,
        referee=row.referee,
        half_time_score=half_time,
        predictions=[
            {
                "id": prediction.id,
                "model": prediction.model_version.name,
                "version": prediction.model_version.version,
                "predicted_at": prediction.predicted_at.isoformat(),
                "home_win_probability": float(prediction.home_win_probability),
                "draw_probability": float(prediction.draw_probability),
                "away_win_probability": float(prediction.away_win_probability),
            }
            for prediction in stored
        ],
    )


def matches(
    session: Session,
    *,
    competition_id: int | None,
    team_id: int | None,
    status: str | None,
    from_date: date | None,
    to_date: date | None,
    limit: int,
    offset: int,
) -> tuple[int, list[MatchRead]]:
    """Matches newest first, optionally filtered by status."""
    query = _match_query(
        competition_id=competition_id, team_id=team_id, from_date=from_date, to_date=to_date
    )
    if status is not None:
        try:
            query = query.where(Match.status == MatchStatus(status))
        except ValueError:
            known = ", ".join(sorted(member.value for member in MatchStatus))
            raise ValueError(f"unknown status {status!r}; known: {known}") from None
    total = count(session, query)
    rows = session.scalars(
        query.order_by(Match.kickoff_at.desc(), Match.id.desc()).limit(limit).offset(offset)
    )
    return total, [match_read(row) for row in rows]


def team_detail(session: Session, team: Team, *, matches: int) -> TeamDetail:
    from app.services import analytics

    rows = analytics.team_matches(session, team.id, limit=matches)
    record = analytics.team_summary(session, team.id, window=max(matches, 10))
    return TeamDetail(
        team=TeamRead(id=team.id, name=team.name, short_name=team.short_name, country=team.country),
        record=TeamRecord.model_validate(record),
        recent_matches=[match_read(row.match) for row in rows],
        trend=[TeamMatchRead.model_validate(entry) for entry in analytics.team_trend(rows)],
    )
