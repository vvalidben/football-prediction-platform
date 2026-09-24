"""Competitions, teams and matches."""

import logging
from datetime import UTC, date, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.dependencies import DbSessionDep
from app.models import Team
from app.schemas.catalog import (
    CompetitionPage,
    MatchDetail,
    MatchPage,
    SeasonPage,
    TeamDetail,
    TeamPage,
)
from app.services import catalog

logger = logging.getLogger(__name__)
router = APIRouter(tags=["catalog"])

LimitDep = Annotated[int, Query(ge=1, le=100, description="Rows to return")]
OffsetDep = Annotated[int, Query(ge=0, description="Rows to skip")]


@router.get("/competitions", response_model=CompetitionPage, summary="List competitions")
def list_competitions(
    session: DbSessionDep,
    country: Annotated[str | None, Query(max_length=80, description="Filter by country")] = None,
    limit: LimitDep = 50,
    offset: OffsetDep = 0,
) -> CompetitionPage:
    """Competitions known to the platform, with the current season when there is one."""
    total, items = catalog.competitions(session, country=country, limit=limit, offset=offset)
    return CompetitionPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/seasons", response_model=SeasonPage, summary="List seasons")
def list_seasons(
    session: DbSessionDep,
    competition_id: Annotated[int | None, Query(description="Seasons of one competition")] = None,
    limit: LimitDep = 50,
    offset: OffsetDep = 0,
) -> SeasonPage:
    """Seasons, newest first. They identify a season for the history filters."""
    total, items = catalog.seasons(
        session, competition_id=competition_id, limit=limit, offset=offset
    )
    return SeasonPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/teams", response_model=TeamPage, summary="List teams")
def list_teams(
    session: DbSessionDep,
    search: Annotated[
        str | None, Query(max_length=120, description="Match part of a team's name")
    ] = None,
    competition_id: Annotated[int | None, Query(description="Teams playing in it")] = None,
    limit: LimitDep = 50,
    offset: OffsetDep = 0,
) -> TeamPage:
    total, items = catalog.teams(
        session, search=search, competition_id=competition_id, limit=limit, offset=offset
    )
    return TeamPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/matches/upcoming", response_model=MatchPage, summary="Matches not yet played")
def upcoming_matches(
    session: DbSessionDep,
    competition_id: Annotated[int | None, Query()] = None,
    team_id: Annotated[int | None, Query()] = None,
    from_date: Annotated[date | None, Query(description="Earliest kickoff date")] = None,
    to_date: Annotated[date | None, Query(description="Latest kickoff date (inclusive)")] = None,
    limit: LimitDep = 50,
    offset: OffsetDep = 0,
) -> MatchPage:
    """Scheduled and postponed matches with a kickoff in the future, earliest first."""
    total, items = catalog.upcoming_matches(
        session,
        competition_id=competition_id,
        team_id=team_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
        now=datetime.now(UTC),
    )
    return MatchPage(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/matches/{match_id}",
    response_model=MatchDetail,
    summary="One match",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such match"}},
)
def read_match(session: DbSessionDep, match_id: Annotated[int, Path(ge=1)]) -> MatchDetail:
    """A match with its teams, competition, result so far and stored predictions."""
    row = catalog.find_match(session, match_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"match {match_id} was not found"
        )
    return catalog.match_detail(session, row)


@router.get("/matches", response_model=MatchPage, summary="Matches, played or scheduled")
def list_matches(
    session: DbSessionDep,
    competition_id: Annotated[int | None, Query()] = None,
    team_id: Annotated[int | None, Query()] = None,
    match_status: Annotated[
        str | None, Query(alias="status", description="e.g. finished or scheduled")
    ] = None,
    from_date: Annotated[date | None, Query()] = None,
    to_date: Annotated[date | None, Query()] = None,
    limit: LimitDep = 20,
    offset: OffsetDep = 0,
) -> MatchPage:
    """Matches with the most recent first; the same filters as `/matches/upcoming`."""
    try:
        total, items = catalog.matches(
            session,
            competition_id=competition_id,
            team_id=team_id,
            status=match_status,
            from_date=from_date,
            to_date=to_date,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MatchPage(total=total, limit=limit, offset=offset, items=items)


@router.get(
    "/teams/{team_id}",
    response_model=TeamDetail,
    summary="A team with its record and recent form",
    responses={status.HTTP_404_NOT_FOUND: {"description": "No such team"}},
)
def read_team(
    session: DbSessionDep,
    team_id: Annotated[int, Path(ge=1)],
    matches: Annotated[int, Query(ge=1, le=50, description="Recent matches to include")] = 10,
) -> TeamDetail:
    """Scoring and conceding overall and by venue, recent results, and a goals trend.

    Expected goals appear only where the provider supplied them; the response says
    how many of the recent matches have them.
    """
    team = session.get(Team, team_id)
    if team is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"team {team_id} was not found"
        )
    return catalog.team_detail(session, team, matches=matches)
