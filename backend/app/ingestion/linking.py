"""Tell the database that a second provider describes rows it already has.

Each provider has its own ids, so syncing a new one would otherwise create a second
copy of the competition, of every team and of every match, and train on them twice.
This writes only id mappings, so linking never changes or deletes stored football
data, and running it again changes nothing.

Teams are matched from the fixtures both providers report: the same kickoff with the
same score is the same match, whatever each provider calls the teams. The name is
only used for teams with no shared fixture, such as a promoted side last seen before
the second provider's history begins. Seasons and matches then link themselves during
the normal sync, by season name and by teams plus kickoff.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.ingestion.sync import ProviderSync
from app.models import Competition, CompetitionExternalId, Match, Team, TeamExternalId
from app.providers.schemas import ExternalId, ProviderMatch, ProviderTeam

#: Shared fixtures a team needs before they count as evidence of who it is.
MIN_SHARED_FIXTURES = 3
#: Words that say nothing about which club this is.
NOISE_WORDS = frozenset({"fc", "afc", "cf", "sc", "ac", "the", "club"})


@dataclass
class LinkReport:
    competition_id: int
    #: provider team name -> stored team name, with how it was decided.
    by_fixtures: dict[str, str] = field(default_factory=dict)
    by_name: dict[str, str] = field(default_factory=dict)
    #: Provider teams with no stored counterpart: the sync will add them as new.
    new_teams: list[str] = field(default_factory=list)
    #: Provider teams whose evidence pointed at more than one stored team.
    ambiguous: list[str] = field(default_factory=list)
    mappings_written: int = 0


def _name_key(name: str) -> str:
    """First meaningful word: "Leeds United FC" and "Leeds" both give "leeds"."""
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w and w not in NOISE_WORDS]
    return words[0] if words else ""


def _stored_fixtures(
    session: Session, competition_id: int
) -> dict[tuple[str, int, int], list[Match]]:
    stored = session.scalars(
        select(Match)
        .where(
            Match.competition_id == competition_id,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
        )
        .options(joinedload(Match.home_team), joinedload(Match.away_team))
    ).all()
    index: dict[tuple[str, int, int], list[Match]] = defaultdict(list)
    for match in stored:
        key = (match.kickoff_at.isoformat(), match.home_score, match.away_score)
        index[key].append(match)  # type: ignore[index]
    return index


def _votes(
    matches: list[ProviderMatch], index: dict[tuple[str, int, int], list[Match]]
) -> dict[ExternalId, Counter[int]]:
    """For each provider team, how often each stored team played its fixtures."""
    votes: dict[ExternalId, Counter[int]] = defaultdict(Counter)
    for record in matches:
        if record.score is None:
            continue
        key = (record.kickoff_at.isoformat(), record.score.home, record.score.away)
        for stored in index.get(key, []):
            votes[record.home_team_external_id][stored.home_team_id] += 1
            votes[record.away_team_external_id][stored.away_team_id] += 1
    return votes


def _decide(
    teams: dict[ExternalId, ProviderTeam],
    votes: dict[ExternalId, Counter[int]],
    stored_teams: dict[int, Team],
    report: LinkReport,
) -> dict[ExternalId, int]:
    decided: dict[ExternalId, int] = {}
    for external_id, counter in votes.items():
        ranked = counter.most_common(2)
        best, count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if count < MIN_SHARED_FIXTURES or count <= 2 * runner_up:
            continue
        decided[external_id] = best

    claimed = Counter(decided.values())
    for external_id in [e for e, team_id in decided.items() if claimed[team_id] > 1]:
        del decided[external_id]
        report.ambiguous.append(teams[external_id].name if external_id in teams else external_id)
    taken = set(decided.values())
    for external_id, team_id in decided.items():
        if external_id in teams:
            report.by_fixtures[teams[external_id].name] = stored_teams[team_id].name

    by_key: dict[str, list[int]] = defaultdict(list)
    for team_id, stored in stored_teams.items():
        if team_id not in taken:
            by_key[_name_key(stored.name)].append(team_id)
    for external_id, team in teams.items():
        if external_id in decided:
            continue
        candidates = by_key.get(_name_key(team.name), [])
        if len(candidates) == 1:
            decided[external_id] = candidates[0]
            report.by_name[team.name] = stored_teams[candidates[0]].name
            by_key[_name_key(team.name)] = []
        else:
            report.new_teams.append(team.name)
    return decided


def link_competition(
    sync: ProviderSync,
    competition_external_id: ExternalId,
    stored_competition_id: int,
    *,
    dry_run: bool = False,
) -> LinkReport:
    """Map a provider's competition and teams onto rows already stored."""
    provider = sync.provider
    source_id = sync.data_source_id()
    report = LinkReport(stored_competition_id)

    teams: dict[ExternalId, ProviderTeam] = {}
    matches: list[ProviderMatch] = []
    for season in provider.get_seasons(competition_external_id).records:
        teams.update(
            (team.external_id, team)
            for team in provider.get_teams(competition_external_id, season.external_id).records
        )
        matches.extend(provider.get_matches(competition_external_id, season.external_id).records)

    with sync.session_factory() as session, session.begin():
        if session.get(Competition, stored_competition_id) is None:
            raise ValueError(f"no stored competition with id {stored_competition_id}")
        fixtures = _stored_fixtures(session, stored_competition_id)
        stored_teams: dict[int, Team] = {}
        for candidates in fixtures.values():
            for match in candidates:
                stored_teams[match.home_team_id] = match.home_team
                stored_teams[match.away_team_id] = match.away_team
        decided = _decide(teams, _votes(matches, fixtures), stored_teams, report)
        if dry_run:
            return report

        existing_competition = session.scalars(
            select(CompetitionExternalId).where(
                CompetitionExternalId.data_source_id == source_id,
                CompetitionExternalId.external_id == competition_external_id,
            )
        ).one_or_none()
        if existing_competition is None:
            session.add(
                CompetitionExternalId(
                    data_source_id=source_id,
                    external_id=competition_external_id,
                    competition_id=stored_competition_id,
                )
            )
            report.mappings_written += 1
        elif existing_competition.competition_id != stored_competition_id:
            raise ValueError(
                f"competition {competition_external_id} is already linked to stored "
                f"competition {existing_competition.competition_id}"
            )
        mapped = set(
            session.scalars(
                select(TeamExternalId.external_id).where(TeamExternalId.data_source_id == source_id)
            )
        )
        for external_id, team_id in decided.items():
            if external_id not in mapped:
                session.add(
                    TeamExternalId(
                        data_source_id=source_id, external_id=external_id, team_id=team_id
                    )
                )
                report.mappings_written += 1
    return report
