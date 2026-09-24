"""An in-memory provider for tests and local development.

It serves whatever records it is given and can be told to fail, so ingestion
behaviour (idempotency, error handling, partial failures) can be exercised without
a network or an API key. It ships no data of its own.
"""

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.providers.base import FetchResult, FootballDataProvider, ProviderOperation, RejectedRecord
from app.providers.errors import ProviderError, ProviderRequestError
from app.providers.schemas import (
    ExternalId,
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
)


@dataclass
class MockDataset:
    competitions: list[ProviderCompetition] = field(default_factory=list)
    seasons: list[ProviderSeason] = field(default_factory=list)
    #: Teams per season external ID.
    teams: dict[ExternalId, list[ProviderTeam]] = field(default_factory=dict)
    matches: list[ProviderMatch] = field(default_factory=list)
    #: Team statistics per match external ID.
    team_statistics: dict[ExternalId, list[ProviderTeamMatchStatistics]] = field(
        default_factory=dict
    )
    #: Malformed records to report per operation, as a real adapter would.
    rejected: dict[ProviderOperation, list[RejectedRecord]] = field(default_factory=dict)


class MockFootballDataProvider(FootballDataProvider):
    code = "mock"
    name = "Mock provider"

    def __init__(self, dataset: MockDataset | None = None, *, code: str = "mock") -> None:
        self.dataset = dataset or MockDataset()
        self.code = code
        self.calls: list[tuple[ProviderOperation, tuple[str, ...]]] = []
        self._failures: dict[ProviderOperation, deque[ProviderError]] = defaultdict(deque)

    @property
    def capabilities(self) -> frozenset[ProviderOperation]:
        return frozenset(
            {
                ProviderOperation.COMPETITIONS,
                ProviderOperation.SEASONS,
                ProviderOperation.TEAMS,
                ProviderOperation.MATCHES,
                ProviderOperation.TEAM_STATISTICS,
            }
        )

    def fail_next(self, operation: ProviderOperation, *errors: ProviderError) -> None:
        """Make the next calls to `operation` raise `errors`, one per call, in order."""
        self._failures[operation].extend(errors)

    def get_competitions(self) -> FetchResult[ProviderCompetition]:
        self._enter(ProviderOperation.COMPETITIONS)
        return self._result(ProviderOperation.COMPETITIONS, self.dataset.competitions)

    def get_seasons(self, competition_id: ExternalId) -> FetchResult[ProviderSeason]:
        self._enter(ProviderOperation.SEASONS, competition_id)
        seasons = [s for s in self.dataset.seasons if s.competition_external_id == competition_id]
        return self._result(ProviderOperation.SEASONS, seasons)

    def get_teams(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderTeam]:
        self._enter(ProviderOperation.TEAMS, competition_id, season_id)
        self._require_season(competition_id, season_id)
        return self._result(ProviderOperation.TEAMS, self.dataset.teams.get(season_id, []))

    def get_matches(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderMatch]:
        self._enter(ProviderOperation.MATCHES, competition_id, season_id)
        self._require_season(competition_id, season_id)
        matches = [
            m
            for m in self.dataset.matches
            if m.competition_external_id == competition_id and m.season_external_id == season_id
        ]
        return self._result(ProviderOperation.MATCHES, matches)

    def get_team_statistics(
        self, match_ids: Sequence[ExternalId]
    ) -> FetchResult[ProviderTeamMatchStatistics]:
        self._enter(ProviderOperation.TEAM_STATISTICS, *match_ids)
        records = [
            record
            for match_id in dict.fromkeys(match_ids)
            for record in self.dataset.team_statistics.get(match_id, [])
        ]
        return self._result(ProviderOperation.TEAM_STATISTICS, records)

    def _enter(self, operation: ProviderOperation, *args: str) -> None:
        self.calls.append((operation, args))
        if self._failures[operation]:
            raise self._failures[operation].popleft()

    def _require_season(self, competition_id: str, season_id: str) -> None:
        known = {(s.competition_external_id, s.external_id) for s in self.dataset.seasons}
        if (competition_id, season_id) not in known:
            raise ProviderRequestError(
                f"unknown season {season_id!r} for competition {competition_id!r}",
                provider=self.code,
            )

    def _result[T](self, operation: ProviderOperation, records: Iterable[T]) -> FetchResult[T]:
        return FetchResult(list(records), list(self.dataset.rejected.get(operation, [])))
