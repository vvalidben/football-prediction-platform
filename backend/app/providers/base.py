"""The provider-independent interface every football data source implements."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.providers.errors import UnsupportedOperationError
from app.providers.schemas import (
    ExternalId,
    ProviderCompetition,
    ProviderInjury,
    ProviderLineup,
    ProviderMatch,
    ProviderMatchDetails,
    ProviderOdds,
    ProviderPlayerMatchStatistics,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
)


class ProviderOperation(StrEnum):
    COMPETITIONS = "competitions"
    SEASONS = "seasons"
    TEAMS = "teams"
    MATCHES = "matches"
    MATCH_DETAILS = "match_details"
    TEAM_STATISTICS = "team_statistics"
    PLAYER_STATISTICS = "player_statistics"
    INJURIES = "injuries"
    LINEUPS = "lineups"
    ODDS = "odds"


@dataclass(frozen=True)
class RejectedRecord:
    """A provider record that could not be normalised, kept for the ingestion log."""

    reason: str
    external_id: str | None = None


@dataclass(frozen=True)
class FetchResult[T]:
    """Records normalised from one provider call, plus any that were malformed.

    One bad record never fails a whole call: it is reported in `rejected` so the
    ingestion run can count it and carry on with the rest.
    """

    records: list[T] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)


class FootballDataProvider(ABC):
    """A source of football data, independent of any provider's own API format.

    Implementations must:
    - return only the provider-neutral types from `app.providers.schemas`;
    - raise only `app.providers.errors.ProviderError` subclasses;
    - put malformed individual records in `FetchResult.rejected` instead of raising.

    Operations a provider cannot serve raise `UnsupportedOperationError`; check
    `capabilities` first to avoid that.

    Season identifiers are opaque provider strings. Callers pass back exactly what
    `get_seasons` returned and never parse them.
    """

    #: Stable identifier, stored as `data_sources.code` (lowercase, underscores).
    code: str
    #: Human-readable provider name.
    name: str
    #: Base URL recorded on the data source, when there is one.
    base_url: str | None = None

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[ProviderOperation]:
        """Operations this provider implements."""

    def supports(self, operation: ProviderOperation) -> bool:
        return operation in self.capabilities

    @abstractmethod
    def get_competitions(self) -> FetchResult[ProviderCompetition]: ...

    @abstractmethod
    def get_seasons(self, competition_id: ExternalId) -> FetchResult[ProviderSeason]: ...

    @abstractmethod
    def get_teams(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderTeam]: ...

    @abstractmethod
    def get_matches(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderMatch]: ...

    # Operations below are part of the contract but optional for a provider.

    def get_match_details(self, match_id: ExternalId) -> ProviderMatchDetails:
        raise self._unsupported(ProviderOperation.MATCH_DETAILS)

    def get_team_statistics(
        self, match_ids: Sequence[ExternalId]
    ) -> FetchResult[ProviderTeamMatchStatistics]:
        """Statistics for each team in the given matches. Matches the provider has no
        statistics for are simply absent from the result (callers detect the gap)."""
        raise self._unsupported(ProviderOperation.TEAM_STATISTICS)

    def get_player_statistics(
        self, match_id: ExternalId
    ) -> FetchResult[ProviderPlayerMatchStatistics]:
        raise self._unsupported(ProviderOperation.PLAYER_STATISTICS)

    def get_injuries(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderInjury]:
        raise self._unsupported(ProviderOperation.INJURIES)

    def get_lineups(self, match_id: ExternalId) -> FetchResult[ProviderLineup]:
        raise self._unsupported(ProviderOperation.LINEUPS)

    def get_odds(self, match_id: ExternalId) -> FetchResult[ProviderOdds]:
        raise self._unsupported(ProviderOperation.ODDS)

    def close(self) -> None:  # noqa: B027 - optional hook; the default has nothing to release
        """Release network resources. Safe to call more than once."""

    def _unsupported(self, operation: ProviderOperation) -> UnsupportedOperationError:
        return UnsupportedOperationError(
            f"{self.name} does not support {operation.value}", provider=self.code
        )
