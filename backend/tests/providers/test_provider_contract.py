"""Every provider honours the same `FootballDataProvider` contract.

The same checks run against the mock and the API-Football adapter (over a fake
HTTP transport), so ingestion can rely on the contract rather than on either one.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

from app.providers.base import FetchResult, FootballDataProvider, ProviderOperation
from app.providers.errors import ProviderError, UnsupportedOperationError
from app.providers.mock import MockFootballDataProvider
from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderRecord,
    ProviderSeason,
    ProviderTeam,
)
from tests.providers import builders
from tests.providers.fake_api import FakeApiFootball
from tests.providers.fake_football_data import PREMIER_LEAGUE, SEASON_2025, FakeFootballData


@dataclass(frozen=True)
class Subject:
    provider: FootballDataProvider
    competition: str
    season: str


def _mock() -> Subject:
    return Subject(
        MockFootballDataProvider(builders.dataset()), builders.COMPETITION, builders.SEASON
    )


def _api_football() -> Subject:
    return Subject(FakeApiFootball().provider(), "39", "39:2023")


def _football_data() -> Subject:
    return Subject(FakeFootballData().provider(), PREMIER_LEAGUE, SEASON_2025)


@pytest.fixture(
    params=[_mock, _api_football, _football_data],
    ids=["mock", "api_football", "football_data"],
)
def subject(request: pytest.FixtureRequest) -> Iterator[Subject]:
    factory: Callable[[], Subject] = request.param
    made = factory()
    yield made
    made.provider.close()


def _assert_result[T](result: object, record_type: type[T]) -> FetchResult[T]:
    assert isinstance(result, FetchResult)
    assert all(isinstance(record, record_type) for record in result.records)
    return result


def test_identity_is_a_valid_data_source_code(subject: Subject) -> None:
    assert subject.provider.code.replace("_", "").isalnum()
    assert subject.provider.code == subject.provider.code.lower()
    assert subject.provider.name


def test_core_operations_are_supported(subject: Subject) -> None:
    for operation in (
        ProviderOperation.COMPETITIONS,
        ProviderOperation.SEASONS,
        ProviderOperation.TEAMS,
        ProviderOperation.MATCHES,
    ):
        assert subject.provider.supports(operation)


def test_core_operations_return_neutral_records(subject: Subject) -> None:
    p = subject.provider
    competitions = _assert_result(p.get_competitions(), ProviderCompetition)
    seasons = _assert_result(p.get_seasons(subject.competition), ProviderSeason)
    teams = _assert_result(p.get_teams(subject.competition, subject.season), ProviderTeam)
    matches = _assert_result(p.get_matches(subject.competition, subject.season), ProviderMatch)

    assert subject.competition in {c.external_id for c in competitions.records}
    assert subject.season in {s.external_id for s in seasons.records}
    team_ids = {t.external_id for t in teams.records}
    for record in matches.records:
        # Referential integrity within one provider's data.
        assert record.season_external_id == subject.season
        assert record.competition_external_id == subject.competition
        assert {record.home_team_external_id, record.away_team_external_id} <= team_ids


def test_records_are_immutable(subject: Subject) -> None:
    record: ProviderRecord = subject.provider.get_competitions().records[0]

    with pytest.raises(ValueError, match="frozen"):
        record.external_id = "changed"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        (ProviderOperation.MATCH_DETAILS, lambda p: p.get_match_details("1")),
        (ProviderOperation.LINEUPS, lambda p: p.get_lineups("1")),
        (ProviderOperation.ODDS, lambda p: p.get_odds("1")),
        (ProviderOperation.INJURIES, lambda p: p.get_injuries("1", "1")),
        (ProviderOperation.PLAYER_STATISTICS, lambda p: p.get_player_statistics("1")),
    ],
)
def test_unsupported_operations_are_declared_and_raise(
    subject: Subject,
    operation: ProviderOperation,
    call: Callable[[FootballDataProvider], object],
) -> None:
    assert not subject.provider.supports(operation)
    with pytest.raises(UnsupportedOperationError) as exc_info:
        call(subject.provider)
    assert isinstance(exc_info.value, ProviderError)
    assert exc_info.value.provider == subject.provider.code


def test_unknown_season_raises_a_provider_error(subject: Subject) -> None:
    with pytest.raises(ProviderError):
        subject.provider.get_matches(subject.competition, "no-such-season")


def test_interface_cannot_be_instantiated_without_core_operations() -> None:
    class Incomplete(FootballDataProvider):
        code = "incomplete"
        name = "Incomplete"

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]
