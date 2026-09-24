import pytest

from app.providers.base import ProviderOperation, RejectedRecord
from app.providers.errors import ProviderUnavailableError, QuotaExceededError
from app.providers.mock import MockDataset, MockFootballDataProvider
from tests.providers import builders


def test_serves_only_the_requested_season() -> None:
    provider = MockFootballDataProvider(builders.dataset())

    assert provider.get_matches(builders.COMPETITION, builders.PREVIOUS_SEASON).records == []
    assert len(provider.get_matches(builders.COMPETITION, builders.SEASON).records) == 2


def test_empty_by_default_and_ships_no_data() -> None:
    provider = MockFootballDataProvider()

    assert provider.get_competitions().records == []


def test_queued_failures_are_raised_in_order_then_cleared() -> None:
    provider = MockFootballDataProvider(builders.dataset())
    provider.fail_next(
        ProviderOperation.COMPETITIONS,
        ProviderUnavailableError("down"),
        QuotaExceededError("quota"),
    )

    with pytest.raises(ProviderUnavailableError):
        provider.get_competitions()
    with pytest.raises(QuotaExceededError):
        provider.get_competitions()
    assert len(provider.get_competitions().records) == 1


def test_failures_only_affect_their_operation() -> None:
    provider = MockFootballDataProvider(builders.dataset())
    provider.fail_next(ProviderOperation.MATCHES, ProviderUnavailableError("down"))

    assert provider.get_seasons(builders.COMPETITION).records


def test_reports_configured_rejections() -> None:
    rejected = RejectedRecord("bad status", "x1")
    provider = MockFootballDataProvider(
        MockDataset(rejected={ProviderOperation.COMPETITIONS: [rejected]})
    )

    assert provider.get_competitions().rejected == [rejected]


def test_records_calls() -> None:
    provider = MockFootballDataProvider(builders.dataset())

    provider.get_teams(builders.COMPETITION, builders.SEASON)

    assert provider.calls == [(ProviderOperation.TEAMS, (builders.COMPETITION, builders.SEASON))]


class TestProviderRegistry:
    """The registry the CLI and the jobs go through to build a provider."""

    def test_a_configured_provider_is_built_by_code(self) -> None:
        from pydantic import SecretStr

        from app.providers.api_football import ApiFootballProvider
        from app.providers.registry import create_provider
        from tests.tasks.test_celery_and_recording import make_settings

        provider = create_provider(
            "api_football", make_settings(api_football_api_key=SecretStr("k" * 32))
        )

        assert isinstance(provider, ApiFootballProvider)
        assert provider.code == "api_football"

    def test_an_unknown_code_names_the_ones_that_exist(self) -> None:
        from app.providers.registry import create_provider
        from tests.tasks.test_celery_and_recording import make_settings

        with pytest.raises(ValueError, match="unknown provider 'football_oracle'") as error:
            create_provider("football_oracle", make_settings())

        assert "api_football" in str(error.value)

    def test_the_mock_provider_cannot_be_built_from_the_registry(self) -> None:
        """It serves test data; reaching it from the CLI would write invented
        matches into a real database."""
        from app.providers.registry import PROVIDERS, create_provider
        from tests.tasks.test_celery_and_recording import make_settings

        assert "mock" not in PROVIDERS
        with pytest.raises(ValueError, match="unknown provider"):
            create_provider("mock", make_settings())

    def test_a_provider_without_its_credentials_fails_when_it_is_built(self) -> None:
        from app.providers.errors import AuthenticationError
        from app.providers.registry import create_provider
        from tests.tasks.test_celery_and_recording import make_settings

        with pytest.raises(AuthenticationError, match="API_FOOTBALL_API_KEY"):
            create_provider("api_football", make_settings(api_football_api_key=None))
