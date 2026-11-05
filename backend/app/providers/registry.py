"""Construct configured providers by code. The mock provider is deliberately absent:
it serves test data only and must never write to a real database from the CLI."""

from collections.abc import Callable

from app.core.config import Settings
from app.providers.api_football import ApiFootballProvider
from app.providers.base import FootballDataProvider
from app.providers.football_data import FootballDataOrgProvider

PROVIDERS: dict[str, Callable[[Settings], FootballDataProvider]] = {
    ApiFootballProvider.code: ApiFootballProvider.from_settings,
    FootballDataOrgProvider.code: FootballDataOrgProvider.from_settings,
}


def create_provider(code: str, settings: Settings) -> FootballDataProvider:
    try:
        factory = PROVIDERS[code]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {code!r} (known: {known})") from None
    return factory(settings)
