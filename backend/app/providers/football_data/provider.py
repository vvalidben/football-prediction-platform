"""football-data.org v4 adapter.

What it adds to this platform is the current season: fixtures that have not been
played yet, which API-Football's free plan does not serve. It has no per-match
statistics, odds, lineups or injuries on the free plan, and declares none of them.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.providers.base import FetchResult, FootballDataProvider, ProviderOperation, RejectedRecord
from app.providers.errors import AuthenticationError, MalformedResponseError, ProviderRequestError
from app.providers.football_data import normalize, payloads
from app.providers.http import JsonHttpClient, RateLimiter, RetryPolicy, Sleep
from app.providers.schemas import (
    ExternalId,
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
)

logger = logging.getLogger(__name__)

CODE = "football_data"
DEFAULT_BASE_URL = "https://api.football-data.org/v4"


class FootballDataOrgProvider(FootballDataProvider):
    code = CODE
    name = "football-data.org"

    def __init__(
        self, http: JsonHttpClient, base_url: str = DEFAULT_BASE_URL, first_season: int = 2023
    ) -> None:
        self._http = http
        self.base_url = base_url
        self._first_season = first_season

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: Any = None,
        sleep: Sleep | None = None,
    ) -> "FootballDataOrgProvider":
        """Build from settings. `transport` and `sleep` (used between retries) are
        injection points for tests."""
        if settings.football_data_api_key is None:
            raise AuthenticationError("FOOTBALL_DATA_API_KEY is not set", provider=CODE)
        extra: dict[str, Any] = {} if sleep is None else {"sleep": sleep}
        # A relative path keeps the /v4 prefix only if the base URL ends in a slash.
        base_url = settings.football_data_base_url.rstrip("/") + "/"
        http = JsonHttpClient(
            provider=CODE,
            base_url=base_url,
            headers={"X-Auth-Token": settings.football_data_api_key.get_secret_value()},
            timeout_seconds=settings.provider_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=settings.provider_max_attempts),
            rate_limiter=RateLimiter(settings.football_data_requests_per_minute),
            transport=transport,
            **extra,
        )
        return cls(
            http,
            base_url=settings.football_data_base_url,
            first_season=settings.football_data_first_season,
        )

    @property
    def capabilities(self) -> frozenset[ProviderOperation]:
        return frozenset(
            {
                ProviderOperation.COMPETITIONS,
                ProviderOperation.SEASONS,
                ProviderOperation.TEAMS,
                ProviderOperation.MATCHES,
            }
        )

    def get_competitions(self) -> FetchResult[ProviderCompetition]:
        body = self._get("competitions")
        return _convert(
            _items(body, "competitions"),
            payloads.Competition,
            normalize.competition,
        )

    def get_seasons(self, competition_id: ExternalId) -> FetchResult[ProviderSeason]:
        body = self._get(f"competitions/{competition_id}")
        try:
            detail = payloads.CompetitionDetail.model_validate(body)
        except ValidationError as exc:
            raise MalformedResponseError(
                f"competition {competition_id}: invalid response ({exc.error_count()} errors)",
                provider=CODE,
            ) from exc
        current_id = detail.currentSeason.id if detail.currentSeason else None
        result: FetchResult[ProviderSeason] = FetchResult()
        for raw in detail.seasons:
            if raw.startDate.year < self._first_season:
                continue
            try:
                result.records.append(normalize.season(detail.id, raw, current_id=current_id))
            except ValueError as exc:
                result.rejected.append(RejectedRecord(_reason(exc), str(raw.id)))
        return result

    def get_teams(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderTeam]:
        year = self._season_year(competition_id, season_id)
        body = self._get(f"competitions/{competition_id}/teams", {"season": year})
        return _convert(_items(body, "teams"), payloads.Team, normalize.team)

    def get_matches(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderMatch]:
        year = self._season_year(competition_id, season_id)
        body = self._get(f"competitions/{competition_id}/matches", {"season": year})
        return _convert(
            [raw for raw in _items(body, "matches") if not _undecided(raw)],
            payloads.Match,
            lambda raw: normalize.match(raw, competition_id, season_id),
        )

    def close(self) -> None:
        self._http.close()

    # -- helpers ---------------------------------------------------------------

    def _get(self, path: str, params: Mapping[str, str | int] | None = None) -> Any:
        return self._http.get_json(path, params)

    def _season_year(self, competition_id: str, season_id: str) -> int:
        try:
            competition, year = normalize.parse_season_external_id(season_id)
        except ValueError as exc:
            raise ProviderRequestError(str(exc), provider=CODE) from exc
        if competition != competition_id:
            raise ProviderRequestError(
                f"season {season_id!r} does not belong to competition {competition_id!r}",
                provider=CODE,
            )
        return year


def _items(body: Any, key: str) -> list[Any]:
    items = body.get(key) if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise MalformedResponseError(f"response has no {key!r} list", provider=CODE)
    return items


def _undecided(raw: Any) -> bool:
    """A knockout tie listed before its teams are known, such as a cup final."""
    if not isinstance(raw, dict):
        return False
    return any(
        isinstance(raw.get(side), dict) and raw[side].get("id") is None
        for side in ("homeTeam", "awayTeam")
    )


def _convert[P: BaseModel, R](
    items: list[Any], payload_type: type[P], convert: Callable[[P], R]
) -> FetchResult[R]:
    result: FetchResult[R] = FetchResult()
    for raw in items:
        try:
            result.records.append(convert(payload_type.model_validate(raw)))
        except ValueError as exc:  # includes pydantic.ValidationError
            reference = str(raw["id"]) if isinstance(raw, dict) and "id" in raw else None
            logger.warning("rejected record %s: %s", reference, _reason(exc))
            result.rejected.append(RejectedRecord(_reason(exc), reference))
    return result


def _reason(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "record"
        return f"{location}: {first['msg']}"
    return str(exc)
