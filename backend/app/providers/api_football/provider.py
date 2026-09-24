"""API-Football (api-sports.io) v3 adapter."""

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.providers.api_football import normalize, payloads
from app.providers.base import FetchResult, FootballDataProvider, ProviderOperation, RejectedRecord
from app.providers.errors import (
    AuthenticationError,
    MalformedResponseError,
    ProviderRequestError,
    QuotaExceededError,
    RateLimitError,
)
from app.providers.http import JsonHttpClient, RateLimiter, RetryPolicy, Sleep
from app.providers.schemas import (
    ExternalId,
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
)

logger = logging.getLogger(__name__)

CODE = "api_football"
DEFAULT_BASE_URL = "https://v3.football.api-sports.io"
#: API-Football counts a per-minute window; errors under this key are worth retrying.
RATE_LIMIT_RETRY_AFTER_SECONDS = 60.0
#: `GET /fixtures?ids=` accepts at most this many fixture ids.
MAX_FIXTURE_IDS_PER_REQUEST = 20


def check_envelope(body: Any) -> None:
    """Raise for errors API-Football reports inside an HTTP 200 response.

    The API returns `"errors": []` on success and an object such as
    `{"token": "..."}`, `{"requests": "..."}` or `{"rateLimit": "..."}` on failure.
    """
    if not isinstance(body, dict) or "response" not in body:
        raise MalformedResponseError("response is not an API-Football envelope", provider=CODE)
    errors = body.get("errors") or {}
    if not errors:
        return
    if isinstance(errors, list):
        errors = {str(index): message for index, message in enumerate(errors)}
    if not isinstance(errors, dict):
        raise MalformedResponseError("unexpected 'errors' value", provider=CODE)

    summary = "; ".join(f"{key}: {message}" for key, message in errors.items())
    keys = {str(key).lower() for key in errors}
    if "token" in keys:
        raise AuthenticationError(summary, provider=CODE)
    if "ratelimit" in keys:
        raise RateLimitError(summary, provider=CODE, retry_after=RATE_LIMIT_RETRY_AFTER_SECONDS)
    if "requests" in keys:
        raise QuotaExceededError(summary, provider=CODE)
    raise ProviderRequestError(summary, provider=CODE)


class ApiFootballProvider(FootballDataProvider):
    code = CODE
    name = "API-Football"

    def __init__(self, http: JsonHttpClient, base_url: str = DEFAULT_BASE_URL) -> None:
        self._http = http
        self.base_url = base_url

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: Any = None,
        sleep: Sleep | None = None,
    ) -> "ApiFootballProvider":
        """Build from settings. `transport` and `sleep` (used between retries) are
        injection points for tests."""
        if settings.api_football_api_key is None:
            raise AuthenticationError("API_FOOTBALL_API_KEY is not set", provider=CODE)
        extra: dict[str, Any] = {} if sleep is None else {"sleep": sleep}
        http = JsonHttpClient(
            provider=CODE,
            base_url=settings.api_football_base_url,
            headers={"x-apisports-key": settings.api_football_api_key.get_secret_value()},
            timeout_seconds=settings.provider_timeout_seconds,
            retry_policy=RetryPolicy(max_attempts=settings.provider_max_attempts),
            rate_limiter=RateLimiter(settings.api_football_requests_per_minute),
            transport=transport,
            **extra,
        )
        return cls(http, base_url=settings.api_football_base_url)

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

    def get_competitions(self) -> FetchResult[ProviderCompetition]:
        return self._fetch("/leagues", {}, payloads.LeagueItem, normalize.competition, _league_id)

    def get_seasons(self, competition_id: ExternalId) -> FetchResult[ProviderSeason]:
        leagues = self._fetch("/leagues", {"id": competition_id}, payloads.LeagueItem, _identity)
        result: FetchResult[ProviderSeason] = FetchResult(rejected=leagues.rejected)
        for item in leagues.records:
            for raw in item.seasons:
                try:
                    result.records.append(normalize.season(item.league.id, raw))
                except ValueError as exc:
                    result.rejected.append(
                        RejectedRecord(_reason(exc), f"{item.league.id}:{raw.year}")
                    )
        return result

    def get_teams(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderTeam]:
        params = self._league_season_params(competition_id, season_id)
        return self._fetch("/teams", params, payloads.TeamItem, normalize.team, _team_id)

    def get_matches(
        self, competition_id: ExternalId, season_id: ExternalId
    ) -> FetchResult[ProviderMatch]:
        params = self._league_season_params(competition_id, season_id)
        return self._fetch("/fixtures", params, payloads.FixtureItem, normalize.match, _fixture_id)

    def get_team_statistics(
        self, match_ids: Sequence[ExternalId]
    ) -> FetchResult[ProviderTeamMatchStatistics]:
        """Fetched through `/fixtures?ids=`, which returns up to 20 fixtures with their
        statistics in one request."""
        result: FetchResult[ProviderTeamMatchStatistics] = FetchResult()
        unique_ids = list(dict.fromkeys(match_ids))
        for start in range(0, len(unique_ids), MAX_FIXTURE_IDS_PER_REQUEST):
            batch = unique_ids[start : start + MAX_FIXTURE_IDS_PER_REQUEST]
            details = self._fetch(
                "/fixtures",
                {"ids": "-".join(batch)},
                payloads.FixtureDetailItem,
                normalize.team_statistics,
                _fixture_id,
            )
            result.rejected.extend(details.rejected)
            for records in details.records:
                result.records.extend(records)
        return result

    def close(self) -> None:
        self._http.close()

    # -- helpers ---------------------------------------------------------------

    def _league_season_params(self, competition_id: str, season_id: str) -> dict[str, str | int]:
        try:
            league, year = normalize.parse_season_external_id(season_id)
        except ValueError as exc:
            raise ProviderRequestError(str(exc), provider=CODE) from exc
        if str(league) != competition_id:
            raise ProviderRequestError(
                f"season {season_id!r} does not belong to competition {competition_id!r}",
                provider=CODE,
            )
        return {"league": league, "season": year}

    def _pages(self, path: str, params: Mapping[str, str | int]) -> list[Any]:
        """Return the `response` items of every page of `GET path`."""
        items: list[Any] = []
        page = 1
        while True:
            query = dict(params) if page == 1 else {**params, "page": page}
            body = self._http.get_json(path, query, check=check_envelope)
            try:
                envelope = payloads.Envelope.model_validate(body)
            except ValidationError as exc:
                raise MalformedResponseError(
                    f"{path}: invalid envelope ({exc.error_count()} errors)", provider=CODE
                ) from exc
            items.extend(envelope.response)
            if envelope.paging.current >= envelope.paging.total:
                return items
            page += 1

    def _fetch[P: BaseModel, R](
        self,
        path: str,
        params: Mapping[str, str | int],
        payload_type: type[P],
        convert: Callable[[P], R],
        external_id: Callable[[Any], str | None] = lambda _: None,
    ) -> FetchResult[R]:
        result: FetchResult[R] = FetchResult()
        for raw in self._pages(path, params):
            try:
                result.records.append(convert(payload_type.model_validate(raw)))
            except ValueError as exc:  # includes pydantic.ValidationError
                reference = external_id(raw)
                logger.warning("%s: rejected record %s: %s", path, reference, _reason(exc))
                result.rejected.append(RejectedRecord(_reason(exc), reference))
        return result


def _identity[T](value: T) -> T:
    return value


def _reason(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "record"
        return f"{location}: {first['msg']}"
    return str(exc)


def _dig(raw: Any, *keys: str) -> str | None:
    for key in keys:
        if not isinstance(raw, dict):
            return None
        raw = raw.get(key)
    return None if raw is None else str(raw)


def _league_id(raw: Any) -> str | None:
    return _dig(raw, "league", "id")


def _team_id(raw: Any) -> str | None:
    return _dig(raw, "team", "id")


def _fixture_id(raw: Any) -> str | None:
    return _dig(raw, "fixture", "id")
