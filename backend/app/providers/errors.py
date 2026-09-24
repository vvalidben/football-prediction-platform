"""Provider-neutral errors. Adapters translate every provider-specific failure into one
of these, so ingestion code never inspects HTTP details or provider error formats.
"""


class ProviderError(Exception):
    """Base class. `retryable` says whether the same request may succeed if repeated."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class ProviderUnavailableError(ProviderError):
    """Network failure, timeout or server-side (5xx) error."""

    retryable = True


class RateLimitError(ProviderError):
    """Too many requests in a short window; wait `retry_after` seconds, then retry."""

    retryable = True

    def __init__(
        self, message: str, *, provider: str | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after = retry_after


class QuotaExceededError(ProviderError):
    """The plan's daily (or monthly) request quota is used up. Retrying will not help."""


class AuthenticationError(ProviderError):
    """Missing, invalid or unauthorised API credentials."""


class ProviderRequestError(ProviderError):
    """The provider rejected the request itself (bad parameters, unknown resource)."""


class MalformedResponseError(ProviderError):
    """The response could not be parsed or does not have the documented structure."""


class UnsupportedOperationError(ProviderError):
    """The provider (or this adapter) does not offer the requested data."""
