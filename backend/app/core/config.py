"""Application settings, loaded from environment variables (and `.env` for local runs)."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy import URL

from app import __version__

# The repository root `.env` is shared by Docker Compose and local runs. Inside the
# container this path does not exist and values come from the environment only.
_REPO_ROOT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

Environment = Literal["development", "test", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "football-prediction-api"
    app_version: str = __version__
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    postgres_host: str = "127.0.0.1"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)
    #: PostgreSQL cancels a query that runs longer than this, so one slow statement
    #: cannot hold a connection until the pool is exhausted. Applied to the API's
    #: engine only: the jobs run long queries on purpose. 0 disables it.
    database_statement_timeout_seconds: float = Field(default=30.0, ge=0)

    redis_host: str = "127.0.0.1"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_db: int = Field(default=0, ge=0)
    redis_password: SecretStr | None = None

    # Rate limiting. Counters live in Redis; if Redis is down the limiter is
    # skipped rather than failing requests (see app/api/rate_limit.py).
    rate_limit_enabled: bool = True
    #: Requests per window from one address, for everything except predictions.
    rate_limit_requests: int = Field(default=300, ge=1)
    #: Asking for a prediction builds features and runs a model, so it is stricter.
    rate_limit_predict_requests: int = Field(default=30, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)

    # Caching of the slow, slowly-changing status endpoints. 0 disables it.
    cache_enabled: bool = True
    status_cache_seconds: float = Field(default=30.0, ge=0)

    # Socket connect/read timeout for PostgreSQL and Redis connections.
    dependency_timeout_seconds: float = Field(default=2.0, gt=0)
    # Overall budget for /health; checks still running after it are reported as timed out.
    health_check_timeout_seconds: float = Field(default=3.0, gt=0)

    # Football data providers. Keys are optional: without one, that provider is unusable.
    api_football_api_key: SecretStr | None = None
    api_football_base_url: str = "https://v3.football.api-sports.io"
    # Client-side pacing; keep at or below the plan's per-minute limit.
    api_football_requests_per_minute: float = Field(default=10.0, gt=0)
    football_data_api_key: SecretStr | None = None
    football_data_base_url: str = "https://api.football-data.org/v4"
    football_data_requests_per_minute: float = Field(default=10.0, gt=0)
    # Oldest season the plan can read. The API lists seasons back to 1888 and answers
    # 403 for those outside the plan, so older ones are not offered at all.
    football_data_first_season: int = Field(default=2023, ge=1800)
    provider_timeout_seconds: float = Field(default=10.0, gt=0)
    provider_max_attempts: int = Field(default=4, ge=1, le=10)

    # Background jobs (Celery over Redis). Every schedule is in minutes, so a
    # deployment can slow a job down or turn it off without changing code.
    jobs_enabled: bool = True
    celery_broker_url: str | None = None  # defaults to the Redis connection below
    celery_result_backend: str | None = None
    #: Provider competition ids the jobs keep up to date, e.g. "39,140".
    job_competitions: Annotated[list[str], NoDecode] = Field(default_factory=list)
    job_provider: str = "api_football"
    schedule_fixtures_minutes: float = Field(default=360.0, gt=0)
    schedule_results_minutes: float = Field(default=60.0, gt=0)
    schedule_statistics_minutes: float = Field(default=180.0, gt=0)
    schedule_features_minutes: float = Field(default=120.0, gt=0)
    schedule_predictions_minutes: float = Field(default=180.0, gt=0)
    schedule_evaluation_minutes: float = Field(default=120.0, gt=0)
    schedule_retrain_minutes: float = Field(default=1440.0, gt=0)
    schedule_failure_report_minutes: float = Field(default=1440.0, gt=0)
    #: How far ahead predictions are refreshed, and how far back results are checked.
    prediction_horizon_days: int = Field(default=7, ge=1)
    statistics_lookback_days: int = Field(default=7, ge=1)
    #: Retraining: only when this much time has passed and this many new results exist.
    retrain_every_days: float = Field(default=30.0, gt=0)
    retrain_min_new_matches: int = Field(default=100, ge=0)
    retrain_enabled: bool = True
    #: Retries: exponential backoff from `task_retry_backoff_seconds`, capped.
    task_max_retries: int = Field(default=5, ge=0, le=20)
    task_retry_backoff_seconds: float = Field(default=10.0, gt=0)
    task_retry_backoff_max_seconds: float = Field(default=900.0, gt=0)
    task_time_limit_seconds: int = Field(default=1800, gt=0)
    #: Window the failure report covers.
    failure_report_hours: float = Field(default=24.0, gt=0)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @field_validator("job_competitions", mode="before")
    @classmethod
    def _split_competitions(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string such as `http://a,http://b`."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator(
        "redis_password", "api_football_api_key", "football_data_api_key", mode="before"
    )
    @classmethod
    def _empty_secret_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def database_url(self) -> URL:
        # URL.create escapes credentials, so passwords may contain any character.
        return URL.create(
            drivername="postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # required fields come from the environment
