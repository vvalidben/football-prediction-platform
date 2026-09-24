import pytest
from pydantic import ValidationError

from app.core.config import Settings

DB_PASSWORD = "p@ss:word/with?chars"  # noqa: S105 - dummy value for tests


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "postgres_user": "user",
        "postgres_password": DB_PASSWORD,
        "postgres_db": "football",
        **overrides,
    }
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_database_url_escapes_credentials() -> None:
    url = make_settings(postgres_host="db", postgres_port=5433).database_url

    assert url.drivername == "postgresql+psycopg"
    assert url.password == DB_PASSWORD
    assert url.host == "db"
    assert url.port == 5433
    assert DB_PASSWORD not in str(url)  # rendered URLs mask the password


def test_password_is_masked_in_repr() -> None:
    assert DB_PASSWORD not in repr(make_settings())


def test_cors_origins_parsed_from_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173, http://localhost:8080,")

    settings = make_settings()

    assert settings.cors_allowed_origins == ["http://localhost:5173", "http://localhost:8080"]


def test_empty_redis_password_is_treated_as_unset() -> None:
    assert make_settings(redis_password="").redis_password is None


def test_missing_database_credentials_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    missing = {error["loc"][0] for error in exc_info.value.errors()}
    assert missing == {"postgres_user", "postgres_password", "postgres_db"}


def test_invalid_environment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(environment="prod")
