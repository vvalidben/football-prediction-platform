"""A standing guard: no secrets in the tree that gets committed, none in the bundle.

"We checked once" is not a control. This walks the repository the way a commit
would see it — skipping what `.gitignore` excludes — and fails on the patterns that
have actually leaked in projects like this one: a filled-in provider key, a
connection string with a password in it, a private key, or a frontend build that
carries either.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Directories that never reach a commit (they are in .gitignore) or are binary.
SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "htmlcov",
    "mlruns",
    "graphify-out",
    "reports",
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".joblib", ".db", ".pyc"}
#: `.env` holds this machine's real credentials and is git-ignored; the test below
#: asserts that, and its content is deliberately not scanned here.
SKIP_FILES = {".env"}

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "connection string with a password",
        re.compile(r"\b(postgres|redis|amqp)://[^\s:/@]+:[^\s:/@]+@"),
    ),
    # A provider key is 32 hex characters; an empty or placeholder value is fine.
    (
        "API-Football key",
        re.compile(r"(?i)api[_-]?football[_-]?api[_-]?key\s*[=:]\s*[\"']?[0-9a-f]{20,}"),
    ),
    (
        "hard-coded apisports header",
        re.compile(r"(?i)x-apisports-key\"?\s*[=:]\s*[\"'][^\"'{}]{10,}"),
    ),
]


def files_that_would_be_committed() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES or path.name in SKIP_FILES:
            continue
        found.append(path)
    return found


@pytest.mark.parametrize(
    "label,pattern", SECRET_PATTERNS, ids=[name for name, _ in SECRET_PATTERNS]
)
def test_no_secret_shaped_string_is_committed(label: str, pattern: re.Pattern[str]) -> None:
    offenders = []
    for path in files_that_would_be_committed():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"possible {label} in: {offenders}"


def test_the_local_env_file_is_ignored() -> None:
    """The one file that holds real credentials must never be committable."""
    ignore_rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignore_rules
    assert "!.env.example" in ignore_rules


def test_the_example_env_files_ship_no_real_values() -> None:
    """Every `*.example` file is committed, so every one of them is scanned."""
    examples = sorted(REPO_ROOT.glob(".env*.example"))

    assert examples, "the example environment files are missing"
    for path in examples:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if any(word in name for word in ("PASSWORD", "KEY", "SECRET", "TOKEN")):
                assert value in {"", "change-me"}, f"{name} in {path.name} looks real"


def test_the_frontend_reads_no_secret_from_its_environment() -> None:
    """A `VITE_` variable is compiled into the bundle and served to everyone."""
    used = set()
    for path in (REPO_ROOT / "frontend" / "src").rglob("*.ts*"):
        used.update(re.findall(r"import\.meta\.env\.(\w+)", path.read_text(encoding="utf-8")))

    assert used == {"VITE_API_BASE_URL"}, f"unexpected build-time variables: {used}"


def test_the_frontend_sends_no_credentials() -> None:
    """The browser app talks to a same-origin proxy; there is nothing to authenticate
    with, and no place a key could be kept safely."""
    client = (REPO_ROOT / "frontend" / "src" / "api" / "client.ts").read_text(encoding="utf-8")

    for forbidden in ("Authorization", "api_key", "apiKey", "x-apisports-key"):
        assert forbidden not in client
