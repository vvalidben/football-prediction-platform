"""Provider-independent normalisation applied to every record before it is stored.

Adapters already convert provider formats into neutral records. This module makes
those records consistent *across* providers:

- **text**: Unicode NFC, surrounding and repeated whitespace removed, empty → None;
- **season names**: "2023-24", "2023/24", "2023-2024" → "2023/2024"; "2024" stays;
- **dates**: kickoff times as timezone-aware UTC (naive datetimes are rejected);
- **provider IDs**: trimmed strings, never empty;
- **codes**: team codes upper-case.

It never guesses: a season name it cannot parse is kept as cleaned text and
reported, rather than rewritten.
"""

import re
import unicodedata
from datetime import UTC, datetime

from app.providers.schemas import (
    ProviderCompetition,
    ProviderMatch,
    ProviderSeason,
    ProviderTeam,
    ProviderTeamMatchStatistics,
)

_WHITESPACE = re.compile(r"\s+")
# Separators: hyphen, slash or en dash (U+2013), as in "2023-24" or "2023/24".
_EN_DASH = chr(0x2013)
_SEASON_SPAN = re.compile(rf"^(?P<start>\d{{4}})\s*[-/{_EN_DASH}]\s*(?P<end>\d{{2}}|\d{{4}})$")
_SEASON_YEAR = re.compile(r"^\d{4}$")
#: Words that differ between providers' spellings of the same club.
_CLUB_AFFIXES = frozenset({"fc", "cf", "afc", "sc", "ac", "as", "ss", "club", "the"})


class NormalizationError(ValueError):
    """A value is unusable and the record cannot be stored."""


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", value)).strip()
    return cleaned or None


def required_text(value: str, field: str) -> str:
    cleaned = clean_text(value)
    if cleaned is None:
        raise NormalizationError(f"{field} is empty")
    return cleaned


def external_id(value: str) -> str:
    return required_text(str(value), "provider id")


def name_key(value: str) -> str:
    """Comparison key for names: no accents, case, punctuation or club affixes.

    Used only to *detect* possible duplicates ("Atlético Madrid" / "Atletico
    Madrid FC"); stored names keep the provider's spelling.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    words = re.sub(r"[^a-z0-9]+", " ", ascii_only.casefold()).split()
    return " ".join(word for word in words if word not in _CLUB_AFFIXES)


def season_name(value: str) -> tuple[str, bool]:
    """Return (canonical name, recognised). Unrecognised values come back cleaned."""
    cleaned = required_text(value, "season name")
    if _SEASON_YEAR.match(cleaned):
        return cleaned, True
    span = _SEASON_SPAN.match(cleaned)
    if span:
        start = int(span["start"])
        end_text = span["end"]
        if len(end_text) == 4:
            end = int(end_text)
        else:  # two-digit end year, possibly across a century: "1999/00"
            end = (start // 100) * 100 + int(end_text)
            if end < start:
                end += 100
        if end == start + 1:
            return f"{start}/{end}", True
    return cleaned, False


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise NormalizationError("kickoff time has no timezone")
    return value.astimezone(UTC)


def competition(record: ProviderCompetition) -> ProviderCompetition:
    return record.model_copy(
        update={
            "external_id": external_id(record.external_id),
            "name": required_text(record.name, "competition name"),
            "country": clean_text(record.country),
            "logo_url": clean_text(record.logo_url),
        }
    )


def season(record: ProviderSeason) -> tuple[ProviderSeason, bool]:
    name, recognised = season_name(record.name)
    normalised = record.model_copy(
        update={
            "external_id": external_id(record.external_id),
            "competition_external_id": external_id(record.competition_external_id),
            "name": name,
        }
    )
    return normalised, recognised


def team(record: ProviderTeam) -> ProviderTeam:
    code = clean_text(record.code)
    return record.model_copy(
        update={
            "external_id": external_id(record.external_id),
            "name": required_text(record.name, "team name"),
            "short_name": clean_text(record.short_name),
            "code": code.upper() if code else None,
            "country": clean_text(record.country),
            "logo_url": clean_text(record.logo_url),
        }
    )


def match(record: ProviderMatch) -> ProviderMatch:
    return record.model_copy(
        update={
            "external_id": external_id(record.external_id),
            "competition_external_id": external_id(record.competition_external_id),
            "season_external_id": external_id(record.season_external_id),
            "home_team_external_id": external_id(record.home_team_external_id),
            "away_team_external_id": external_id(record.away_team_external_id),
            "kickoff_at": utc(record.kickoff_at),
            "stage": clean_text(record.stage),
            "venue": clean_text(record.venue),
            "referee": clean_text(record.referee),
        }
    )


def team_statistics(record: ProviderTeamMatchStatistics) -> ProviderTeamMatchStatistics:
    return record.model_copy(
        update={
            "match_external_id": external_id(record.match_external_id),
            "team_external_id": external_id(record.team_external_id),
        }
    )
