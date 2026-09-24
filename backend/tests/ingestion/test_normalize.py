from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.ingestion import normalize
from app.ingestion.normalize import NormalizationError
from tests.providers import builders


class TestText:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Manchester   United ", "Manchester United"),
            ("Atlético", "Atlético"),  # decomposed accent → composed (NFC)
            ("Tab\tand\nnewline", "Tab and newline"),
            ("   ", None),
            (None, None),
        ],
    )
    def test_clean_text(self, raw: str | None, expected: str | None) -> None:
        assert normalize.clean_text(raw) == expected

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Atlético Madrid", "Atletico Madrid FC"),
            ("AFC Bournemouth", "Bournemouth"),
            ("Brighton & Hove Albion", "Brighton and Hove Albion".replace(" and ", " & ")),
            ("RAJA CLUB ATHLETIC", "Raja Club Athletic"),
        ],
    )
    def test_name_key_matches_spelling_variants(self, a: str, b: str) -> None:
        assert normalize.name_key(a) == normalize.name_key(b)

    def test_name_key_keeps_different_clubs_apart(self) -> None:
        assert normalize.name_key("Manchester United") != normalize.name_key("Manchester City")

    def test_empty_provider_id_is_rejected(self) -> None:
        with pytest.raises(NormalizationError, match="provider id is empty"):
            normalize.external_id("  ")


class TestSeasonNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2023/2024", "2023/2024"),
            ("2023-2024", "2023/2024"),
            ("2023/24", "2023/2024"),
            ("2023-24", "2023/2024"),
            (f"2023{chr(0x2013)}24", "2023/2024"),  # en dash
            (" 2023 / 24 ", "2023/2024"),
            ("1999/00", "1999/2000"),
            ("2024", "2024"),
        ],
    )
    def test_recognised_formats(self, raw: str, expected: str) -> None:
        assert normalize.season_name(raw) == (expected, True)

    @pytest.mark.parametrize("raw", ["2023-25", "Apertura 2024", "23/24"])
    def test_unrecognised_formats_are_kept_not_guessed(self, raw: str) -> None:
        assert normalize.season_name(raw) == (raw, False)


class TestDates:
    def test_kickoff_is_converted_to_utc(self) -> None:
        casablanca = timezone(timedelta(hours=1))
        kickoff = datetime(2024, 5, 1, 21, 0, tzinfo=casablanca)

        assert normalize.utc(kickoff) == datetime(2024, 5, 1, 20, 0, tzinfo=UTC)
        assert normalize.utc(kickoff).tzinfo is UTC

    def test_naive_kickoff_is_rejected(self) -> None:
        with pytest.raises(NormalizationError, match="no timezone"):
            normalize.utc(datetime(2024, 5, 1, 21, 0))


class TestRecords:
    def test_team_record(self) -> None:
        # Surrounding whitespace is already stripped by the record type itself;
        # normalisation collapses inner whitespace and upper-cases codes.
        record = builders.team("42", "Arsenal   Women", code="ars", country="United   Kingdom")

        cleaned = normalize.team(record)

        assert (cleaned.external_id, cleaned.name, cleaned.code, cleaned.country) == (
            "42",
            "Arsenal Women",
            "ARS",
            "United Kingdom",
        )

    def test_season_record(self) -> None:
        cleaned, recognised = normalize.season(builders.season(name="2023-24"))

        assert (cleaned.name, recognised) == ("2023/2024", True)

    def test_match_record_keeps_exact_kickoff(self) -> None:
        kickoff = datetime(2023, 8, 11, 21, 0, 17, tzinfo=timezone(timedelta(hours=2)))

        cleaned = normalize.match(builders.match(kickoff_at=kickoff, venue="  Turf  Moor "))

        assert cleaned.kickoff_at == datetime(2023, 8, 11, 19, 0, 17, tzinfo=UTC)
        assert cleaned.venue == "Turf Moor"
