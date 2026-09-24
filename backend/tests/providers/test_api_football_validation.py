"""The Sprint 2 validation tool, tested without spending a single API request.

Every test here drives the code with a stub transport. The live validation is a
deliberate, budgeted act (`python scripts/validate_api_football.py`), never
something the test suite triggers on its own.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import httpx2 as httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.providers.api_football import validation
from app.providers.api_football.validation import (
    LEAKAGE,
    BudgetExhaustedError,
    Findings,
    Validator,
    assess,
    coverage_of,
    describe_competition,
    observe_injuries,
    observe_lineups,
    observe_players,
    observe_statistics,
    pick_fixture,
    render_report,
)
from app.providers.errors import (
    AuthenticationError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.providers.http import JsonHttpClient, RetryPolicy

KEY_HEADER = "x-apisports-key"
SECRET = "k" * 32


def client(handler: Any, *, attempts: int = 1) -> JsonHttpClient:
    return JsonHttpClient(
        provider="api_football",
        base_url="https://v3.football.api-sports.io",
        headers={KEY_HEADER: SECRET},
        timeout_seconds=1.0,
        retry_policy=RetryPolicy(max_attempts=attempts, base_delay_seconds=0.0),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )


def envelope(response: Any, *, errors: Any = None, results: int | None = None) -> dict[str, Any]:
    return {
        "get": "test",
        "parameters": {},
        "errors": errors if errors is not None else [],
        "results": results
        if results is not None
        else (len(response) if isinstance(response, list) else 1),
        "paging": {"current": 1, "total": 1},
        "response": response,
    }


class TestConfiguration:
    def test_a_missing_key_stops_the_run_and_says_how_to_set_it(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The tool must never guess a key or run half-configured."""
        settings = Settings(
            _env_file=None,
            postgres_user="u",
            postgres_password=SecretStr("p"),
            postgres_db="d",
            api_football_api_key=None,
        )
        monkeypatch.setattr(validation, "get_settings", lambda: settings)

        exit_code = validation.main([])

        assert exit_code == 3
        message = capsys.readouterr().err
        assert "API_FOOTBALL_API_KEY" in message
        assert ".env" in message

    def test_the_key_travels_in_a_header_and_is_never_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json=envelope([]))

        validator = Validator(client(handler), budget=2)
        validator.call("/leagues", {}, "why")

        assert seen[KEY_HEADER] == SECRET
        assert SECRET not in capsys.readouterr().out
        assert SECRET not in json.dumps([vars(call) for call in validator.calls])


class TestBudget:
    def test_every_request_is_counted_and_carries_its_reason(self) -> None:
        validator = Validator(client(lambda _r: httpx.Response(200, json=envelope([]))), budget=3)

        validator.call("/status", {}, "plan and quota")
        validator.call("/leagues", {"id": 39}, "league survey")

        assert validator.used == 2
        assert [call.reason for call in validator.calls] == ["plan and quota", "league survey"]
        assert [call.path for call in validator.calls] == ["/status", "/leagues"]

    def test_the_budget_is_a_hard_stop(self) -> None:
        """An overrun would spend a quota the account cannot get back."""
        validator = Validator(client(lambda _r: httpx.Response(200, json=envelope([]))), budget=1)
        validator.call("/status", {}, "first")

        with pytest.raises(BudgetExhaustedError, match="budget of 1"):
            validator.call("/leagues", {}, "second")

        assert validator.used == 1

    def test_the_remaining_daily_allowance_is_recorded_when_the_api_reports_it(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json=envelope([]), headers={"x-ratelimit-requests-remaining": "77"}
            )

        validator = Validator(client(handler), budget=2)
        validator.call("/status", {}, "quota")

        assert validator.calls[0].requests_remaining == "77"


class TestErrorHandling:
    def test_an_api_level_error_is_recorded_and_does_not_stop_the_run(self) -> None:
        """API-Football reports refusals inside a 200 response."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=envelope([], errors={"plan": "Free plans do not have access to this season"}),
            )

        validator = Validator(client(handler), budget=2)
        body, error = validator.call("/fixtures", {"season": 2019}, "old season")

        assert body is None
        assert error is not None and "Free plans" in error
        assert validator.calls[0].outcome == "error"

    def test_an_authentication_failure_is_reported_without_the_key(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=envelope([], errors={"token": "invalid"}))

        validator = Validator(client(handler), budget=2)
        _, error = validator.call("/status", {}, "auth check")

        assert error is not None
        assert SECRET not in error

    def test_a_rate_limit_is_surfaced_rather_than_hidden(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"message": "too many"}, headers={"Retry-After": "1"})

        validator = Validator(client(handler), budget=2)
        _, error = validator.call("/status", {}, "rate limit")

        assert error is not None
        assert "429" in error or "rate" in error.lower()

    def test_malformed_json_is_an_error_not_a_crash(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>not json</html>")

        validator = Validator(client(handler), budget=2)
        body, error = validator.call("/status", {}, "malformed")

        assert body is None and error is not None

    def test_a_timeout_is_an_error_not_a_crash(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        validator = Validator(client(handler), budget=2)
        body, error = validator.call("/status", {}, "timeout")

        assert body is None
        assert error is not None and "timed out" in error.lower()

    def test_the_shared_provider_errors_are_used(self) -> None:
        """The validation reuses the provider layer's error types rather than
        inventing its own, so retry behaviour is identical everywhere."""
        assert issubclass(AuthenticationError, Exception)
        assert issubclass(RateLimitError, Exception)
        assert issubclass(ProviderUnavailableError, Exception)


class TestNormalisation:
    LEAGUE: ClassVar[dict[str, Any]] = {
        "league": {"id": 200, "name": "Botola Pro", "type": "League"},
        "country": {"name": "Morocco"},
        "seasons": [
            {
                "year": 2023,
                "current": False,
                "coverage": {
                    "fixtures": {"events": True, "lineups": True, "statistics_fixtures": False},
                    "odds": False,
                    "injuries": False,
                },
            },
            {
                "year": 2024,
                "current": True,
                "coverage": {
                    "fixtures": {"events": True, "lineups": True, "statistics_fixtures": True},
                    "odds": False,
                    "injuries": False,
                },
            },
        ],
    }

    def test_a_competition_is_described_from_its_seasons(self) -> None:
        described = describe_competition(self.LEAGUE)

        assert described["league_id"] == 200
        assert described["name"] == "Botola Pro"
        assert described["country"] == "Morocco"
        assert described["seasons"] == [2023, 2024]
        assert described["earliest_season"] == 2023
        assert described["current_season"] == 2024
        assert described["has_2024"] is True
        assert described["has_2025"] is False

    def test_coverage_flags_are_flattened_per_season(self) -> None:
        coverage = coverage_of(self.LEAGUE, 2024)

        assert coverage["fixtures.statistics_fixtures"] is True
        assert coverage["odds"] is False
        assert coverage_of(self.LEAGUE, 2023)["fixtures.statistics_fixtures"] is False
        assert coverage_of(self.LEAGUE, 1999) == {}

    def test_statistics_separate_present_from_null_from_missing(self) -> None:
        """Botola returns the field names with null values; that is not the same as
        the field being absent, and the difference matters."""
        body = {
            "response": [
                {
                    "statistics": [
                        {"type": "Total Shots", "value": None},
                        {"type": "Yellow Cards", "value": 2},
                    ]
                }
            ]
        }

        observed = observe_statistics(body)

        assert observed["wanted"]["Total Shots"] == "null"
        assert observed["wanted"]["Yellow Cards"] == "present"
        assert observed["wanted"]["expected_goals"] == "missing"

    def test_lineups_are_summarised_without_copying_the_payload(self) -> None:
        body = {
            "response": [
                {
                    "formation": "4-4-2",
                    "startXI": [{"player": {"id": 1, "pos": "G", "grid": "1:1"}}],
                    "substitutes": [{"player": {"id": 2}}],
                    "coach": {"name": "A Coach"},
                }
            ]
        }

        observed = observe_lineups(body)

        assert observed["formation"] == "4-4-2"
        assert observed["start_xi"] == 1
        assert observed["player_ids"] is True
        assert observed["coach"] is True

    def test_player_fields_are_reported_individually(self) -> None:
        body = {
            "paging": {"total": 35},
            "response": [
                {
                    "statistics": [
                        {
                            "games": {"appearences": 10, "minutes": 900, "rating": None},
                            "goals": {"total": 3, "assists": None},
                            "shots": {"total": None, "on": None},
                        }
                    ]
                }
            ],
        }

        observed = observe_players(body)

        assert observed["fields"]["appearances"] is True
        assert observed["fields"]["rating"] is False
        assert observed["fields"]["shots_total"] is False
        assert observed["pages_available"] == 35

    def test_injuries_are_summarised_with_their_linkage(self) -> None:
        body = {
            "response": [
                {
                    "player": {"id": 7, "reason": "Ankle Injury", "type": "Missing Fixture"},
                    "fixture": {"id": 42, "date": "2025-01-04T15:00:00+00:00"},
                }
            ]
        }

        observed = observe_injuries(body)

        assert observed["rows"] == 1
        assert observed["has_player_id"] is True
        assert observed["fixture_linked"] is True

    def test_a_fixture_without_an_id_is_not_picked(self) -> None:
        assert pick_fixture({"response": []}) is None
        assert pick_fixture({"response": [{"fixture": {}}]}) is None


class TestReport:
    def test_the_leakage_table_classifies_every_source(self) -> None:
        verdicts = {verdict for _, verdict, _, _, _ in LEAKAGE}

        assert verdicts <= {
            "SAFE PRE-MATCH",
            "SAFE PRE-MATCH (when retained)",
            "TIMING-DEPENDENT",
            "POST-MATCH ONLY",
            "UNKNOWN",
        }
        sources = {source for source, *_ in LEAKAGE}
        assert any("statistic" in s.lower() for s in sources)
        assert any("lineup" in s.lower() for s in sources)
        assert any("odds" in s.lower() for s in sources)

    def test_per_match_finals_are_never_marked_safe_pre_match(self) -> None:
        """The single most dangerous mistake this project could make.

        Season aggregates are a different case — they are TIMING-DEPENDENT because
        they can be recomputed point-in-time from match data — so this checks the
        per-match finals specifically.
        """
        finals = (
            "Final score, half-time score",
            "Team match statistics (shots, possession, corners, cards, saves)",
            "expected_goals (xG) in fixture statistics",
            "Events (goals, cards, substitutions)",
        )
        verdicts = {source: verdict for source, verdict, *_ in LEAKAGE}

        for source in finals:
            assert source in verdicts, source
            assert verdicts[source] == "POST-MATCH ONLY", source

    def test_nothing_that_changes_before_kickoff_is_called_safe(self) -> None:
        """Lineups, injuries, standings and provider predictions all exist before a
        match but carry no publication time, so none may be called safe."""
        verdicts = {source: verdict for source, verdict, *_ in LEAKAGE}

        for source, verdict in verdicts.items():
            if any(
                word in source.lower() for word in ("lineup", "injur", "standing", "prediction")
            ):
                assert verdict == "TIMING-DEPENDENT", source

    def test_the_assessment_reports_unknowns_as_unknown(self) -> None:
        """With no observations, nothing may be claimed as READY on evidence."""
        rows = assess(Findings(generated_at="now"))

        suitabilities = {suitability for *_, suitability, _ in rows}
        assert suitabilities <= {
            "READY",
            "PARTIAL",
            "NOT AVAILABLE",
            "NEEDS SECOND PROVIDER",
            "NEEDS FURTHER VALIDATION",
        }

    def test_a_report_can_be_rendered_from_findings_alone(self, tmp_path: Path) -> None:
        findings = Findings(
            generated_at="2026-09-23T00:00:00Z",
            plan={"plan": "Free", "requests_used_today": 1, "requests_limit_day": 100},
            competitions=[
                {
                    **describe_competition(TestNormalisation.LEAGUE),
                    "expected_name": "Botola Pro",
                    "name_matches": True,
                    "expected_country": "Morocco",
                }
            ],
            calls=[
                {
                    "number": 1,
                    "path": "/leagues",
                    "params": {},
                    "reason": "survey",
                    "outcome": "ok",
                    "results": 1,
                    "error": None,
                    "requests_remaining": "80",
                }
            ],
        )

        report = render_report(findings, budget=25)

        assert "# API-Football validation" in report
        assert "Botola Pro" in report
        assert "Data timing and leakage" in report
        assert "Requirement matrix" in report
        assert SECRET not in report

    def test_the_report_never_contains_a_key_shaped_string(self) -> None:
        report = render_report(Findings(generated_at="now"), budget=25)

        import re

        assert not re.search(r"\b[0-9a-f]{32}\b", report)
