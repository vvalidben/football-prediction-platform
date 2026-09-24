"""API-Football adapter: normalisation of documented response shapes and error mapping."""

import copy
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.models.enums import CompetitionType, MatchStatus
from app.providers.api_football import ApiFootballProvider
from app.providers.api_football.normalize import STATUS_MAP, parse_round
from app.providers.errors import (
    AuthenticationError,
    MalformedResponseError,
    ProviderRequestError,
    QuotaExceededError,
    RateLimitError,
)
from app.providers.schemas import ScorePair
from tests.providers.fake_api import FakeApiFootball, body, envelope, load_fixture, raw

PL, PL_2023 = "39", "39:2023"


@pytest.fixture
def api() -> FakeApiFootball:
    return FakeApiFootball()


@pytest.fixture
def provider(api: FakeApiFootball) -> ApiFootballProvider:
    return api.provider()


class TestCompetitions:
    def test_normalises_leagues_and_rejects_malformed_record(
        self, provider: ApiFootballProvider
    ) -> None:
        result = provider.get_competitions()

        by_id = {c.external_id: c for c in result.records}
        assert set(by_id) == {"39", "2", "667"}
        premier_league = by_id["39"]
        assert premier_league.name == "Premier League"
        assert premier_league.type is CompetitionType.LEAGUE
        assert premier_league.country == "England"
        assert premier_league.logo_url == "https://media.api-sports.io/football/leagues/39.png"
        # "World" means international: no country.
        assert by_id["2"].country is None
        assert by_id["2"].type is CompetitionType.CUP
        assert by_id["667"].type is CompetitionType.FRIENDLY

        assert len(result.rejected) == 1
        assert "league.id" in result.rejected[0].reason

    def test_sends_api_key_header(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        provider.get_competitions()

        assert api.requests[0].headers["x-apisports-key"] == "test-key"


class TestSeasons:
    def test_season_ids_include_the_league(self, provider: ApiFootballProvider) -> None:
        result = provider.get_seasons(PL)

        assert [s.external_id for s in result.records] == ["39:2022", "39:2023"]
        current = result.records[1]
        assert current.name == "2023/2024"
        assert current.start_date == date(2023, 8, 11)
        assert current.end_date == date(2024, 5, 19)
        assert current.is_current is True
        assert result.records[0].is_current is False
        assert all(s.competition_external_id == PL for s in result.records)

    def test_calendar_year_season_uses_a_single_year_name(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        league = copy.deepcopy(load_fixture("leagues_id_39.json")["response"][0])
        league["seasons"] = [
            {"year": 2024, "start": "2024-03-01", "end": "2024-11-30", "current": False}
        ]
        api.serve_json("/leagues", {"id": "39"}, envelope([league]))

        assert provider.get_seasons(PL).records[0].name == "2024"

    def test_season_with_reversed_dates_is_rejected(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        league = copy.deepcopy(load_fixture("leagues_id_39.json")["response"][0])
        league["seasons"][0].update(start="2023-05-01", end="2022-08-01")
        api.serve_json("/leagues", {"id": "39"}, envelope([league]))

        result = provider.get_seasons(PL)

        assert [r.external_id for r in result.rejected] == ["39:2022"]
        assert [s.external_id for s in result.records] == ["39:2023"]


class TestTeams:
    def test_normalises_teams(self, api: FakeApiFootball, provider: ApiFootballProvider) -> None:
        result = provider.get_teams(PL, PL_2023)

        assert len(result.records) == 20
        assert result.rejected == []
        arsenal = next(t for t in result.records if t.external_id == "42")
        assert (arsenal.name, arsenal.code, arsenal.country) == ("Arsenal", "ARS", "England")
        assert arsenal.is_national_team is False
        assert dict(api.requests[-1].url.params) == {"league": "39", "season": "2023"}

    @pytest.mark.parametrize("season_id", ["2023", "39-2023", "140:2023"])
    def test_rejects_foreign_or_malformed_season_ids(
        self, provider: ApiFootballProvider, season_id: str
    ) -> None:
        with pytest.raises(ProviderRequestError):
            provider.get_teams(PL, season_id)


class TestMatches:
    def test_normalises_finished_fixtures(self, provider: ApiFootballProvider) -> None:
        result = provider.get_matches(PL, PL_2023)

        assert len(result.records) == 10
        assert result.rejected == []
        opener = next(m for m in result.records if m.external_id == "1035037")
        assert opener.home_team_external_id == "44"  # Burnley
        assert opener.away_team_external_id == "50"  # Manchester City
        assert opener.kickoff_at == datetime(2023, 8, 11, 19, 0, tzinfo=UTC)
        assert opener.status is MatchStatus.FINISHED
        assert opener.score == ScorePair(home=0, away=3)
        assert opener.half_time_score == ScorePair(home=0, away=2)
        assert opener.penalty_score is None
        assert (opener.matchday, opener.stage) == (1, "regular_season")
        assert opener.venue == "Turf Moor"
        assert opener.referee is None  # missing in the source: stays unknown
        assert opener.season_external_id == PL_2023

    def test_kickoff_is_converted_to_utc(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        item = copy.deepcopy(load_fixture("fixtures_league_39_season_2023.json")["response"][0])
        item["fixture"]["date"] = "2023-08-11T21:00:00+02:00"
        api.serve_json("/fixtures", {"league": "39", "season": "2023"}, envelope([item]))

        match = provider.get_matches(PL, PL_2023).records[0]

        assert match.kickoff_at == datetime(2023, 8, 11, 19, 0, tzinfo=UTC)
        assert match.kickoff_at.utcoffset() is not None

    def test_scheduled_fixture_has_no_score(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        item = copy.deepcopy(load_fixture("fixtures_league_39_season_2023.json")["response"][0])
        item["fixture"]["status"] = {"long": "Not Started", "short": "NS", "elapsed": None}
        item["goals"] = {"home": None, "away": None}
        item["score"]["halftime"] = {"home": None, "away": None}
        api.serve_json("/fixtures", {"league": "39", "season": "2023"}, envelope([item]))

        match = provider.get_matches(PL, PL_2023).records[0]

        assert match.status is MatchStatus.SCHEDULED
        assert match.score is None
        assert match.half_time_score is None

    def test_malformed_fixtures_are_rejected_individually(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        items = copy.deepcopy(load_fixture("fixtures_league_39_season_2023.json")["response"][:4])
        items[0]["fixture"]["status"]["short"] = "ZZZ"  # unknown status
        items[1]["goals"] = {"home": None, "away": None}  # finished without a score
        del items[2]["teams"]  # structurally broken
        api.serve_json("/fixtures", {"league": "39", "season": "2023"}, envelope(items))

        result = provider.get_matches(PL, PL_2023)

        assert [m.external_id for m in result.records] == ["1035040"]
        reasons = {r.external_id: r.reason for r in result.rejected}
        assert set(reasons) == {"1035037", "1035038", "1035039"}
        assert "unknown fixture status 'ZZZ'" in reasons["1035037"]
        assert "finished match has no score" in reasons["1035038"]
        assert "teams" in reasons["1035039"]

    def test_follows_pagination(self, api: FakeApiFootball, provider: ApiFootballProvider) -> None:
        items = load_fixture("fixtures_league_39_season_2023.json")["response"]
        params = {"league": "39", "season": "2023"}
        api.serve_json("/fixtures", params, envelope(items[:6], paging={"current": 1, "total": 2}))
        api.serve_json(
            "/fixtures",
            {**params, "page": "2"},
            envelope(items[6:], paging={"current": 2, "total": 2}),
        )

        assert len(provider.get_matches(PL, PL_2023).records) == 10


@pytest.mark.parametrize(
    ("short", "expected"),
    [
        ("NS", MatchStatus.SCHEDULED),
        ("TBD", MatchStatus.SCHEDULED),
        ("1H", MatchStatus.LIVE),
        ("HT", MatchStatus.LIVE),
        ("P", MatchStatus.LIVE),
        ("FT", MatchStatus.FINISHED),
        ("AET", MatchStatus.FINISHED),
        ("PEN", MatchStatus.FINISHED),
        ("PST", MatchStatus.POSTPONED),
        ("CANC", MatchStatus.CANCELLED),
        ("ABD", MatchStatus.ABANDONED),
        ("SUSP", MatchStatus.SUSPENDED),
    ],
)
def test_status_mapping(short: str, expected: MatchStatus) -> None:
    assert STATUS_MAP[short] is expected


@pytest.mark.parametrize(
    ("round_name", "expected"),
    [
        ("Regular Season - 12", ("regular_season", 12)),
        ("Group Stage - 3", ("group_stage", 3)),
        ("Round of 16", ("round_of_16", None)),
        ("Final", ("final", None)),
        (None, (None, None)),
    ],
)
def test_round_parsing(round_name: str | None, expected: tuple[str | None, int | None]) -> None:
    assert parse_round(round_name) == expected


class TestErrorsInsideSuccessfulResponses:
    @pytest.mark.parametrize(
        ("errors", "exception"),
        [
            ({"token": "Error/Missing application key."}, AuthenticationError),
            ({"requests": "You have reached the request limit for the day."}, QuotaExceededError),
            ({"season": "The Season field must contain 4 characters."}, ProviderRequestError),
            (["Something went wrong"], ProviderRequestError),
        ],
    )
    def test_permanent_errors(
        self,
        api: FakeApiFootball,
        provider: ApiFootballProvider,
        errors: object,
        exception: type[Exception],
    ) -> None:
        api.serve_json("/leagues", {}, envelope([], errors=errors))

        with pytest.raises(exception):
            provider.get_competitions()
        assert len(api.requests) == 1  # not retried

    def test_rate_limit_error_in_body_is_retried(self) -> None:
        api = FakeApiFootball()
        sleeps: list[float] = []
        provider = api.provider(sleeps=sleeps)
        api.fail_next(body(envelope([], errors={"rateLimit": "Too many requests."})))

        result = provider.get_competitions()

        assert len(result.records) == 3
        assert len(api.requests) == 2
        assert len(sleeps) == 1
        assert sleeps[0] >= 60  # waits out the per-minute window

    def test_persistent_rate_limit_surfaces(self) -> None:
        api = FakeApiFootball()
        provider = api.provider(provider_max_attempts=2)
        api.fail_next(*[body(envelope([], errors={"rateLimit": "Too many"}))] * 2)

        with pytest.raises(RateLimitError):
            provider.get_competitions()

    @pytest.mark.parametrize(
        "payload",
        [raw(b"not json"), body({"unexpected": True}), body({"response": "not-a-list"})],
        ids=["not-json", "no-envelope", "bad-response-type"],
    )
    def test_malformed_envelopes(self, payload: object) -> None:
        api = FakeApiFootball()
        provider = api.provider()
        api.fail_next(payload)  # type: ignore[arg-type]

        with pytest.raises(MalformedResponseError):
            provider.get_competitions()


def test_missing_api_key_is_reported_clearly() -> None:
    settings = Settings(
        _env_file=None,
        postgres_user="u",
        postgres_password=SecretStr("p"),
        postgres_db="d",
        api_football_api_key=None,
    )

    with pytest.raises(AuthenticationError, match="API_FOOTBALL_API_KEY is not set"):
        ApiFootballProvider.from_settings(settings)


class TestStatistics:
    def test_normalises_documented_statistic_labels(self, provider: ApiFootballProvider) -> None:
        result = provider.get_team_statistics(["1035037"])

        assert result.rejected == []
        home = next(r for r in result.records if r.team_external_id == "44")
        raw = load_fixture("fixture_details_league_39_season_2023.json")["response"][0]
        source = {s["type"]: s["value"] for s in raw["statistics"][0]["statistics"]}
        assert home.shots == source["Total Shots"]
        assert home.shots_on_target == source["Shots on Goal"]
        assert home.corners == source["Corner Kicks"]
        assert home.possession == Decimal(source["Ball Possession"].rstrip("%"))
        assert home.expected_goals == Decimal(source["expected_goals"])
        assert home.red_cards is None  # null in the source: not reported, not zero
        # Labels without a column are kept, nulls dropped.
        assert home.extra == {
            "shots_outsidebox": source["Shots outsidebox"],
            "passes": source["Passes %"],
        }

    def test_fixture_without_statistics_yields_nothing(self, provider: ApiFootballProvider) -> None:
        assert provider.get_team_statistics(["1035046"]).records == []

    def test_requests_are_batched_by_twenty(self, api: FakeApiFootball) -> None:
        provider = api.provider()
        ids = [str(1035037 + n) for n in range(10)] + [str(900 + n) for n in range(15)]

        result = provider.get_team_statistics(ids)

        batches = [r.url.params["ids"].split("-") for r in api.requests]
        assert [len(b) for b in batches] == [20, 5]
        assert len(result.records) == 18  # 9 fixtures with statistics x 2 teams

    def test_malformed_statistic_value_rejects_the_fixture(
        self, api: FakeApiFootball, provider: ApiFootballProvider
    ) -> None:
        item = copy.deepcopy(api.fixture_details[1035037])
        item["statistics"][0]["statistics"][2]["value"] = "lots"  # Total Shots
        api.fixture_details[1035037] = item

        result = provider.get_team_statistics(["1035037"])

        assert result.records == []
        assert result.rejected[0].external_id == "1035037"
        assert "not a number" in result.rejected[0].reason

    def test_season_coverage_flag_is_exposed(self, provider: ApiFootballProvider) -> None:
        seasons = provider.get_seasons(PL).records

        assert all(s.statistics_available is True for s in seasons)
