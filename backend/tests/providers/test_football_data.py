"""The football-data.org adapter, against payloads shaped like the real API's."""

from datetime import UTC, date, datetime
from typing import Any

import httpx2 as httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.models.enums import CompetitionType, MatchStatus
from app.providers.base import ProviderOperation
from app.providers.errors import AuthenticationError, ProviderRequestError
from app.providers.football_data import FootballDataOrgProvider
from app.providers.schemas import ProviderMatch, ScorePair
from tests.providers.fake_football_data import (
    PREMIER_LEAGUE,
    SEASON_2025,
    FakeFootballData,
    match_payload,
    score,
)


@pytest.fixture
def api() -> FakeFootballData:
    return FakeFootballData()


def one_match(api: FakeFootballData, **overrides: Any) -> ProviderMatch:
    api.matches = [match_payload(**overrides)]
    result = api.provider().get_matches(PREMIER_LEAGUE, SEASON_2025)
    assert not result.rejected
    return result.records[0]


class TestCompetitions:
    def test_a_league_keeps_its_country(self, api: FakeFootballData) -> None:
        league = api.provider().get_competitions().records[0]

        assert league.external_id == PREMIER_LEAGUE
        assert league.name == "Premier League"
        assert league.type is CompetitionType.LEAGUE
        assert league.country == "England"

    def test_international_competitions_have_no_country(self, api: FakeFootballData) -> None:
        records = {c.external_id: c for c in api.provider().get_competitions().records}

        assert records["2000"].country is None
        assert records["2001"].country is None
        assert records["2000"].type is CompetitionType.CUP
        assert records["9"].type is CompetitionType.SUPER_CUP


class TestSeasons:
    def test_seasons_the_plan_cannot_read_are_not_offered(self, api: FakeFootballData) -> None:
        """The API lists every season back to 1888 but answers 403 before the plan's
        window; offering them would create rows no import can ever fill."""
        seasons = api.provider().get_seasons(PREMIER_LEAGUE).records

        assert {s.external_id for s in seasons} == {
            "2021:2026",
            "2021:2025",
            "2021:2024",
            "2021:2023",
        }

    def test_the_first_season_is_configurable_for_paid_plans(self, api: FakeFootballData) -> None:
        seasons = api.provider(football_data_first_season=2022).get_seasons(PREMIER_LEAGUE).records

        assert "2021:2022" in {s.external_id for s in seasons}

    def test_season_fields(self, api: FakeFootballData) -> None:
        seasons = {s.external_id: s for s in api.provider().get_seasons(PREMIER_LEAGUE).records}

        current = seasons["2021:2026"]
        assert current.name == "2026/2027"
        assert current.is_current
        assert current.start_date == date(2026, 8, 21)
        assert current.competition_external_id == PREMIER_LEAGUE
        assert not seasons["2021:2025"].is_current

    def test_the_plan_offers_no_statistics(self, api: FakeFootballData) -> None:
        seasons = api.provider().get_seasons(PREMIER_LEAGUE).records

        assert all(s.statistics_available is False for s in seasons)


class TestTeams:
    def test_team_fields(self, api: FakeFootballData) -> None:
        team = api.provider().get_teams(PREMIER_LEAGUE, SEASON_2025).records[0]

        assert team.external_id == "64"
        assert team.name == "Liverpool FC"
        assert team.short_name == "Liverpool"
        assert team.code == "LIV"
        assert team.country == "England"
        assert not team.is_national_team

    def test_the_season_year_is_what_is_requested(self, api: FakeFootballData) -> None:
        api.provider().get_teams(PREMIER_LEAGUE, SEASON_2025)

        assert api.requests[-1].url.params["season"] == "2025"


class TestMatches:
    def test_finished_match(self, api: FakeFootballData) -> None:
        match = one_match(api)

        assert match.external_id == "537785"
        assert match.competition_external_id == PREMIER_LEAGUE
        assert match.season_external_id == SEASON_2025
        assert match.home_team_external_id == "64"
        assert match.away_team_external_id == "1044"
        assert match.kickoff_at == datetime(2025, 8, 15, 19, 0, tzinfo=UTC)
        assert match.status is MatchStatus.FINISHED
        assert match.matchday == 1
        assert match.stage == "regular_season"
        assert match.score == ScorePair(home=4, away=2)
        assert match.half_time_score == ScorePair(home=1, away=0)
        assert match.penalty_score is None

    def test_an_upcoming_match_has_no_score(self, api: FakeFootballData) -> None:
        match = one_match(api, status="TIMED", score=score())

        assert match.status is MatchStatus.SCHEDULED
        assert match.score is None
        assert match.half_time_score is None

    def test_a_shootout_score_excludes_the_penalties(self, api: FakeFootballData) -> None:
        """Recorded from Liverpool v PSG, 2025-03-11: the match ended 0-1 and the API's
        `fullTime` says 1-5, because it adds the shootout."""
        match = one_match(
            api,
            score=score((1, 5), (0, 1), duration="PENALTY_SHOOTOUT", penalties=(1, 4)),
        )

        assert match.score == ScorePair(home=0, away=1)
        assert match.penalty_score == ScorePair(home=1, away=4)

    def test_an_extra_time_score_counts_extra_time(self, api: FakeFootballData) -> None:
        match = one_match(api, score=score((3, 2), (0, 1), duration="EXTRA_TIME"))

        assert match.score == ScorePair(home=3, away=2)
        assert match.penalty_score is None

    def test_the_referee_is_the_main_official(self, api: FakeFootballData) -> None:
        assert one_match(api).referee == "Anthony Taylor"

    def test_no_referees_listed(self, api: FakeFootballData) -> None:
        assert one_match(api, referees=[]).referee is None

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("SCHEDULED", MatchStatus.SCHEDULED),
            ("TIMED", MatchStatus.SCHEDULED),
            ("IN_PLAY", MatchStatus.LIVE),
            ("PAUSED", MatchStatus.LIVE),
            ("EXTRA_TIME", MatchStatus.LIVE),
            ("PENALTY_SHOOTOUT", MatchStatus.LIVE),
            ("LIVE", MatchStatus.LIVE),
            ("SUSPENDED", MatchStatus.SUSPENDED),
            ("POSTPONED", MatchStatus.POSTPONED),
            ("CANCELLED", MatchStatus.CANCELLED),
            ("AWARDED", MatchStatus.FINISHED),
        ],
    )
    def test_statuses(self, api: FakeFootballData, code: str, expected: MatchStatus) -> None:
        assert one_match(api, status=code).status is expected

    def test_a_malformed_match_does_not_lose_the_others(self, api: FakeFootballData) -> None:
        api.matches = [match_payload(1, status="NEVER_HEARD_OF_IT"), match_payload(2)]

        result = api.provider().get_matches(PREMIER_LEAGUE, SEASON_2025)

        assert [m.external_id for m in result.records] == ["2"]
        assert result.rejected[0].external_id == "1"
        assert "NEVER_HEARD_OF_IT" in result.rejected[0].reason

    def test_a_finished_match_without_a_score_is_rejected(self, api: FakeFootballData) -> None:
        api.matches = [match_payload(1, score=score())]

        result = api.provider().get_matches(PREMIER_LEAGUE, SEASON_2025)

        assert not result.records
        assert result.rejected[0].external_id == "1"

    def test_a_tie_whose_teams_are_not_decided_is_not_a_match_yet(
        self, api: FakeFootballData
    ) -> None:
        """Recorded from the 2026 Copa Libertadores final. It is not malformed, so it must
        not be logged as a data fault, and it arrives once the semi-finals are played."""
        nobody = {"id": None, "name": None, "shortName": None, "tla": None, "crest": None}
        api.matches = [
            match_payload(1, status="SCHEDULED", score=score(), homeTeam=nobody, awayTeam=nobody),
            match_payload(2),
        ]

        result = api.provider().get_matches(PREMIER_LEAGUE, SEASON_2025)

        assert [m.external_id for m in result.records] == ["2"]
        assert not result.rejected

    def test_a_season_of_another_competition_is_refused(self, api: FakeFootballData) -> None:
        with pytest.raises(ProviderRequestError, match="does not belong"):
            api.provider().get_matches(PREMIER_LEAGUE, "2000:2026")


class TestTransport:
    def test_the_token_travels_in_the_documented_header(self, api: FakeFootballData) -> None:
        api.provider().get_competitions()

        assert api.requests[-1].headers["X-Auth-Token"] == "test-token"
        assert "test-token" not in str(api.requests[-1].url)

    def test_a_rejected_token_is_an_authentication_error(self, api: FakeFootballData) -> None:
        api.failures.append(lambda _: httpx.Response(403, json={"message": "restricted"}))

        with pytest.raises(AuthenticationError):
            api.provider().get_competitions()

    def test_a_rate_limit_waits_for_the_window_the_api_names(self, api: FakeFootballData) -> None:
        api.failures.append(
            lambda _: httpx.Response(429, headers={"X-RequestCounter-Reset": "37"}, json={})
        )
        sleeps: list[float] = []

        records = api.provider(sleeps).get_competitions().records

        assert records
        assert sleeps == [37.0]

    def test_without_a_token_it_says_which_variable_is_missing(self) -> None:
        settings = Settings(
            _env_file=None,
            postgres_user="unused",
            postgres_password=SecretStr("unused"),
            postgres_db="unused",
            football_data_api_key=None,
        )

        with pytest.raises(AuthenticationError, match="FOOTBALL_DATA_API_KEY"):
            FootballDataOrgProvider.from_settings(settings)


class TestCapabilities:
    def test_it_only_claims_what_it_implements(self, api: FakeFootballData) -> None:
        provider = api.provider()

        assert provider.supports(ProviderOperation.MATCHES)
        assert not provider.supports(ProviderOperation.TEAM_STATISTICS)
        assert not provider.supports(ProviderOperation.ODDS)
