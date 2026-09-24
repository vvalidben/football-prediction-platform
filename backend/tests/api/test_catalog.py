"""GET /competitions, /teams, /matches/upcoming and /matches/{match_id}."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


class TestCompetitions:
    def test_lists_competitions_with_the_current_season(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        response = client.get("/competitions")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert [c["name"] for c in body["items"]] == ["FA Cup", "Premier League"]
        premier = body["items"][1]
        assert premier["type"] == "league"
        assert premier["country"] == "England"
        assert premier["current_season"] == "2025/2026"

    def test_filters_and_paginates(self, client: TestClient, league: dict[str, int]) -> None:
        assert client.get("/competitions", params={"country": "england"}).json()["total"] == 2
        assert client.get("/competitions", params={"country": "spain"}).json()["items"] == []

        page = client.get("/competitions", params={"limit": 1, "offset": 1}).json()

        assert page["total"] == 2 and page["limit"] == 1 and page["offset"] == 1
        assert [c["name"] for c in page["items"]] == ["Premier League"]

    def test_rejects_an_impossible_page(self, client: TestClient) -> None:
        assert client.get("/competitions", params={"limit": 0}).status_code == 422
        assert client.get("/competitions", params={"limit": 500}).status_code == 422
        assert client.get("/competitions", params={"offset": -1}).status_code == 422


class TestTeams:
    def test_lists_and_searches_teams(self, client: TestClient, league: dict[str, int]) -> None:
        body = client.get("/teams").json()

        assert body["total"] == 6
        assert [t["name"] for t in body["items"]][:2] == ["Arsenal", "Brentford"]
        assert body["items"][0]["short_name"] == "ARS"

        found = client.get("/teams", params={"search": "ars"}).json()
        assert [t["name"] for t in found["items"]] == ["Arsenal"]

    def test_filters_by_competition(self, client: TestClient, league: dict[str, int]) -> None:
        cup_teams = client.get("/teams", params={"competition_id": league["cup"]}).json()

        assert sorted(t["name"] for t in cup_teams["items"]) == ["Arsenal", "Chelsea"]


class TestUpcomingMatches:
    def test_returns_future_matches_in_order(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        body = client.get("/matches/upcoming").json()

        assert body["total"] == 3
        ids = [m["id"] for m in body["items"]]
        assert ids == [league["next_match"], league["later_match"], league["cup_match"]]
        first = body["items"][0]
        assert first["status"] == "scheduled"
        assert first["home_team"]["name"] == "Arsenal"
        assert first["away_team"]["name"] == "Chelsea"
        assert first["competition"]["name"] == "Premier League"
        assert first["home_score"] is None
        assert datetime.fromisoformat(first["kickoff_at"]) > datetime.now(UTC)

    def test_filters_by_competition_team_and_dates(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        by_competition = client.get(
            "/matches/upcoming", params={"competition_id": league["cup"]}
        ).json()
        assert [m["id"] for m in by_competition["items"]] == [league["cup_match"]]

        by_team = client.get("/matches/upcoming", params={"team_id": league["Liverpool"]}).json()
        assert [m["id"] for m in by_team["items"]] == [league["later_match"]]

        soon = (datetime.now(UTC) + timedelta(days=5)).date().isoformat()
        by_date = client.get("/matches/upcoming", params={"to_date": soon}).json()
        assert [m["id"] for m in by_date["items"]] == [league["next_match"]]

    def test_finished_matches_are_not_upcoming(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        ids = [m["id"] for m in client.get("/matches/upcoming").json()["items"]]

        assert league["finished_match"] not in ids


class TestMatchDetail:
    def test_returns_one_match(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get(f"/matches/{league['next_match']}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == league["next_match"]
        assert body["venue"] == "Emirates Stadium"
        assert body["matchday"] == 1
        assert body["season"] == "2026/2027"
        assert body["predictions"] == []

    def test_includes_the_result_of_a_finished_match(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        body = client.get(f"/matches/{league['finished_match']}").json()

        assert body["status"] == "finished"
        assert body["home_score"] is not None and body["away_score"] is not None

    def test_unknown_match_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get("/matches/999999")

        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]

    def test_invalid_id_is_rejected(self, client: TestClient) -> None:
        assert client.get("/matches/0").status_code == 422
        assert client.get("/matches/abc").status_code == 422
