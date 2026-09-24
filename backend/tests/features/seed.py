"""A small league written directly to the database, for feature tests."""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session, sessionmaker

from app.features.builder import FeatureBuilder, Features
from app.features.config import FeatureConfig
from app.features.timeline import load_targets
from app.models import (
    Bookmaker,
    Competition,
    DataSource,
    Injury,
    Match,
    Odds,
    Player,
    Season,
    Team,
    TeamMatchStatistics,
)
from app.models.enums import (
    AbsenceType,
    AvailabilityStatus,
    CompetitionType,
    MatchStatus,
    OddsMarket,
)

KICKOFF = datetime(2024, 3, 2, 15, 0, tzinfo=UTC)  # the target match


@dataclass
class League:
    factory: sessionmaker[Session]
    source_id: int = 0
    competition_id: int = 0
    season_id: int = 0
    teams: dict[str, int] = field(default_factory=dict)
    target_id: int = 0

    def add_match(
        self,
        home: str,
        away: str,
        kickoff: datetime,
        score: tuple[int, int] | None,
        *,
        xg: tuple[float, float] | None = None,
        season_id: int | None = None,
    ) -> int:
        with self.factory() as session, session.begin():
            match = Match(
                competition_id=self.competition_id,
                season_id=season_id or self.season_id,
                home_team_id=self.teams[home],
                away_team_id=self.teams[away],
                kickoff_at=kickoff,
                status=MatchStatus.FINISHED if score else MatchStatus.SCHEDULED,
                home_score=score[0] if score else None,
                away_score=score[1] if score else None,
            )
            session.add(match)
            session.flush()
            if xg is not None:
                self._add_xg(session, match.id, home, away, xg)
            return match.id

    def add_xg(self, match_id: int, home: str, away: str, xg: tuple[float, float]) -> None:
        with self.factory() as session, session.begin():
            self._add_xg(session, match_id, home, away, xg)

    def _add_xg(
        self, session: Session, match_id: int, home: str, away: str, xg: tuple[float, float]
    ) -> None:
        for team, value in ((home, xg[0]), (away, xg[1])):
            session.add(
                TeamMatchStatistics(
                    match_id=match_id,
                    team_id=self.teams[team],
                    data_source_id=self.source_id,
                    expected_goals=Decimal(str(value)),
                    shots=30,
                )
            )

    def finish_target(self, score: tuple[int, int]) -> None:
        """Simulate the target match being played and its result stored."""
        with self.factory() as session, session.begin():
            match = session.get_one(Match, self.target_id)
            match.status = MatchStatus.FINISHED
            match.home_score, match.away_score = score

    def add_odds(
        self, recorded_at: datetime, home: str, draw: str, away: str, book: str = "Book"
    ) -> None:
        with self.factory() as session, session.begin():
            bookmaker = session.query(Bookmaker).filter_by(name=book).one_or_none()
            if bookmaker is None:
                bookmaker = Bookmaker(name=book)
                session.add(bookmaker)
                session.flush()
            for selection, price in (("home", home), ("draw", draw), ("away", away)):
                session.add(
                    Odds(
                        match_id=self.target_id,
                        bookmaker_id=bookmaker.id,
                        data_source_id=self.source_id,
                        market=OddsMarket.MATCH_WINNER,
                        selection=selection,
                        price=Decimal(price),
                        recorded_at=recorded_at,
                    )
                )

    def add_injury(
        self,
        team: str,
        *,
        start: date | None,
        status: AvailabilityStatus = AvailabilityStatus.OUT,
        for_target: bool = False,
    ) -> None:
        with self.factory() as session, session.begin():
            player = Player(name=f"Player {team} {start}", current_team_id=self.teams[team])
            session.add(player)
            session.flush()
            session.add(
                Injury(
                    data_source_id=self.source_id,
                    player_id=player.id,
                    team_id=self.teams[team],
                    match_id=self.target_id if for_target else None,
                    absence_type=AbsenceType.INJURY,
                    status=status,
                    start_date=start,
                )
            )

    def features(
        self, config: FeatureConfig | None = None, lead: timedelta = timedelta(0)
    ) -> Features:
        builder = FeatureBuilder(self.factory, config)
        with self.factory() as session:
            targets = load_targets(session, match_ids=[self.target_id], lead_time=lead)
        (row,) = builder.build(targets)
        return row.values


def build_league(factory: sessionmaker[Session]) -> League:
    """Four teams; Arsenal (home) v Brighton (away) at KICKOFF is the target.

    Before the target: six rounds of results (one per week), some with xG.
    """
    league = League(factory)
    with factory() as session, session.begin():
        source = DataSource(code="feature_test", name="Feature test")
        competition = Competition(
            name="Test League", type=CompetitionType.LEAGUE, country="England"
        )
        season = Season(
            competition=competition,
            name="2023/2024",
            start_date=date(2023, 8, 1),
            end_date=date(2024, 5, 31),
        )
        teams = {
            name: Team(name=name, country="England")
            for name in ("Arsenal", "Brighton", "Chelsea", "Derby")
        }
        session.add_all([source, competition, season, *teams.values()])
        session.flush()
        league.source_id, league.competition_id, league.season_id = (
            source.id,
            competition.id,
            season.id,
        )
        league.teams = {name: team.id for name, team in teams.items()}

    week = timedelta(days=7)
    start = KICKOFF - 6 * week
    fixtures = [
        ("Arsenal", "Chelsea", (2, 0), (1.8, 0.6)),
        ("Brighton", "Derby", (1, 1), None),
        ("Chelsea", "Brighton", (0, 3), (0.9, 2.1)),
        ("Derby", "Arsenal", (1, 2), None),
        ("Arsenal", "Brighton", (1, 1), (1.2, 1.0)),
        ("Chelsea", "Derby", (2, 2), None),
        ("Brighton", "Chelsea", (2, 1), None),
        ("Arsenal", "Derby", (3, 0), (2.4, 0.3)),
        ("Derby", "Brighton", (0, 1), None),
        ("Chelsea", "Arsenal", (1, 1), None),
        ("Brighton", "Arsenal", (0, 2), (0.7, 1.9)),
        ("Derby", "Chelsea", (1, 0), None),
    ]
    for index, (home, away, score, xg) in enumerate(fixtures):
        league.add_match(home, away, start + (index // 2) * week, score, xg=xg)
    league.target_id = league.add_match("Arsenal", "Brighton", KICKOFF, None)
    return league
