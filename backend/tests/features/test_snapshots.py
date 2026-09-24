"""Feature snapshots: versioned, immutable, reproducible; and the CLI."""

import csv
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.features.__main__ import main
from app.features.builder import FeatureBuilder
from app.features.config import FeatureConfig
from app.features.store import FeatureStore, SaveResult, reproduce
from app.features.timeline import load_targets
from app.models import FeatureSnapshot, Match
from tests.features.seed import League, build_league

pytestmark = pytest.mark.integration


@pytest.fixture
def league(session_factory: sessionmaker[Session]) -> League:
    return build_league(session_factory)


def generate(factory: sessionmaker[Session], config: FeatureConfig | None = None) -> SaveResult:
    builder = FeatureBuilder(factory, config)
    with factory() as session:
        targets = load_targets(session)
    return FeatureStore(factory).save(builder.build(targets))


def snapshots(factory: sessionmaker[Session]) -> list[FeatureSnapshot]:
    with factory() as session:
        return list(session.scalars(select(FeatureSnapshot).order_by(FeatureSnapshot.id)))


def test_snapshots_are_stored_with_version_and_cutoff(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    result = generate(session_factory)

    assert (result.created, result.unchanged, result.drifted) == (13, 0, [])
    stored = snapshots(session_factory)
    assert {s.feature_set_version for s in stored} == {FeatureConfig().version()}
    with session_factory() as session:
        kickoffs = {m.id: m.kickoff_at for m in session.scalars(select(Match))}
    assert all(s.data_cutoff_at == kickoffs[s.match_id] for s in stored)


def test_regenerating_is_idempotent(league: League, session_factory: sessionmaker[Session]) -> None:
    generate(session_factory)

    result = generate(session_factory)

    assert (result.created, result.unchanged) == (0, 13)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FeatureSnapshot)) == 13


def test_stored_snapshots_are_reproducible(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    generate(session_factory)
    builder = FeatureBuilder(session_factory)

    for snapshot in snapshots(session_factory):
        assert reproduce(builder, snapshot) == snapshot.features


def test_corrected_history_is_reported_as_drift_and_snapshots_are_kept(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    generate(session_factory)
    target_snapshot = next(s for s in snapshots(session_factory) if s.match_id == league.target_id)
    # A past result is corrected after the snapshot was taken.
    with session_factory() as session, session.begin():
        first = session.scalars(select(Match).order_by(Match.kickoff_at)).first()
        assert first is not None
        first.home_score = 5

    result = generate(session_factory)

    assert league.target_id in result.drifted
    kept = next(s for s in snapshots(session_factory) if s.match_id == league.target_id)
    assert kept.features == target_snapshot.features  # immutable


def test_a_new_definition_gets_a_new_version_alongside_the_old(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    generate(session_factory)
    generate(session_factory, FeatureConfig(elo_k=30))

    versions = {s.feature_set_version for s in snapshots(session_factory)}
    assert versions == {FeatureConfig().version(), FeatureConfig(elo_k=30).version()}


def test_reproducing_with_other_definitions_is_refused(
    league: League, session_factory: sessionmaker[Session]
) -> None:
    generate(session_factory)
    other = FeatureBuilder(session_factory, FeatureConfig(elo_k=30))

    with pytest.raises(ValueError, match="version"):
        reproduce(other, snapshots(session_factory)[0])


class TestCommands:
    def test_generate_show_and_export(
        self,
        league: League,
        session_factory: sessionmaker[Session],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert main(["generate"], session_factory=session_factory) == 0
        assert "created 13" in capsys.readouterr().out

        assert (
            main(["show", "--match-id", str(league.target_id)], session_factory=session_factory)
            == 0
        )
        shown = capsys.readouterr().out
        assert "Arsenal v Brighton" in shown
        for group in (
            "[form]",
            "[venue]",
            "[strength]",
            "[context]",
            "[h2h]",
            "[players]",
            "[odds]",
        ):
            assert group in shown
        assert "home_form3_goals_for" in shown

        out = tmp_path / "features.csv"
        assert main(["export", "--out", str(out)], session_factory=session_factory) == 0
        with out.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 13
        assert rows[-1]["home_team"] == "Arsenal"
        assert rows[-1]["target_home_goals"] == ""  # target not played yet

    def test_lead_time_creates_separate_snapshots(
        self, league: League, session_factory: sessionmaker[Session]
    ) -> None:
        main(["generate"], session_factory=session_factory)
        main(["generate", "--lead-hours", "24"], session_factory=session_factory)

        cutoffs = {(s.match_id, s.data_cutoff_at) for s in snapshots(session_factory)}
        assert len(cutoffs) == 26

    def test_version_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["version"]) == 0
        assert FeatureConfig().version() in capsys.readouterr().out
