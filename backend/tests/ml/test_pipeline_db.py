"""End to end on PostgreSQL: matches → feature snapshots → dataset → trained, tracked,
registered models."""

import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.features.__main__ import main as features_main
from app.features.config import FeatureConfig
from app.ml.__main__ import main as ml_main
from app.ml.dataset import load_dataset
from app.models import Competition, Match, ModelVersion, Season, Team
from app.models.enums import CompetitionType, MatchStatus

pytestmark = pytest.mark.integration


@pytest.fixture
def league(session_factory: sessionmaker[Session]) -> sessionmaker[Session]:
    """Three seasons, six teams, double round robin, results driven by team strength."""
    return _build_league(session_factory, (2021, 2022, 2023))


@pytest.fixture
def long_league(session_factory: sessionmaker[Session]) -> sessionmaker[Session]:
    """Five seasons: three training seasons leave room for ensemble folds."""
    return _build_league(session_factory, (2019, 2020, 2021, 2022, 2023))


def _build_league(
    session_factory: sessionmaker[Session], years: tuple[int, ...]
) -> sessionmaker[Session]:
    rng = random.Random(3)
    with session_factory() as session, session.begin():
        competition = Competition(name="Synthetic League", type=CompetitionType.LEAGUE)
        teams = [Team(name=f"Team {n}") for n in range(6)]
        strength = {id(t): rng.gauss(0, 0.4) for t in teams}
        session.add_all([competition, *teams])
        for year in years:
            season = Season(
                competition=competition,
                name=f"{year}/{year + 1}",
                start_date=date(year, 8, 1),
                end_date=date(year + 1, 5, 31),
            )
            kickoff = datetime(year, 8, 7, 15, tzinfo=UTC)
            for home in teams:
                for away in teams:
                    if home is away:
                        continue
                    gap = strength[id(home)] - strength[id(away)]
                    session.add(
                        Match(
                            competition=competition,
                            season=season,
                            home_team=home,
                            away_team=away,
                            kickoff_at=kickoff,
                            status=MatchStatus.FINISHED,
                            home_score=_poisson(rng, 1.5 * 2.718**gap),
                            away_score=_poisson(rng, 1.1 * 2.718**-gap),
                        )
                    )
                    kickoff += timedelta(days=4)
    return session_factory


def _poisson(rng: random.Random, rate: float) -> int:
    count, total = 0, rng.expovariate(rate)
    while total < 1.0:
        count += 1
        total += rng.expovariate(rate)
    return count


def test_dataset_is_built_from_snapshots_at_kickoff(league: sessionmaker[Session]) -> None:
    assert features_main(["generate"], session_factory=league) == 0
    features_main(["generate", "--lead-hours", "24"], session_factory=league)  # ignored

    with league() as session:
        data = load_dataset(session, FeatureConfig().version())
        matches = session.scalars(select(Match).order_by(Match.kickoff_at)).all()

    assert len(data) == len(matches) == 90
    assert [e.match_id for e in data.examples] == [m.id for m in matches]
    assert (data.examples[0].home_goals, data.examples[0].away_goals) == (
        matches[0].home_score,
        matches[0].away_score,
    )
    assert "elo_diff" in data.examples[-1].features


def test_registered_models_record_their_test_results_per_competition(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    ml_main(
        ["train", "--tracking-uri", uri, "--experiment", "db-test", "--register"],
        session_factory=league,
    )

    with league() as session:
        row = session.scalars(select(ModelVersion)).one()
        competition = session.scalars(select(Competition)).one()
    measured = row.training_metrics["test_by_competition"][str(competition.id)]
    assert measured["matches"] == row.training_metrics["test"]["matches"]
    assert measured["accuracy"] == pytest.approx(row.training_metrics["test"]["accuracy"])


def test_an_existing_model_can_be_measured_only_on_matches_after_its_training(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    ml_main(
        ["train", "--tracking-uri", uri, "--experiment", "db-test", "--register"],
        session_factory=league,
    )
    with league() as session, session.begin():
        row = session.scalars(select(ModelVersion)).one()
        name, version, trained_to = row.name, row.version, row.training_data_end
        before = dict(row.training_metrics)
        before.pop("test_by_competition")
        row.training_metrics = before
    assert trained_to is not None

    too_early = ml_main(
        [
            "registry",
            "--tracking-uri",
            uri,
            "measure",
            name,
            version,
            "--test-start",
            trained_to.isoformat(),
            "--test-end",
            "2030-01-01",
        ],
        session_factory=league,
    )
    with league() as session:
        untouched = session.scalars(select(ModelVersion)).one()
    assert too_early == 2
    assert "test_by_competition" not in untouched.training_metrics

    after = (trained_to + timedelta(days=1)).isoformat()
    code = ml_main(
        [
            "registry",
            "--tracking-uri",
            uri,
            "measure",
            name,
            version,
            "--test-start",
            after,
            "--test-end",
            "2030-01-01",
        ],
        session_factory=league,
    )

    with league() as session:
        measured = session.scalars(select(ModelVersion)).one().training_metrics
    assert code == 0
    assert measured["test_by_competition"]
    assert measured["validation"] == before["validation"]


def test_train_command_tracks_and_registers(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)
    capsys.readouterr()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    code = ml_main(
        ["train", "--tracking-uri", uri, "--experiment", "db-test", "--register"],
        session_factory=league,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "seasons: 1 train, 1 validation, 1 test" in output
    for name in (
        "historical_baseline",
        "elo",
        "logistic_regression",
        "poisson",
        "random_forest",
        "xgboost",
        "lightgbm",
    ):
        assert name in output
    assert "selected on validation" in output
    with league() as session:
        registered = session.scalars(select(ModelVersion)).one()
    assert registered.feature_set_version == FeatureConfig().version()
    assert registered.is_active is False
    assert registered.artifact_uri and registered.artifact_uri.endswith("/model")
    assert {"validation", "test", "mlflow_run_id"} <= set(registered.training_metrics)
    # The fitted period, not the split boundary: the warm-up filter drops the first
    # matches, while teams have fewer than 3 rated matches.
    assert registered.training_data_start is not None
    assert registered.training_data_start > date(2021, 8, 7)
    with league() as session:
        last_validation = session.scalars(
            select(Match.kickoff_at)
            .where(Match.kickoff_at < datetime(2023, 7, 1, tzinfo=UTC))
            .order_by(Match.kickoff_at.desc())
        ).first()
    assert last_validation is not None
    assert registered.training_data_end == last_validation.date()


def test_train_without_snapshots_explains_what_to_do(
    league: sessionmaker[Session], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = ml_main(
        ["train", "--tracking-uri", f"sqlite:///{(tmp_path / 'm.db').as_posix()}"],
        session_factory=league,
    )

    assert code == 2
    assert "app.features generate" in capsys.readouterr().err


def test_registering_never_replaces_or_deactivates_a_production_model(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)
    names = ("elo", "xgboost")
    with league() as session, session.begin():
        for name in names:
            session.add(
                ModelVersion(
                    name=name,
                    version="production",
                    algorithm="EloModel",
                    dataset_version="ds-production",
                    feature_set_version=FeatureConfig().version(),
                    hyperparameters={"kept": True},
                    training_metrics={},
                    trained_at=datetime(2024, 1, 1, tzinfo=UTC),
                    artifact_uri="file:///production/model",
                    is_active=True,
                )
            )
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    for _ in range(2):
        code = ml_main(
            ["train", "--tracking-uri", uri, "--models", ",".join(names), "--register"],
            session_factory=league,
        )
        assert code == 0

    with league() as session:
        rows = session.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()
    production = [r for r in rows if r.version == "production"]
    new = [r for r in rows if r.version != "production"]
    assert len(production) == 2 and all(r.is_active for r in production)
    assert all(r.hyperparameters == {"kept": True} for r in production)
    assert all(r.artifact_uri == "file:///production/model" for r in production)
    assert len(new) == 2 and not any(r.is_active for r in new)
    assert len({(r.name, r.version) for r in new}) == 2  # two separate versions
    assert all(r.hyperparameters["model_type"] for r in new)


def test_model_subset_cross_validation_and_report(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)
    capsys.readouterr()
    report = tmp_path / "reports" / "comparison.md"

    code = ml_main(
        [
            "train",
            "--tracking-uri",
            f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "--models",
            "elo,random_forest,lightgbm",
            "--cv-folds",
            "3",
            "--report",
            str(report),
        ],
        session_factory=league,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "poisson" not in output and "lightgbm" in output
    text = report.read_text(encoding="utf-8")
    assert "## Test metrics (reported only)" in text
    for name in ("elo", "random_forest", "lightgbm"):
        assert name in text
    # Three seasons: one training season, which cannot hold out a fold (nothing precedes
    # it), so tuning falls back to the validation period.
    assert "validation_log_loss" in text


def test_unknown_models_are_rejected(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)

    code = ml_main(
        [
            "train",
            "--tracking-uri",
            f"sqlite:///{(tmp_path / 'm.db').as_posix()}",
            "--models",
            "elo,catboost",
        ],
        session_factory=league,
    )

    assert code == 2
    assert "catboost" in capsys.readouterr().err


def test_ensembles_are_evaluated_reported_and_registered_as_their_own_version(
    long_league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=long_league)
    capsys.readouterr()
    report = tmp_path / "report.md"

    code = ml_main(
        [
            "train",
            "--tracking-uri",
            f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            "--models",
            "elo,logistic_regression,poisson,lightgbm",
            "--cv-folds",
            "3",
            "--ensemble",
            "--ensemble-members",
            "elo,logistic_regression,poisson,lightgbm",
            "--report",
            str(report),
            "--register",
        ],
        session_factory=long_league,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "forward-chained validation over 2 blocks" in output
    assert "recommended:" in output
    assert "## Ensembles" in report.read_text(encoding="utf-8")
    with long_league() as session:
        rows = session.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()
    assert len(rows) == 2
    individual, ensemble = rows
    assert not individual.name.startswith("ensemble_")
    assert ensemble.name.startswith("ensemble_")
    assert ensemble.is_active is False
    assert ensemble.hyperparameters["model_type"] == "EnsembleModel"
    assert set(ensemble.hyperparameters["members"]) == {
        "elo",
        "logistic_regression",
        "poisson",
        "lightgbm",
    }
    assert ensemble.training_metrics["selection_metric"] == "forward_validation_log_loss"
    assert ensemble.artifact_uri and ensemble.artifact_uri.endswith("/model")


def test_ensemble_needs_cross_validation_folds(
    league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)

    code = ml_main(
        ["train", "--tracking-uri", f"sqlite:///{(tmp_path / 'm.db').as_posix()}", "--ensemble"],
        session_factory=league,
    )

    assert code == 2
    assert "--cv-folds >= 2" in capsys.readouterr().err


def test_backtest_stores_predictions_before_revealing_results(
    long_league: sessionmaker[Session],
) -> None:
    from app.ml.backtest import BacktestConfig, ModelSpec, run_backtest
    from app.ml.backtest_store import BacktestStore, metrics_from_store
    from app.ml.experiment import default_candidates
    from app.models import (
        Prediction,
        PredictionOverUnderProbability,
        PredictionResult,
        PredictionScoreProbability,
    )

    features_main(["generate"], session_factory=long_league)
    with long_league() as session:
        data = load_dataset(session, FeatureConfig().version())
    config = BacktestConfig(
        start=datetime(2021, 7, 1, tzinfo=UTC),
        retrain_every=timedelta(days=14),
        min_training_matches=30,
        min_calibration_matches=20,
    )
    models = [ModelSpec("elo")]
    store = BacktestStore(
        long_league,
        models=models,
        config=config,
        data=data,
        algorithms={"elo": "EloModel"},
    )

    result = run_backtest(
        data, models, config, {c.name: c.build for c in default_candidates()}, sink=store
    )
    store.finish(result)

    assert len(store.versions) == 3  # elo raw, platt, isotonic
    with long_league() as session:
        versions = session.scalars(
            select(ModelVersion).where(ModelVersion.id.in_(store.versions.values()))
        ).all()
        assert {v.name for v in versions} == {
            "backtest_elo_none",
            "backtest_elo_platt",
            "backtest_elo_isotonic",
        }
        assert not any(v.is_active for v in versions)
        rows = session.execute(
            select(Prediction, PredictionResult, Match)
            .join(PredictionResult, PredictionResult.prediction_id == Prediction.id)
            .join(Match, Match.id == Prediction.match_id)
        ).all()
        assert len(rows) == len(result.records) > 0
        for prediction, evaluation, match in rows:
            assert prediction.predicted_at <= match.kickoff_at  # stored before kickoff
            assert evaluation.evaluated_at == match.kickoff_at + timedelta(hours=3)
            assert prediction.feature_snapshot_id is not None
        prediction_ids = [p.id for p, _, _ in rows]
        cells = session.scalar(
            select(func.count()).where(PredictionScoreProbability.prediction_id.in_(prediction_ids))
        )
        lines = session.scalar(
            select(func.count()).where(
                PredictionOverUnderProbability.prediction_id.in_(prediction_ids)
            )
        )
        assert cells == 121 * len(rows) and lines == 4 * len(rows)
        for (label, variant), version_id in store.versions.items():
            stored = metrics_from_store(session, version_id)
            in_memory = result.metrics(label, variant)
            # Stored values have 6 decimals: probabilities agree to ~1e-5. Expected goals
            # weight the rounded tail cells by up to 10 goals (~1e-3). ECE is
            # discontinuous: a probability on a bin edge can change bins when rounded.
            for key, tolerance in (
                ("matches", 0),
                ("accuracy", 1e-9),
                ("log_loss", 1e-4),
                ("brier", 1e-4),
                ("ece", 5e-3),
                ("total_goals_mae", 2e-3),
            ):
                assert stored[key] == pytest.approx(in_memory[key], abs=tolerance), key
        raw_version = session.get(ModelVersion, store.versions[("elo", "none")])
        assert raw_version is not None
        assert raw_version.training_metrics["predictions"] == len(result.records) // 3


def test_backtest_command_reports_and_stores(
    long_league: sessionmaker[Session], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=long_league)
    capsys.readouterr()
    report = tmp_path / "backtest.md"

    code = ml_main(
        [
            "backtest",
            "--start",
            "2021-07-01",
            "--model",
            "elo",
            "--model",
            "poisson:alpha=3",
            "--window",
            "rolling",
            "--window-days",
            "800",
            "--retrain-every-days",
            "14",
            "--min-training-matches",
            "30",
            "--min-calibration-matches",
            "20",
            "--report",
            str(report),
        ],
        session_factory=long_league,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "rolling window" in output and "poisson(alpha=3) + isotonic" in output
    text = report.read_text(encoding="utf-8")
    for heading in ("## Overall", "## By season", "## By competition", "## By prediction type"):
        assert heading in text
    assert "Synthetic League" in text and "2022/2023" in text
    with long_league() as session:
        assert session.scalar(select(func.count()).select_from(ModelVersion)) == 6


def test_backtest_rejects_bad_options(
    league: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    features_main(["generate"], session_factory=league)

    code = ml_main(
        ["backtest", "--start", "2022-07-01", "--window", "rolling"], session_factory=league
    )

    assert code == 2
    assert "window_days" in capsys.readouterr().err
