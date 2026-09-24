"""Walk-forward backtesting: only past information, real-world order."""

import dataclasses
from datetime import timedelta

import numpy as np
import pytest

from app.ml.backtest import (
    BacktestConfig,
    BacktestLeakError,
    BacktestRecord,
    BacktestResult,
    ModelSpec,
    _check_no_leak,
    run_backtest,
)
from app.ml.backtest_report import render_backtest_report
from app.ml.dataset import Dataset
from app.ml.engine import ScorePrediction
from app.ml.experiment import default_candidates
from tests.ml.synthetic import START, make_dataset
from tests.ml.test_score_engine import assert_consistent

BUILDERS = {c.name: c.build for c in default_candidates()}
MODELS = [ModelSpec("elo"), ModelSpec("poisson", {"alpha": 3.0})]
DELAY = timedelta(hours=3)


@pytest.fixture(scope="module")
def data() -> Dataset:
    """Four seasons of 56 matches, 6 hours apart; seasons a year apart."""
    return make_dataset(seasons=4, teams=8)


def config(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {
        "start": START + timedelta(days=365 * 2),
        "retrain_every": timedelta(days=2),
        "result_delay": DELAY,
        "min_training_matches": 50,
        "min_calibration_matches": 30,
    }
    values.update(overrides)
    return BacktestConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def result(data: Dataset) -> BacktestResult:
    return run_backtest(data, MODELS, config(), BUILDERS)


@pytest.fixture(scope="module")
def offset_result(data: Dataset) -> BacktestResult:
    """Every step falls 1 hour after a kickoff: at each step one earlier prediction's
    result is not yet known, which the calibrator must not use."""
    return run_backtest(data, MODELS, config(start=config().start + timedelta(hours=1)), BUILDERS)


# Only previous information ----------------------------------------------------------------


def test_every_prediction_uses_only_results_known_at_prediction_time(
    result: BacktestResult,
) -> None:
    assert result.records
    for r in result.records:
        assert r.train_end + DELAY <= r.predicted_at  # newest training result was known
        assert r.predicted_at <= r.example.kickoff_at  # predicted before kickoff
        assert r.revealed_at == r.example.kickoff_at + DELAY


def test_changing_future_results_changes_no_earlier_prediction(
    data: Dataset, result: BacktestResult
) -> None:
    """Poison every result from time T on. Predictions made at or before T, raw and
    calibrated, must be bit-for-bit identical."""
    cut = START + timedelta(days=365 * 3 + 5)
    poisoned = Dataset(
        tuple(
            dataclasses.replace(e, home_goals=e.away_goals, away_goals=e.home_goals + 2)
            if e.kickoff_at >= cut
            else e
            for e in data.examples
        ),
        data.feature_set_version,
    )

    other = run_backtest(poisoned, MODELS, config(), BUILDERS)

    def key(r: BacktestRecord) -> tuple[str, str, int]:
        return r.model, r.variant, r.example.match_id

    later = {key(r): r for r in other.records}
    compared = 0
    for r in result.records:
        if r.predicted_at <= cut:
            np.testing.assert_array_equal(later[key(r)].prediction.grid, r.prediction.grid)
            assert later[key(r)].calibrated_on == r.calibrated_on
            compared += 1
    assert compared > 100
    assert any(
        not np.array_equal(later[key(r)].prediction.grid, r.prediction.grid)
        for r in result.records
        if r.predicted_at > cut + timedelta(days=4)
    )


def test_the_result_delay_keeps_just_finished_matches_out(data: Dataset) -> None:
    """A step 1 hour after a kickoff must not train on that match (known after 3 h)."""
    kickoff = data.examples[-40].kickoff_at
    run = run_backtest(
        data, [ModelSpec("elo")], config(start=kickoff + timedelta(hours=1)), BUILDERS
    )

    first = min(run.records, key=lambda r: r.predicted_at)
    assert first.predicted_at == kickoff + timedelta(hours=1)
    assert first.train_end < kickoff


def test_the_leak_guard_rejects_every_kind_of_look_ahead(data: Dataset) -> None:
    t = data.examples[100].kickoff_at + timedelta(hours=1)  # match 100 kicked off 1 h ago
    cfg = config()
    before = data.where(lambda e: e.kickoff_at < t)
    later = data.where(lambda e: e.kickoff_at >= t)
    known = data.where(lambda e: e.kickoff_at + DELAY <= t)

    with pytest.raises(BacktestLeakError, match="not yet known"):
        _check_no_leak(before, later, t, cfg, None)
    with pytest.raises(BacktestLeakError, match="already kicked off"):
        _check_no_leak(known, before, t, cfg, None)
    with pytest.raises(BacktestLeakError, match="rolling window"):
        _check_no_leak(known, later, t, cfg, timedelta(days=1))
    _check_no_leak(known, later, t, cfg, None)  # the correct split passes


# Walking forward ---------------------------------------------------------------------------


def test_each_match_is_predicted_once_per_model_and_variant(result: BacktestResult) -> None:
    targets = {e for r in result.records for e in [r.example.match_id]}
    for model, variant in result.variants:
        ids = [
            r.example.match_id for r in result.records if (r.model, r.variant) == (model, variant)
        ]
        assert len(ids) == len(set(ids)) == len(targets)
    kickoffs = [
        r.example.kickoff_at for r in result.records if r.variant == "none" and r.model == "elo"
    ]
    assert kickoffs == sorted(kickoffs)  # chronological


def test_expanding_window_grows_and_keeps_its_start(result: BacktestResult) -> None:
    elo = [r for r in result.records if r.model == "elo" and r.variant == "none"]
    sizes = [r.train_matches for r in elo]

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert len({r.train_start for r in elo}) == 1


def test_rolling_window_only_trains_on_recent_matches(data: Dataset) -> None:
    window = timedelta(days=400)
    run = run_backtest(
        data, [ModelSpec("elo")], config(window="rolling", window_days=400), BUILDERS
    )

    assert run.records
    for r in run.records:
        assert r.train_start >= r.predicted_at - window
        assert r.train_end + DELAY <= r.predicted_at
    expanding = run_backtest(data, [ModelSpec("elo")], config(), BUILDERS)
    last_rolling = max(r.train_matches for r in run.records)
    assert last_rolling < max(r.train_matches for r in expanding.records)


def test_steps_without_enough_training_data_are_skipped_and_counted(data: Dataset) -> None:
    run = run_backtest(data, [ModelSpec("elo")], config(min_training_matches=150), BUILDERS)

    assert run.skipped_matches > 0
    predicted = {r.example.match_id for r in run.records}
    total = len(data.where(lambda e: e.kickoff_at >= config().start))
    assert len(predicted) + run.skipped_matches == total


# Calibration within temporal validation -------------------------------------------------


@pytest.mark.parametrize("run", ["result", "offset_result"])
def test_calibrators_use_only_earlier_revealed_predictions(
    run: str, request: pytest.FixtureRequest
) -> None:
    result: BacktestResult = request.getfixturevalue(run)
    if run == "offset_result":  # the case this test exists for really occurs
        raw = [r for r in result.records if r.variant == "none"]
        assert any(p.revealed_at > r.predicted_at > p.predicted_at for r in raw[-20:] for p in raw)
    for spec in MODELS:
        raw = [r for r in result.records if r.model == spec.label and r.variant == "none"]
        for method in ("platt", "isotonic"):
            calibrated = [
                r for r in result.records if r.model == spec.label and r.variant == method
            ]
            for r in calibrated:
                known = [p for p in raw if p.revealed_at <= r.predicted_at]
                enough = (
                    len(known) >= config().min_calibration_matches
                    and len({p.example.outcome for p in known}) == 3
                )
                if r.calibrated_on:
                    assert r.calibrated_on == len(known)
                    assert enough
                else:
                    assert not enough


def test_uncalibrated_steps_fall_back_to_the_raw_prediction(result: BacktestResult) -> None:
    raw = {(r.model, r.example.match_id): r for r in result.records if r.variant == "none"}
    early = [r for r in result.records if r.variant == "platt" and not r.calibrated_on]
    late = [r for r in result.records if r.variant == "platt" and r.calibrated_on]

    assert early and late
    for r in early:
        np.testing.assert_array_equal(
            r.prediction.grid, raw[(r.model, r.example.match_id)].prediction.grid
        )
    assert any(
        not np.allclose(
            r.prediction.outcomes, raw[(r.model, r.example.match_id)].prediction.outcomes
        )
        for r in late
    )


def test_calibrated_predictions_stay_consistent(result: BacktestResult) -> None:
    for r in result.records:
        if r.calibrated_on:
            assert_consistent(ScorePrediction.from_grid(r.prediction.grid))


# Metrics and reports -------------------------------------------------------------------


def test_metrics_include_every_requested_measure(result: BacktestResult) -> None:
    metrics = result.metrics("elo", "none")

    for key in (
        "accuracy",
        "log_loss",
        "brier",
        "ece",
        "total_goals_mae",
        "total_goals_rmse",
        "exact_score_accuracy",
        "top3_score_coverage",
        "top5_score_coverage",
        "over_2_5_ece",
        "btts_accuracy",
    ):
        assert key in metrics


def test_grouped_metrics_partition_the_predictions(result: BacktestResult) -> None:
    overall = result.metrics("elo", "none")
    for key in ("season_id", "competition_id"):
        groups = result.grouped("elo", "none", key)
        assert sum(m["matches"] for m in groups.values()) == overall["matches"]
        pooled_loss = (
            sum(m["log_loss"] * m["matches"] for m in groups.values()) / overall["matches"]
        )
        assert pooled_loss == pytest.approx(overall["log_loss"])


def test_report_has_every_section(result: BacktestResult) -> None:
    report = render_backtest_report(
        result, competitions={1: "Synthetic League"}, seasons={3: "S3", 4: "S4"}
    )

    for heading in (
        "## Protocol",
        "## Overall",
        "## Calibration",
        "### Reliability (raw probabilities)",
        "## By season",
        "## By competition",
        "## By prediction type",
    ):
        assert heading in report
    assert "Synthetic League" in report and "| S3 |" in report
    assert "poisson(alpha=3.0) + isotonic" in report
    assert "Over/under 2.5" in report and "Both teams to score" in report


# Configuration -----------------------------------------------------------------------------


def test_model_specs_parse_fixed_hyperparameters() -> None:
    spec = ModelSpec.parse("lightgbm:num_leaves=3, n_estimators=250,learning_rate=0.02")

    assert spec.name == "lightgbm"
    assert spec.params == {"num_leaves": 3, "n_estimators": 250, "learning_rate": 0.02}
    assert ModelSpec.parse("random_forest:max_depth=None").params == {"max_depth": None}
    assert ModelSpec.parse("elo").label == "elo"
    with pytest.raises(ValueError, match="key=value"):
        ModelSpec.parse("poisson:alpha")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"window": "sliding"}, "window must be"),
        ({"window": "rolling"}, "window_days > 0"),
        ({"window_days": 30}, "only applies to a rolling"),
        ({"retrain_every": timedelta(0)}, "positive"),
        ({"end": START}, "end after"),
        ({"calibration": ("beta",)}, "calibration must be"),
    ],
)
def test_invalid_configurations_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config(**overrides)


def test_unknown_models_are_rejected(data: Dataset) -> None:
    with pytest.raises(ValueError, match="unknown models"):
        run_backtest(data, [ModelSpec("catboost")], config(), BUILDERS)


def test_calibration_waits_until_every_outcome_has_occurred(data: Dataset) -> None:
    """A history without draws cannot calibrate the draw probability: raw is kept."""
    no_draws = Dataset(
        tuple(
            dataclasses.replace(e, home_goals=e.home_goals + 1)
            if e.home_goals == e.away_goals
            else e
            for e in data.examples
        ),
        data.feature_set_version,
    )

    run = run_backtest(no_draws, [ModelSpec("elo")], config(), BUILDERS)

    assert run.records
    assert not any(r.calibrated_on for r in run.records)
