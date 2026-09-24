"""Calibration measurement and calibrators."""

import numpy as np
import pytest

from app.ml.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    classwise_ece,
    expected_calibration_error,
    reliability,
)


def simulate(n: int, seed: int, sharpen: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """True probabilities, outcomes drawn from them, and the probabilities a model
    would report: sharpened (overconfident) when `sharpen` > 1."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 0.8, size=(n, 3)) + np.array([0.3, 0.0, -0.2])
    true = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    outcomes = np.array([rng.choice(3, p=p) for p in true])
    reported = true**sharpen
    reported /= reported.sum(axis=1, keepdims=True)
    return reported, outcomes


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(len(outcomes)), outcomes])))


class TestMeasurement:
    def test_reliability_bins_cover_every_prediction(self) -> None:
        probs, outcomes = simulate(2000, 0)

        rows = reliability(probs[:, 0], outcomes == 0)

        assert sum(r["count"] for r in rows) == 2000
        for r in rows:
            assert r["low"] <= r["mean_predicted"] <= r["high"]
            assert 0 <= r["observed"] <= 1

    def test_a_calibrated_model_has_small_ece(self) -> None:
        probs, outcomes = simulate(20000, 1)

        assert classwise_ece(probs, outcomes) < 0.015

    def test_an_overconfident_model_has_large_ece(self) -> None:
        calibrated, outcomes = simulate(20000, 1)
        overconfident, _ = simulate(20000, 1, sharpen=2.5)

        assert classwise_ece(overconfident, outcomes) > 3 * classwise_ece(calibrated, outcomes)

    def test_ece_of_a_known_case(self) -> None:
        # All predictions 0.8 in one bin; the event happens half the time: gap 0.3.
        assert expected_calibration_error(np.full(10, 0.8), np.arange(10) % 2 == 0) == (
            pytest.approx(0.3)
        )

    def test_ece_needs_predictions(self) -> None:
        with pytest.raises(ValueError, match="no predictions"):
            expected_calibration_error(np.array([]), np.array([], dtype=bool))


@pytest.mark.parametrize("calibrator_class", [PlattCalibrator, IsotonicCalibrator])
class TestCalibrators:
    def test_fixes_overconfidence_on_new_data(self, calibrator_class: type) -> None:
        """Fitted on one sample, applied to a later, independent one."""
        history, history_outcomes = simulate(4000, 2, sharpen=2.0)
        future, future_outcomes = simulate(4000, 3, sharpen=2.0)

        calibrated = calibrator_class().fit(history, history_outcomes).transform(future)

        assert classwise_ece(calibrated, future_outcomes) < 0.5 * classwise_ece(
            future, future_outcomes
        )
        assert log_loss(calibrated, future_outcomes) < log_loss(future, future_outcomes)

    def test_outputs_are_probability_vectors(self, calibrator_class: type) -> None:
        probs, outcomes = simulate(1000, 4, sharpen=3.0)

        calibrated = calibrator_class().fit(probs, outcomes).transform(probs)

        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        assert (calibrated > 0).all() and (calibrated < 1).all()

    def test_leaves_a_calibrated_model_nearly_unchanged(self, calibrator_class: type) -> None:
        history, history_outcomes = simulate(8000, 5)
        future, future_outcomes = simulate(4000, 6)

        calibrated = calibrator_class().fit(history, history_outcomes).transform(future)

        assert log_loss(calibrated, future_outcomes) < log_loss(future, future_outcomes) + 0.01

    def test_rejects_bad_input_and_use_before_fitting(self, calibrator_class: type) -> None:
        probs, outcomes = simulate(100, 7)
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator_class().transform(probs)
        with pytest.raises(ValueError, match="one outcome per match"):
            calibrator_class().fit(probs, outcomes[:-1])
        with pytest.raises(ValueError, match="draws"):
            calibrator_class().fit(probs, np.where(outcomes == 1, 0, outcomes))


def test_platt_records_its_parameters() -> None:
    probs, outcomes = simulate(2000, 8, sharpen=2.0)

    params = PlattCalibrator().fit(probs, outcomes).params()

    assert params["fitted_on"] == 2000
    assert len(params["slopes"]) == 3
    assert all(0 < s < 1 for s in params["slopes"])  # overconfidence: slopes below 1
