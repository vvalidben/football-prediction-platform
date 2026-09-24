"""Historical backtesting under real-world prediction conditions.

The backtest walks forward through time in steps (weekly by default). At each step
time t, for every model:

1. **Only previous information.** A fresh model is fitted on matches whose results
   were known at t: kickoff + result delay (3 h, as in the feature system) <= t.
   With a rolling window, only matches with kickoff >= t - window are used. Every
   step checks this and raises `BacktestLeakError` if it is ever violated.
2. **Predict.** The model predicts every match kicking off in [t, t + step), from
   the feature snapshots taken at kickoff (Sprint 4: they only contain information
   available before kickoff).
3. **Store.** The predictions are recorded (and, with a store, written to the
   `predictions` tables) before any of their results are used.
4. **Reveal.** A result becomes usable at kickoff + result delay: for training at
   later steps, for calibration, and for the `prediction_results` rows.
5. **Metrics** are computed from the recorded predictions and the revealed results.
6. Move forward to t + step.

**Calibration** (Platt, isotonic) is fitted at each step on the same model's
*earlier backtest predictions* whose results were known at t: genuinely out of
sample, never including the matches being predicted. Until enough such predictions
exist (`min_calibration_matches`), the calibrated variant uses the raw
probabilities, and the record says so. The calibrated 1X2 is applied to the score
grid with `reconcile`, so all markets stay consistent.

Hyperparameters are fixed for the whole backtest (given by the caller). If they were
chosen on data that overlaps the backtest period, that is a look-ahead the
framework cannot see; the report records the hyperparameters used.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np

from app.features.config import FeatureConfig
from app.ml.calibration import CALIBRATION_METHODS, CALIBRATORS
from app.ml.dataset import Dataset, Example
from app.ml.experiment import warm
from app.ml.metrics import evaluate
from app.ml.predictors import Predictor
from app.ml.probability import MatchPrediction, reconcile

WINDOWS = ("expanding", "rolling")


class BacktestLeakError(RuntimeError):
    """A step would have used information not available at its prediction time."""


@dataclass(frozen=True)
class ModelSpec:
    """A model family and fixed hyperparameters, e.g. `poisson:alpha=3`."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        name, _, rest = text.partition(":")
        params: dict[str, Any] = {}
        for item in filter(None, (part.strip() for part in rest.split(","))):
            key, sep, value = item.partition("=")
            if not sep or not key:
                raise ValueError(f"expected key=value in {text!r}")
            params[key.strip()] = _value(value.strip())
        if not name.strip():
            raise ValueError(f"no model name in {text!r}")
        return cls(name.strip(), params)

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


def _value(text: str) -> Any:
    if text.lower() in ("none", "null"):
        return None
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


@dataclass(frozen=True)
class BacktestConfig:
    start: datetime
    end: datetime | None = None
    window: str = "expanding"
    #: Rolling window length (rolling only).
    window_days: int | None = None
    retrain_every: timedelta = timedelta(days=7)
    result_delay: timedelta = field(default_factory=lambda: FeatureConfig().result_delay)
    min_team_history: int = 3
    #: Steps with fewer training matches than this predict nothing (counted as skipped).
    min_training_matches: int = 300
    calibration: tuple[str, ...] = CALIBRATION_METHODS
    min_calibration_matches: int = 300

    def __post_init__(self) -> None:
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}")
        if self.window == "rolling" and not (self.window_days and self.window_days > 0):
            raise ValueError("a rolling window needs window_days > 0")
        if self.window == "expanding" and self.window_days is not None:
            raise ValueError("window_days only applies to a rolling window")
        if self.retrain_every <= timedelta(0):
            raise ValueError("retrain_every must be positive")
        if self.end is not None and self.end <= self.start:
            raise ValueError("the backtest must end after it starts")
        unknown = set(self.calibration) - set(CALIBRATION_METHODS)
        if unknown or not self.calibration:
            raise ValueError(f"calibration must be among {CALIBRATION_METHODS}")

    def describe(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat() if self.end else None,
            "window": self.window,
            "window_days": self.window_days,
            "retrain_every_days": self.retrain_every / timedelta(days=1),
            "result_delay_hours": self.result_delay / timedelta(hours=1),
            "min_team_history": self.min_team_history,
            "min_training_matches": self.min_training_matches,
            "calibration": list(self.calibration),
            "min_calibration_matches": self.min_calibration_matches,
        }


@dataclass(frozen=True)
class BacktestRecord:
    """One prediction, as made at `predicted_at`."""

    model: str  # ModelSpec.label
    variant: str  # "none", "platt" or "isotonic"
    example: Example
    predicted_at: datetime
    #: When the result becomes known (kickoff + result delay).
    revealed_at: datetime
    train_matches: int
    train_start: datetime
    train_end: datetime
    #: Earlier predictions the calibrator was fitted on (0: not calibrated).
    calibrated_on: int
    prediction: MatchPrediction


@dataclass
class BacktestStep:
    time: datetime
    targets: int
    train_matches: int
    skipped: bool


class PredictionSink(Protocol):
    """Where predictions go as they are made (e.g. the database)."""

    def save(self, records: Sequence[BacktestRecord]) -> None: ...

    def reveal(self, up_to: datetime) -> None: ...


@dataclass
class BacktestResult:
    config: BacktestConfig
    models: list[ModelSpec]
    records: list[BacktestRecord]
    steps: list[BacktestStep]
    feature_set_version: str

    @property
    def variants(self) -> list[tuple[str, str]]:
        return [(m.label, v) for m in self.models for v in self.config.calibration]

    @property
    def skipped_matches(self) -> int:
        return sum(s.targets for s in self.steps if s.skipped)

    def select(
        self,
        model: str,
        variant: str,
        keep: Callable[[BacktestRecord], bool] | None = None,
    ) -> tuple[list[MatchPrediction], Dataset]:
        chosen = [
            r
            for r in self.records
            if r.model == model and r.variant == variant and (keep is None or keep(r))
        ]
        return [r.prediction for r in chosen], Dataset(
            tuple(r.example for r in chosen), self.feature_set_version
        )

    def metrics(
        self,
        model: str,
        variant: str,
        keep: Callable[[BacktestRecord], bool] | None = None,
    ) -> dict[str, float]:
        predictions, data = self.select(model, variant, keep)
        return evaluate(predictions, data) if predictions else {}

    def grouped(self, model: str, variant: str, key: str) -> dict[int, dict[str, float]]:
        """Metrics per season (`key="season_id"`) or competition (`"competition_id"`)."""
        groups = sorted({getattr(r.example, key) for r in self.records})
        return {g: self.metrics(model, variant, _in_group(key, g)) for g in groups}


def run_backtest(
    data: Dataset,
    models: Sequence[ModelSpec],
    config: BacktestConfig,
    builders: Mapping[str, Callable[..., Predictor]],
    *,
    sink: PredictionSink | None = None,
) -> BacktestResult:
    unknown = [m.name for m in models if m.name not in builders]
    if unknown:
        raise ValueError(f"unknown models {unknown}; available: {sorted(builders)}")
    if not models:
        raise ValueError("no models to backtest")
    labels = [m.label for m in models]
    if len(set(labels)) != len(labels):
        raise ValueError("each model specification must be unique")
    end = config.end or (data.examples[-1].kickoff_at + timedelta(seconds=1))
    delay = config.result_delay
    window = timedelta(days=config.window_days) if config.window_days else None
    targets_all = data.where(lambda e: config.start <= e.kickoff_at < end)
    history: dict[str, list[tuple[datetime, np.ndarray, int]]] = {label: [] for label in labels}
    records: list[BacktestRecord] = []
    steps: list[BacktestStep] = []

    t = config.start
    while t < end:
        next_t = min(t + config.retrain_every, end)
        if sink is not None:
            sink.reveal(t)
        targets = targets_all.where(_kickoff_between(t, next_t))
        if not len(targets):
            t = next_t
            continue
        known = data.where(_result_known(t, delay))
        if window is not None:
            known = known.where(_kickoff_between(t - window, t))
        train = warm(known, config.min_team_history)
        _check_no_leak(train, targets, t, config, window)
        if len(train) < config.min_training_matches:
            steps.append(BacktestStep(t, len(targets), len(train), skipped=True))
            t = next_t
            continue
        steps.append(BacktestStep(t, len(targets), len(train), skipped=False))
        step_records: list[BacktestRecord] = []
        for spec in models:
            model = builders[spec.name](**spec.params).fit(train)
            raw = model.predict(targets)
            raw_probs = np.array([p.outcomes for p in raw], dtype=float)
            # Only earlier predictions whose results are known at t. A calibrator also
            # needs every outcome (home, draw, away) to have occurred at least once.
            past = [h for h in history[spec.label] if h[0] <= t]
            enough = len(past) >= config.min_calibration_matches and len({h[2] for h in past}) == 3
            for variant in config.calibration:
                predictions, used = raw, 0
                if variant != "none" and enough:
                    calibrator = CALIBRATORS[variant]().fit(
                        np.array([h[1] for h in past]), np.array([h[2] for h in past])
                    )
                    calibrated = calibrator.transform(raw_probs)
                    predictions = [
                        MatchPrediction(reconcile(p.grid, (float(q[0]), float(q[1]), float(q[2]))))
                        for p, q in zip(raw, calibrated, strict=True)
                    ]
                    used = len(past)
                step_records += [
                    BacktestRecord(
                        model=spec.label,
                        variant=variant,
                        example=example,
                        predicted_at=t,
                        revealed_at=example.kickoff_at + delay,
                        train_matches=len(train),
                        train_start=train.examples[0].kickoff_at,
                        train_end=train.examples[-1].kickoff_at,
                        calibrated_on=used,
                        prediction=prediction,
                    )
                    for example, prediction in zip(targets.examples, predictions, strict=True)
                ]
            # Raw predictions join the calibration history; each becomes usable only
            # once its result is revealed (filtered by time above).
            history[spec.label] += [
                (e.kickoff_at + delay, probs, e.outcome)
                for e, probs in zip(targets.examples, raw_probs, strict=True)
            ]
        records += step_records
        if sink is not None:
            sink.save(step_records)
        t = next_t
    if sink is not None:
        sink.reveal(end + delay)
    return BacktestResult(config, list(models), records, steps, data.feature_set_version)


def _in_group(key: str, value: object) -> Callable[[BacktestRecord], bool]:
    return lambda r: bool(getattr(r.example, key) == value)


def _kickoff_between(start: datetime, end: datetime) -> Callable[[Example], bool]:
    return lambda e: start <= e.kickoff_at < end


def _result_known(now: datetime, delay: timedelta) -> Callable[[Example], bool]:
    return lambda e: e.kickoff_at + delay <= now


def _check_no_leak(
    train: Dataset,
    targets: Dataset,
    t: datetime,
    config: BacktestConfig,
    window: timedelta | None,
) -> None:
    if len(train) and train.examples[-1].kickoff_at + config.result_delay > t:
        raise BacktestLeakError(f"training at {t} includes a result not yet known")
    if len(targets) and targets.examples[0].kickoff_at < t:
        raise BacktestLeakError(f"step {t} predicts a match that has already kicked off")
    if window is not None and len(train) and train.examples[0].kickoff_at < t - window:
        raise BacktestLeakError(f"training at {t} reaches outside the rolling window")
