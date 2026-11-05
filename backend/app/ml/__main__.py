"""Train, compare and evaluate the prediction models on stored feature snapshots.

    python -m app.ml train                        # season split: last=test, previous=validation
    python -m app.ml train --validation-start 2023-08-01 --test-start 2024-08-01
    python -m app.ml train --cv-folds 3 --report reports/model-comparison.md
    python -m app.ml train --models elo,xgboost --with-odds --register
    python -m app.ml train --cv-folds 4 --ensemble --report report.md   # ensembles too
    python -m app.ml backtest --start 2021-07-01 --model elo --model poisson:alpha=3
    python -m app.ml backtest --start 2021-07-01 --window rolling --window-days 1095
    python -m app.ml registry list
    python -m app.ml registry activate elo 20260922141211-20c9f0ee --reason "Sprint 8"
    python -m app.ml registry rollback elo

Prerequisite: `python -m app.features generate` (snapshots at kickoff).
Results are tracked in MLflow; see docs/models.md.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory
from app.features.config import FeatureConfig
from app.ml.backtest import WINDOWS, BacktestConfig, ModelSpec, run_backtest
from app.ml.backtest_report import render_backtest_report
from app.ml.backtest_store import BacktestStore
from app.ml.calibration import CALIBRATION_METHODS
from app.ml.dataset import Dataset, load_dataset
from app.ml.ensemble import DEFAULT_MEMBERS, EnsembleResult, run_ensembles
from app.ml.experiment import ExperimentResult, default_candidates, run_experiment
from app.ml.registry import (
    load_model_version,
    measure_by_competition,
    register_ensemble,
    register_selected,
)
from app.ml.report import comparisons, forward_comparisons, render_report
from app.ml.splits import ChronologicalSplit, SplitError, split_by_dates, split_by_seasons
from app.ml.tracking import Tracker
from app.models import Competition, Season
from app.registry.mlflow_sync import sync_active
from app.registry.service import ModelRegistry, RegistryError, activation_record, metadata

TABLE_METRICS = (
    ("log_loss", "logloss"),
    ("brier", "brier"),
    ("rps", "rps"),
    ("accuracy", "acc"),
    ("total_goals_mae", "goalMAE"),
    ("total_goals_rmse", "goalRMSE"),
    ("exact_score_accuracy", "exact"),
    ("top3_score_coverage", "top3"),
    ("top5_score_coverage", "top5"),
    ("actual_score_log_prob", "logP(score)"),
)


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ml",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="tune, select and evaluate the models")
    train.add_argument("--feature-version", help="feature-set version (default: current code)")
    train.add_argument("--validation-start", type=_date, help="YYYY-MM-DD; with --test-start")
    train.add_argument("--test-start", type=_date, help="YYYY-MM-DD")
    train.add_argument("--test-end", type=_date, help="YYYY-MM-DD (exclusive)")
    train.add_argument("--with-odds", action="store_true", help="also try a model using odds")
    train.add_argument(
        "--models", help="comma-separated candidate names (default: all; see docs/models.md)"
    )
    train.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="tuning folds: 1 = validation only; n > 1 adds the last n-1 training seasons "
        "as rolling-origin folds",
    )
    train.add_argument("--report", type=Path, help="write a Markdown comparison report here")
    train.add_argument(
        "--ensemble",
        action="store_true",
        help="also build and evaluate ensembles (needs --cv-folds >= 2)",
    )
    train.add_argument(
        "--ensemble-members",
        default=",".join(DEFAULT_MEMBERS),
        help="comma-separated members (default: %(default)s)",
    )
    train.add_argument("--min-team-history", type=int, default=3)
    train.add_argument("--experiment", default="baseline-models")
    train.add_argument("--tracking-uri", help="MLflow tracking URI (default: backend/mlruns)")
    train.add_argument(
        "--register", action="store_true", help="record the selected model in model_versions"
    )

    backtest = commands.add_parser(
        "backtest", help="walk forward through history, predicting each match from the past"
    )
    backtest.add_argument("--feature-version", help="feature-set version (default: current code)")
    backtest.add_argument("--start", type=_date, required=True, help="first prediction time")
    backtest.add_argument("--end", type=_date, help="YYYY-MM-DD (exclusive; default: last match)")
    backtest.add_argument(
        "--model",
        action="append",
        help="model and fixed hyperparameters, e.g. elo, poisson:alpha=3 (repeatable; "
        "default: elo)",
    )
    backtest.add_argument("--window", choices=WINDOWS, default="expanding")
    backtest.add_argument("--window-days", type=int, help="rolling window length")
    backtest.add_argument("--retrain-every-days", type=float, default=7.0)
    backtest.add_argument(
        "--calibration",
        default=",".join(CALIBRATION_METHODS),
        help="comma-separated variants (default: %(default)s)",
    )
    backtest.add_argument("--min-calibration-matches", type=int, default=300)
    backtest.add_argument("--min-training-matches", type=int, default=300)
    backtest.add_argument("--min-team-history", type=int, default=3)
    backtest.add_argument("--report", type=Path, help="write a Markdown backtest report here")
    backtest.add_argument(
        "--no-store", action="store_true", help="do not write predictions to the database"
    )

    registry = commands.add_parser("registry", help="list, activate, roll back and load models")
    registry.add_argument("--tracking-uri", help="MLflow tracking URI (default: backend/mlruns)")
    actions = registry.add_subparsers(dest="action", required=True)
    listing = actions.add_parser("list", help="every registered version, newest first")
    listing.add_argument("--name", help="only this model name")
    show = actions.add_parser("show", help="all metadata of a version (default: the active one)")
    show.add_argument("name")
    show.add_argument("--version")
    history = actions.add_parser("history", help="activations of a model name, newest first")
    history.add_argument("name")
    activate = actions.add_parser("activate", help="make a version the active one")
    activate.add_argument("name")
    activate.add_argument("version")
    activate.add_argument("--reason")
    deactivate = actions.add_parser("deactivate", help="leave a name without an active version")
    deactivate.add_argument("name")
    deactivate.add_argument("--reason")
    rollback = actions.add_parser(
        "rollback", help="re-activate the previously active version (or --to VERSION)"
    )
    rollback.add_argument("name")
    rollback.add_argument("--to", dest="to_version")
    rollback.add_argument("--reason")
    measure = actions.add_parser(
        "measure", help="record a version's results per competition, on matches after training"
    )
    measure.add_argument("name")
    measure.add_argument("version")
    measure.add_argument("--test-start", type=_date, required=True, help="YYYY-MM-DD")
    measure.add_argument("--test-end", type=_date, required=True, help="YYYY-MM-DD (exclusive)")
    measure.add_argument("--model-dir", type=Path, help="the artifact, if not at its stored path")
    load = actions.add_parser("load", help="load a version's artifact to check it is usable")
    load.add_argument("name")
    load.add_argument("--version")
    load.add_argument("--feature-version", help="refuse a model trained on other features")
    return parser


def _registry(args: argparse.Namespace, session_factory: sessionmaker[Session]) -> int:
    try:
        with session_factory() as session, session.begin():
            registry = ModelRegistry(session)
            if args.action == "list":
                rows = registry.versions(args.name)
                print(
                    f"{'name':<34}{'version':<26}{'algorithm':<20}{'active':<8}"
                    f"{'dataset':<21}{'trained on':<26}competitions"
                )
                for row in rows:
                    period = f"{row.training_data_start} .. {row.training_data_end}"
                    names = ", ".join(str(c.get("name") or c["id"]) for c in row.competitions)
                    print(
                        f"{row.name:<34}{row.version:<26}{row.algorithm:<20}"
                        f"{'yes' if row.is_active else '':<8}{row.dataset_version:<21}"
                        f"{period:<26}{names}"
                    )
                return 0
            if args.action == "show":
                data = metadata(registry.get(args.name, args.version))
                print(json.dumps(data.model_dump(mode="json"), indent=2))
                return 0
            if args.action == "history":
                for activation in registry.history(args.name):
                    record = activation_record(activation)
                    until = record.deactivated_at or "now"
                    kind = " (rollback)" if record.is_rollback else ""
                    print(
                        f"{record.version:<26}{record.activated_at} .. {until}{kind}"
                        f"  previous: {record.previous_version}  {record.reason or ''}"
                    )
                return 0
            if args.action == "measure":
                try:
                    measured = measure_by_competition(
                        session,
                        registry.get(args.name, args.version),
                        args.test_start,
                        args.test_end,
                        model_dir=args.model_dir,
                    )
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                for competition_id, figures in measured.items():
                    print(
                        f"competition {competition_id:<6}{figures['matches']:>6} matches  "
                        f"right result {figures['accuracy']:.1%}  "
                        f"exact score {figures['exact_score_accuracy']:.1%}"
                    )
                return 0
            if args.action == "load":
                loaded = load_model_version(
                    session,
                    args.name,
                    args.version,
                    feature_set_version=args.feature_version,
                    tracking_uri=args.tracking_uri,
                )
                print(
                    f"loaded {loaded.version.name} {loaded.version.version}: "
                    f"{type(loaded.model).__name__}, {len(loaded.model.used_features)} features, "
                    f"feature set {loaded.version.feature_set_version}"
                )
                return 0
            if args.action == "activate":
                registry.activate(args.name, args.version, reason=args.reason)
            elif args.action == "rollback":
                registry.rollback(args.name, to_version=args.to_version, reason=args.reason)
            else:
                registry.deactivate(args.name, reason=args.reason)
            active = registry.active(args.name)
            mirror = (
                (active.mlflow_model_name, active.mlflow_model_version)
                if active is not None
                else (_mirrored_name(registry, args.name), None)
            )
            summary = (
                f"{args.name}: active version {active.version}"
                if active is not None
                else f"{args.name}: no active version"
            )
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(summary)
    print(sync_active(*mirror, uri=args.tracking_uri))
    return 0


def _mirrored_name(registry: ModelRegistry, name: str) -> str | None:
    return next(
        (row.mlflow_model_name for row in registry.versions(name) if row.mlflow_model_name),
        None,
    )


def print_results(result: ExperimentResult) -> None:
    print(f"split: {result.split['split']}")
    for part in ("train", "validation", "test"):
        info = result.split[part]
        print(f"  {part:<11}{info['matches']:>6} matches  {info['start']} .. {info['end']}")
    header = f"  {'model':<30}{'set':<6}" + "".join(f"{label:>12}" for _, label in TABLE_METRICS)
    print("\n" + header)
    for candidate in result.candidates:
        mark = " *" if candidate.name == result.selected else ""
        for label, metrics in (
            ("val", candidate.validation_metrics),
            ("test", candidate.test_metrics or {}),
        ):
            values = "".join(
                f"{metrics[key]:>12.4f}" if key in metrics else f"{'-':>12}"
                for key, _ in TABLE_METRICS
            )
            print(f"  {(candidate.name + mark) if label == 'val' else '':<30}{label:<6}{values}")
        print(f"  {'':<30}params: {candidate.params}")
    if result.selection_metric.startswith("cv"):
        print(f"\n  pooled over {len(result.cv_folds)} rolling-origin folds (lower is better):")
        for candidate in sorted(result.candidates, key=lambda c: c.selection_score):
            print(f"  {candidate.name:<30}cv logloss {candidate.selection_score:.4f}")
    diffs = comparisons(result)
    if diffs:
        print(f"\n  test log loss minus {result.selected!r}, 95% paired bootstrap interval:")
        for name, row in sorted(diffs.items(), key=lambda kv: kv[1]["difference"]):
            print(f"  {name:<30}{row['difference']:+.4f}  [{row['low']:+.4f}, {row['high']:+.4f}]")
    print(
        f"\n* selected on {result.selection_metric}: {result.selected!r}. "
        "Test figures are reported only, never used to choose."
    )


def print_ensembles(ensemble: EnsembleResult) -> None:
    matches = sum(f["validation"]["matches"] for f in ensemble.folds[1:])
    print(
        f"\nensembles of {', '.join(ensemble.members)}: forward-chained validation over "
        f"{len(ensemble.folds) - 1} blocks ({matches} matches), combiners fitted on earlier "
        "blocks only"
    )
    methods = {c.method for c in ensemble.candidates}
    for name, loss in ensemble.ranking:
        kind = "ensemble" if name in methods else "individual"
        mark = " *" if name == ensemble.best_overall else ""
        print(f"  {name + mark:<32}{kind:<12}logloss {loss:.4f}")
    for entry in ensemble.candidates:
        weights = entry.combiner.params().get("weights")
        shown = ", ".join(f"{k} {v:.2f}" for k, v in weights.items() if v) if weights else "meta"
        test = entry.test_metrics.get("log_loss")
        test_text = f"  test logloss {test:.4f}" if test is not None else ""
        print(f"  {entry.method:<18} weights: {shown}{test_text}")
    print(
        f"  forward log loss minus {ensemble.best_individual!r} (best individual), "
        "95% paired bootstrap interval:"
    )
    for name, row in forward_comparisons(ensemble).items():
        print(f"  {name:<32}{row['difference']:+.4f}  [{row['low']:+.4f}, {row['high']:+.4f}]")
    print(f"* lowest forward-validation log loss: {ensemble.best_overall!r}")
    print(
        f"* recommended: {ensemble.recommendation['recommended']!r}. "
        f"{ensemble.recommendation['reason']}"
    )


def _backtest(
    args: argparse.Namespace, data: Dataset, session_factory: sessionmaker[Session]
) -> int:
    builders = {c.name: c.build for c in default_candidates(with_odds=True)}
    try:
        models = [ModelSpec.parse(text) for text in (args.model or ["elo"])]
        config = BacktestConfig(
            start=args.start,
            end=args.end,
            window=args.window,
            window_days=args.window_days,
            retrain_every=timedelta(days=args.retrain_every_days),
            min_team_history=args.min_team_history,
            min_training_matches=args.min_training_matches,
            calibration=tuple(c.strip() for c in args.calibration.split(",") if c.strip()),
            min_calibration_matches=args.min_calibration_matches,
        )
        unknown = sorted({m.name for m in models} - set(builders))
        if unknown:
            raise ValueError(f"unknown models {unknown}; available: {', '.join(builders)}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    store = None
    if not args.no_store:
        store = BacktestStore(
            session_factory,
            models=models,
            config=config,
            data=data,
            algorithms={m.label: type(builders[m.name](**m.params)).__name__ for m in models},
        )
    result = run_backtest(data, models, config, builders, sink=store)
    if store is not None:
        store.finish(result)
    with session_factory() as session:
        # (a Result has .keys(), so dict() would treat it as a mapping)
        competitions = {i: n for i, n in session.execute(select(Competition.id, Competition.name))}
        seasons = {i: n for i, n in session.execute(select(Season.id, Season.name))}
    predicted = sum(1 for s in result.steps if not s.skipped)
    print(
        f"backtest {config.start.date()} .. {config.end.date() if config.end else 'end'}: "
        f"{predicted} steps, {config.window} window, "
        f"{result.skipped_matches} matches skipped (not enough training data)"
    )
    header = f"  {'model':<40}" + "".join(f"{label:>10}" for _, label in BACKTEST_METRICS)
    print(header)
    for model, variant in result.variants:
        metrics = result.metrics(model, variant)
        name = model if variant == "none" else f"{model} + {variant}"
        values = "".join(
            f"{metrics[k]:>10.4f}" if k in metrics else f"{'-':>10}" for k, _ in BACKTEST_METRICS
        )
        print(f"  {name:<40}{values}")
    if store is not None:
        rows = ", ".join(f"{m}/{v}: {i}" for (m, v), i in store.versions.items())
        print(f"stored predictions under model_versions {rows}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            render_backtest_report(result, competitions=competitions, seasons=seasons),
            encoding="utf-8",
        )
        print(f"report: {args.report}")
    return 0


BACKTEST_METRICS = (
    ("matches", "matches"),
    ("accuracy", "acc"),
    ("log_loss", "logloss"),
    ("brier", "brier"),
    ("ece", "ece"),
    ("total_goals_mae", "goalMAE"),
    ("total_goals_rmse", "goalRMSE"),
    ("exact_score_accuracy", "exact"),
    ("top3_score_coverage", "top3"),
    ("top5_score_coverage", "top5"),
)


def main(
    argv: Sequence[str] | None = None, *, session_factory: sessionmaker[Session] | None = None
) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    version = getattr(args, "feature_version", None) or FeatureConfig().version()

    engine = None
    if session_factory is None:
        engine = create_db_engine(settings, statement_timeout=False)
        session_factory = create_session_factory(engine)
    try:
        if args.command == "registry":
            return _registry(args, session_factory)
        with session_factory() as session:
            data = load_dataset(session, version)
        if len(data) == 0:
            print(
                f"error: no feature snapshots for {version}; run `python -m app.features generate`",
                file=sys.stderr,
            )
            return 2
        if args.command == "backtest":
            return _backtest(args, data, session_factory)
        try:
            split: ChronologicalSplit
            if args.validation_start and args.test_start:
                split = split_by_dates(data, args.validation_start, args.test_start, args.test_end)
            else:
                split = split_by_seasons(data)
        except SplitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        candidates = default_candidates(with_odds=args.with_odds)
        if args.models:
            wanted = [name.strip() for name in args.models.split(",") if name.strip()]
            unknown = set(wanted) - {c.name for c in candidates}
            if unknown or not wanted:
                available = ", ".join(c.name for c in candidates)
                print(
                    f"error: unknown models {sorted(unknown)}; available: {available}",
                    file=sys.stderr,
                )
                return 2
            candidates = [c for c in candidates if c.name in wanted]
        if args.cv_folds < 1:
            print("error: --cv-folds must be at least 1", file=sys.stderr)
            return 2
        members = [m.strip() for m in args.ensemble_members.split(",") if m.strip()]
        if args.ensemble:
            if args.cv_folds < 2:
                print(
                    "error: --ensemble needs --cv-folds >= 2: ensemble weights are fitted on "
                    "earlier validation blocks and checked on later ones",
                    file=sys.stderr,
                )
                return 2
            not_trained = sorted(set(members) - {c.name for c in candidates})
            if not_trained or len(members) < 2:
                print(
                    f"error: ensemble members must be at least two trained models; "
                    f"not trained: {not_trained}",
                    file=sys.stderr,
                )
                return 2
        tracker = Tracker(args.experiment, args.tracking_uri)
        result = run_experiment(
            split,
            candidates,
            tracker=tracker,
            min_team_history=args.min_team_history,
            cv_folds=args.cv_folds,
        )
        print_results(result)
        ensemble = None
        if args.ensemble:
            ensemble = run_ensembles(
                split,
                result,
                candidates,
                cv_folds=args.cv_folds,
                min_team_history=args.min_team_history,
                members=members,
                tracker=tracker,
            )
            print_ensembles(ensemble)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(render_report(result, ensemble=ensemble), encoding="utf-8")
            print(f"report: {args.report}")
        print(f"\ntracking: {tracker.tracking_uri} (experiment {args.experiment!r})")
        if args.register:
            with session_factory() as session, session.begin():
                model_version = register_selected(session, result, tracker)
                ensemble_version = (
                    register_ensemble(session, ensemble, result, tracker) if ensemble else None
                )
            print(f"registered model_versions row {model_version}")
            if ensemble_version is not None:
                print(f"registered ensemble as model_versions row {ensemble_version}")
        return 0
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    sys.exit(main())
