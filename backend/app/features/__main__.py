"""Feature generation and inspection.

    python -m app.features version
    python -m app.features generate                        # every match, cutoff = kickoff
    python -m app.features generate --competition-id 1 --lead-hours 24
    python -m app.features show --match-id 42              # grouped values, for inspection
    python -m app.features export --out features.csv       # stored snapshots as CSV

Exit codes: 0 success, 1 some snapshots drifted (stored ones were kept), 2 error.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased, sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory
from app.features.builder import FeatureBuilder
from app.features.config import FeatureConfig
from app.features.registry import FEATURE_NAMES, SPECS
from app.features.store import FeatureStore
from app.features.timeline import load_targets
from app.models import FeatureSnapshot, Match, Team


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.features",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="print the feature-set version and parameters")

    def selection(command: argparse.ArgumentParser) -> None:
        command.add_argument("--competition-id", type=int)
        command.add_argument("--season-id", type=int)
        command.add_argument("--match-id", type=int, action="append")
        command.add_argument(
            "--lead-hours",
            type=float,
            default=0.0,
            help="cutoff = kickoff minus this many hours (default 0)",
        )

    generate = commands.add_parser("generate", help="compute and store feature snapshots")
    selection(generate)
    generate.add_argument("--finished-only", action="store_true")

    show = commands.add_parser("show", help="compute features for one match and print them")
    show.add_argument("--match-id", type=int, required=True)
    show.add_argument("--lead-hours", type=float, default=0.0)

    export = commands.add_parser("export", help="write stored snapshots to CSV")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--competition-id", type=int)
    return parser


def main(
    argv: Sequence[str] | None = None, *, session_factory: sessionmaker[Session] | None = None
) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    config = FeatureConfig()
    if args.command == "version":
        print(config.version())
        for key, value in vars(config).items():
            print(f"  {key:<32}{value}")
        print(f"  {'features':<32}{len(FEATURE_NAMES)}")
        return 0

    engine = None
    if session_factory is None:
        engine = create_db_engine(settings, statement_timeout=False)
        session_factory = create_session_factory(engine)
    try:
        builder = FeatureBuilder(session_factory, config)
        if args.command == "generate":
            return _generate(builder, args)
        if args.command == "show":
            return _show(builder, args.match_id, timedelta(hours=args.lead_hours))
        return _export(session_factory, builder.version, args.out, args.competition_id)
    finally:
        if engine is not None:
            engine.dispose()


def _generate(builder: FeatureBuilder, args: argparse.Namespace) -> int:
    with builder.session_factory() as session:
        targets = load_targets(
            session,
            match_ids=args.match_id,
            competition_id=args.competition_id,
            season_id=args.season_id,
            finished_only=args.finished_only,
            lead_time=timedelta(hours=args.lead_hours),
        )
    rows = builder.build(targets)
    result = FeatureStore(builder.session_factory).save(rows)
    print(f"feature set {builder.version}: {len(rows)} match(es)")
    print(
        f"  created {result.created}, unchanged {result.unchanged}, drifted {len(result.drifted)}"
    )
    if result.drifted:
        print(f"  drifted (stored snapshots kept): {result.drifted[:20]}")
        return 1
    return 0


def _show(builder: FeatureBuilder, match_id: int, lead: timedelta) -> int:
    with builder.session_factory() as session:
        targets = load_targets(session, match_ids=[match_id], lead_time=lead)
        if not targets:
            print(f"error: no match {match_id}", file=sys.stderr)
            return 2
        match = session.get_one(Match, match_id)
        home, away = (
            session.get_one(Team, match.home_team_id),
            session.get_one(Team, match.away_team_id),
        )
        title = f"{home.name} v {away.name}, {match.kickoff_at:%Y-%m-%d %H:%M} UTC"
    (row,) = builder.build(targets)
    print(title)
    print(f"feature set {row.version}; cutoff {row.target.cutoff:%Y-%m-%d %H:%M} UTC")
    print(f"newest information used became known at {row.latest_information_at}")
    group = None
    for spec in SPECS:
        if spec.group != group:
            group = spec.group
            print(f"\n[{group}]")
        value = row.values[spec.name]
        shown = "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"  {spec.name:<34}{shown:>10}   {spec.description}")
    return 0


def _export(
    session_factory: sessionmaker[Session], version: str, out: Path, competition_id: int | None
) -> int:
    home, away = aliased(Team), aliased(Team)
    query = (
        select(FeatureSnapshot, Match, home.name, away.name)
        .join(Match, Match.id == FeatureSnapshot.match_id)
        .join(home, home.id == Match.home_team_id)
        .join(away, away.id == Match.away_team_id)
        .where(FeatureSnapshot.feature_set_version == version)
        .order_by(Match.kickoff_at, Match.id)
    )
    if competition_id is not None:
        query = query.where(Match.competition_id == competition_id)
    with session_factory() as session:
        rows = session.execute(query).all()
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # Result columns are for inspection only and are prefixed "target_": they
        # are what a model predicts, never an input.
        writer.writerow(
            [
                "match_id",
                "kickoff_at",
                "cutoff",
                "home_team",
                "away_team",
                "target_home_goals",
                "target_away_goals",
                *FEATURE_NAMES,
            ]
        )
        for snapshot, match, home_name, away_name in rows:
            writer.writerow(
                [
                    match.id,
                    match.kickoff_at.isoformat(),
                    snapshot.data_cutoff_at.isoformat(),
                    home_name,
                    away_name,
                    match.home_score,
                    match.away_score,
                    *(snapshot.features.get(name) for name in FEATURE_NAMES),
                ]
            )
    print(f"wrote {len(rows)} snapshot(s) of {version} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
