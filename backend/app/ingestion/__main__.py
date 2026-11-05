"""Command-line entry point for data ingestion and data quality.

    python -m app.ingestion competitions --country Morocco
    python -m app.ingestion sync --competition 39                     # current season
    python -m app.ingestion backfill --competition 39                 # every season offered
    python -m app.ingestion backfill --competition 39 --from 2015 --to 2023
    python -m app.ingestion audit                                     # database-wide checks
    python -m app.ingestion report [--json]                           # data-quality report

Exit codes: 0 success, 1 some records failed (partial), 2 an operation failed,
3 configuration or provider setup error.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import create_db_engine, create_session_factory
from app.ingestion.history import HistoricalImport, ImportReport
from app.ingestion.linking import link_competition
from app.ingestion.runs import RunStats
from app.ingestion.sync import ProviderSync, SeasonSelection, summarise
from app.models.enums import IngestionStatus
from app.providers.base import FootballDataProvider
from app.providers.errors import ProviderError
from app.providers.registry import PROVIDERS, create_provider
from app.quality.audit import run_audit
from app.quality.report import build_report, render_text

logger = logging.getLogger("app.ingestion")

EXIT_CODES = {IngestionStatus.SUCCEEDED: 0, IngestionStatus.PARTIAL: 1, IngestionStatus.FAILED: 2}
PROVIDER_COMMANDS = {"competitions", "sync", "backfill", "link"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default="api_football", choices=sorted(PROVIDERS))
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("competitions", help="list the provider's competitions")
    listing.add_argument("--country", help="filter by country (case-insensitive)")
    listing.add_argument("--search", help="filter by name substring (case-insensitive)")

    def competition_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--competition",
            action="append",
            required=True,
            metavar="ID",
            help="provider competition ID (repeatable)",
        )
        command.add_argument(
            "--season",
            action="append",
            metavar="ID",
            help="'current', 'all', or provider season IDs (repeatable)",
        )

    sync = commands.add_parser("sync", help="sync competitions, seasons, teams and matches")
    competition_args(sync)

    backfill = commands.add_parser(
        "backfill", help="import history: every season the provider offers, with statistics"
    )
    competition_args(backfill)
    backfill.add_argument("--from", dest="first_year", type=int, help="first season start year")
    backfill.add_argument("--to", dest="last_year", type=int, help="last season start year")
    backfill.add_argument("--no-statistics", action="store_true", help="skip team match statistics")
    backfill.add_argument(
        "--retry-missing-statistics",
        action="store_true",
        help="ask again for matches the provider previously had no statistics for",
    )

    link = commands.add_parser(
        "link", help="map a provider's competition and teams onto rows already stored"
    )
    link.add_argument("--competition", required=True, metavar="ID", help="provider competition ID")
    link.add_argument(
        "--to", dest="stored_competition", required=True, type=int, help="stored competition id"
    )
    link.add_argument("--dry-run", action="store_true", help="show the mapping, write nothing")

    commands.add_parser("audit", help="run database-wide data-quality checks")
    report = commands.add_parser("report", help="print the data-quality report")
    report.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def _season_selection(values: Sequence[str] | None, default: SeasonSelection) -> SeasonSelection:
    if not values:
        return default
    if list(values) == ["current"]:
        return "current"
    if list(values) == ["all"]:
        return "all"
    return list(values)


def _print_runs(runs: Sequence[RunStats]) -> None:
    print(
        f"{'run':>6}  {'job':<16}{'status':<11}{'fetched':>8}"
        f"{'new':>6}{'upd':>6}{'same':>6}{'fail':>6}"
    )
    for run in runs:
        print(
            f"{run.run_id:>6}  {run.job:<16}{run.status.value:<11}{run.fetched:>8}"
            f"{run.inserted:>6}{run.updated:>6}{run.unchanged:>6}{run.failed:>6}"
        )
        if run.error_message:
            print(f"        {run.error_message}")


def _print_import(report: ImportReport) -> None:
    print(f"\nCompetition {report.competition_id}: {len(report.seasons)} season(s) selected")
    _print_runs(report.runs)
    if report.skipped_statistics:
        print(
            f"  statistics not offered by the provider for: {', '.join(report.skipped_statistics)}"
        )
    for season_id, error in report.failed_seasons.items():
        print(f"  season {season_id} failed: {error}")
    if report.stopped:
        print(f"  stopped: {report.stopped}")


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: FootballDataProvider | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> int:
    """Run a command. `provider` and `session_factory` are injection points for tests."""
    args = _parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command in PROVIDER_COMMANDS and provider is None:
        try:
            provider = create_provider(args.provider, settings)
        except (ProviderError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

    engine = None
    if session_factory is None and args.command != "competitions":
        engine = create_db_engine(settings, statement_timeout=False)
        session_factory = create_session_factory(engine)
    try:
        return _dispatch(args, provider, session_factory)
    finally:
        if provider is not None:
            provider.close()
        if engine is not None:
            engine.dispose()


def _dispatch(
    args: argparse.Namespace,
    provider: FootballDataProvider | None,
    session_factory: sessionmaker[Session] | None,
) -> int:
    if args.command == "competitions":
        assert provider is not None  # noqa: S101 - guaranteed by main()
        return _list_competitions(provider, args.country, args.search)
    assert session_factory is not None  # noqa: S101 - guaranteed by main()

    if args.command == "audit":
        result = run_audit(session_factory)
        print(f"Audit found {len(result.issues)} issue(s); resolved {result.resolved}.")
        for issue_type, count in result.count_by_type().items():
            print(f"  {issue_type:<28}{count:>6}")
        return 0
    if args.command == "report":
        with session_factory() as session:
            report = build_report(session, datetime.now(UTC))
        print(report.to_json() if args.json else render_text(report))
        return 0

    assert provider is not None  # noqa: S101 - guaranteed by main()
    sync = ProviderSync(provider, session_factory)
    if args.command == "link":
        return _link(sync, args.competition, args.stored_competition, dry_run=args.dry_run)
    if args.command == "sync":
        runs: list[RunStats] = []
        for competition_id in args.competition:
            try:
                runs.extend(
                    sync.sync_competition(competition_id, _season_selection(args.season, "current"))
                )
            except ProviderError as exc:
                logger.error("Competition %s failed: %s", competition_id, exc)
                runs.append(RunStats(job="competition", status=IngestionStatus.FAILED))
        _print_runs([run for run in runs if run.run_id])
        return EXIT_CODES[summarise(runs)]

    importer = HistoricalImport(sync)
    statuses = []
    for competition_id in args.competition:
        imported = importer.run(
            competition_id,
            seasons=_season_selection(args.season, "all"),
            first_year=args.first_year,
            last_year=args.last_year,
            statistics=not args.no_statistics,
            retry_missing_statistics=args.retry_missing_statistics,
        )
        _print_import(imported)
        statuses.append(imported.status)
        if imported.stopped:
            break
    worst = max(statuses, key=lambda s: EXIT_CODES[s], default=IngestionStatus.SUCCEEDED)
    return EXIT_CODES[worst]


def _link(sync: ProviderSync, competition: str, stored: int, *, dry_run: bool) -> int:
    try:
        report = link_competition(sync, competition, stored, dry_run=dry_run)
    except (ProviderError, ValueError) as exc:
        logger.error("Linking competition %s failed: %s", competition, exc)
        return 1
    for title, mapping in (("by fixtures", report.by_fixtures), ("by name", report.by_name)):
        print(f"Linked {title} ({len(mapping)}):")
        for new, old in sorted(mapping.items()):
            print(f"  {new} -> {old}")
    print(f"New teams ({len(report.new_teams)}): {', '.join(sorted(report.new_teams))}")
    if report.ambiguous:
        print(f"AMBIGUOUS, left unlinked ({len(report.ambiguous)}): {report.ambiguous}")
    print("Dry run: nothing written." if dry_run else f"Wrote {report.mappings_written} mappings.")
    return 1 if report.ambiguous else 0


def _list_competitions(
    provider: FootballDataProvider, country: str | None, search: str | None
) -> int:
    try:
        result = provider.get_competitions()
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for competition in sorted(result.records, key=lambda c: (c.country or "", c.name)):
        if country and (competition.country or "").lower() != country.lower():
            continue
        if search and search.lower() not in competition.name.lower():
            continue
        print(
            f"{competition.external_id:>6}  {competition.type.value:<10}"
            f"{competition.country or 'International':<20}{competition.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
