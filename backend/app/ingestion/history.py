"""Historical backfill: every available season of a competition, in order.

How far back the import goes is decided by the provider, not hard-coded: the
seasons it lists are imported, optionally limited by `first_year`/`last_year`.
Statistics are fetched only for seasons the provider says have them. Seasons it
says lack them are recorded as `missing_statistics` (info), not requested.

Re-running is safe and cheap: stored rows are updated in place, and statistics are
only requested for finished matches that do not have them yet.
"""

import logging
from dataclasses import dataclass, field

from app.ingestion.runs import RunStats
from app.ingestion.sync import ProviderSync, SeasonSelection
from app.models.enums import IngestionStatus
from app.providers.base import ProviderOperation
from app.providers.errors import AuthenticationError, ProviderError, QuotaExceededError
from app.providers.schemas import ExternalId

logger = logging.getLogger(__name__)

#: Errors after which further requests to the provider cannot succeed today.
STOP_ERRORS = (QuotaExceededError, AuthenticationError)


@dataclass
class ImportReport:
    competition_id: str
    seasons: list[str] = field(default_factory=list)
    runs: list[RunStats] = field(default_factory=list)
    skipped_statistics: list[str] = field(default_factory=list)
    failed_seasons: dict[str, str] = field(default_factory=dict)
    stopped: str | None = None

    @property
    def status(self) -> IngestionStatus:
        if self.stopped or self.failed_seasons:
            return IngestionStatus.FAILED
        if any(run.status is not IngestionStatus.SUCCEEDED for run in self.runs):
            return IngestionStatus.PARTIAL
        return IngestionStatus.SUCCEEDED


@dataclass
class HistoricalImport:
    sync: ProviderSync

    def run(
        self,
        competition_id: ExternalId,
        *,
        seasons: SeasonSelection = "all",
        first_year: int | None = None,
        last_year: int | None = None,
        statistics: bool = True,
        retry_missing_statistics: bool = False,
    ) -> ImportReport:
        report = ImportReport(competition_id)
        try:
            report.runs.append(self.sync.sync_competitions(only=[competition_id]))
            report.runs.append(self.sync.sync_seasons(competition_id))
        except ProviderError as exc:
            report.stopped = f"{type(exc).__name__}: {exc}"
            return report

        report.seasons = self.sync.select_seasons(
            competition_id, seasons, first_year=first_year, last_year=last_year
        )
        with_statistics = statistics and self.sync.provider.supports(
            ProviderOperation.TEAM_STATISTICS
        )
        for season_id in report.seasons:
            try:
                report.runs.append(self.sync.sync_teams(competition_id, season_id))
                report.runs.append(self.sync.sync_matches(competition_id, season_id))
                if not with_statistics:
                    continue
                if self.sync.season_statistics.get(season_id) is False:
                    self.sync.record_season_without_statistics(season_id)
                    report.skipped_statistics.append(season_id)
                    continue
                self.sync.clear_season_statistics_gap(season_id)
                report.runs.append(
                    self.sync.sync_team_statistics(
                        competition_id, season_id, retry_missing=retry_missing_statistics
                    )
                )
            except STOP_ERRORS as exc:
                report.stopped = f"{type(exc).__name__}: {exc}"
                logger.error("Stopping import of %s: %s", competition_id, exc)
                break
            except ProviderError as exc:
                report.failed_seasons[season_id] = f"{type(exc).__name__}: {exc}"
                logger.error("Season %s of %s failed: %s", season_id, competition_id, exc)
        return report
