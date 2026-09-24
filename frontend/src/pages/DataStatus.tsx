import { useCallback } from 'react'
import { getDataStatus } from '../api/endpoints.ts'
import { Async } from '../components/Async.tsx'
import { Stat, StatGrid } from '../components/Stat.tsx'
import { StatusBadge } from '../components/StatusBadge.tsx'
import { Card, Empty } from '../components/states.tsx'
import { config } from '../config.ts'
import { dateTime, relative } from '../format.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { useApi } from '../hooks/useApi.ts'
import { PageHeader } from './PageHeader.tsx'

const runTone = (status: string) =>
  status === 'success'
    ? 'ok'
    : status === 'failed'
      ? 'error'
      : status === 'skipped'
        ? 'warning'
        : 'pending'

/** How much of a competition's data is present, as a proportion of its matches. */
function CoverageBar({ label, value, total }: { label: string; value: number; total: number }) {
  const share = total === 0 ? 0 : value / total
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0 text-slate-500">{label}</span>
      <span className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
        <span
          className={`block h-full rounded-full ${share > 0.9 ? 'bg-pitch-600' : share > 0.5 ? 'bg-amber-500' : 'bg-red-500'}`}
          style={{ width: `${String(share * 100)}%` }}
        />
      </span>
      <span className="w-24 shrink-0 text-right tabular-nums text-slate-600">
        {value} / {total}
      </span>
    </div>
  )
}

export function DataStatus() {
  useDocumentTitle('Data status')
  const status = useApi(
    useCallback((signal: AbortSignal) => getDataStatus(signal), []),
    [],
    { refreshIntervalMs: config.healthPollIntervalMs },
  )

  return (
    <div className="space-y-6">
      <PageHeader
        title="Data status"
        subtitle="What has been imported, what is missing, and when each provider last ran."
      />

      <Async state={status.state} onRetry={status.reload} loadingLabel="Loading data status…">
        {(data) => (
          <div className="space-y-6">
            <Card title="Last synchronisation">
              <StatGrid>
                <Stat
                  label="Last import"
                  value={relative(data.last_successful_ingestion_at)}
                  note={dateTime(data.last_successful_ingestion_at)}
                />
                <Stat
                  label="Last result"
                  value={relative(data.latest_finished_match_at)}
                  note={dateTime(data.latest_finished_match_at)}
                />
                <Stat
                  label="Newest features"
                  value={relative(data.latest_feature_snapshot_at)}
                  note={`Feature set ${data.feature_set_version}`}
                />
                <Stat
                  label="Matches without features"
                  value={data.matches_missing_features}
                  note="Finished matches not yet processed"
                />
              </StatGrid>
            </Card>

            <Card title="Coverage by competition">
              {data.coverage.length === 0 ? (
                <Empty
                  title="Nothing imported yet"
                  hint="Run an import to populate competitions, matches and features."
                />
              ) : (
                <ul className="space-y-4">
                  {data.coverage.map((entry) => (
                    <li key={entry.competition_id} className="space-y-1.5">
                      <div className="flex items-baseline justify-between gap-3">
                        <h3 className="text-sm font-medium text-slate-900">{entry.competition}</h3>
                        <p className="text-xs text-slate-500">
                          {entry.matches} matches · {entry.seasons}{' '}
                          {entry.seasons === 1 ? 'season' : 'seasons'}
                        </p>
                      </div>
                      <CoverageBar label="Finished" value={entry.finished} total={entry.matches} />
                      <CoverageBar
                        label="With features"
                        value={entry.with_features}
                        total={entry.finished}
                      />
                      <CoverageBar
                        label="With statistics"
                        value={entry.with_statistics}
                        total={entry.finished}
                      />
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Providers">
                {data.providers.length === 0 ? (
                  <Empty
                    title="No provider configured"
                    hint="Providers appear once credentials are supplied and an import has been attempted."
                  />
                ) : (
                  <ul className="divide-y divide-slate-100">
                    {data.providers.map((provider) => (
                      <li
                        key={provider.code}
                        className="flex items-center justify-between gap-3 py-2.5"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-slate-900">
                            {provider.name}
                          </p>
                          <p className="truncate text-xs text-slate-500">
                            {provider.runs} {provider.runs === 1 ? 'run' : 'runs'} · last{' '}
                            {relative(provider.last_run_at)}
                          </p>
                        </div>
                        <StatusBadge
                          tone={provider.is_active ? 'ok' : 'pending'}
                          label={provider.is_active ? 'Enabled' : 'Disabled'}
                        />
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              <Card title="Imports">
                {data.ingestion.length === 0 ? (
                  <Empty title="No import has run yet" />
                ) : (
                  <ul className="divide-y divide-slate-100">
                    {data.ingestion.map((run) => (
                      <li
                        key={`${run.job}-${run.status}`}
                        className="flex items-center justify-between gap-3 py-2.5"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-slate-900">{run.job}</p>
                          <p className="truncate text-xs text-slate-500">
                            {run.records_inserted} rows · {relative(run.last_run_at)}
                          </p>
                        </div>
                        <StatusBadge tone={runTone(run.status)} label={run.status} />
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              <Card title="Background jobs">
                {data.jobs.length === 0 ? (
                  <Empty
                    title="No job has run yet"
                    hint="The scheduler records a row the first time each task runs."
                  />
                ) : (
                  <ul className="divide-y divide-slate-100">
                    {data.jobs.map((job) => (
                      <li
                        key={`${job.task}-${job.status}`}
                        className="flex items-center justify-between gap-3 py-2.5"
                      >
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium text-slate-900">{job.task}</p>
                          <p className="text-xs text-slate-500">{relative(job.last_run_at)}</p>
                        </div>
                        <StatusBadge tone={runTone(job.status)} label={job.status} />
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              <Card title="Open data-quality issues">
                {data.issues.length === 0 ? (
                  <Empty title="No open issues" hint="Nothing has been flagged by the checks." />
                ) : (
                  <ul className="divide-y divide-slate-100">
                    {data.issues.map((issue) => (
                      <li
                        key={`${issue.severity}-${issue.issue_type}`}
                        className="flex items-center justify-between gap-3 py-2.5 text-sm"
                      >
                        <span className="truncate text-slate-900">{issue.issue_type}</span>
                        <span className="flex shrink-0 items-center gap-2">
                          <span className="tabular-nums text-slate-600">{issue.count}</span>
                          <StatusBadge
                            tone={issue.severity === 'error' ? 'error' : 'warning'}
                            label={issue.severity}
                          />
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>
            </div>

            <p className="text-xs text-slate-500">Generated {dateTime(data.generated_at)}.</p>
          </div>
        )}
      </Async>
    </div>
  )
}
