import { useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  getFreshness,
  getModels,
  getStoredPredictions,
  getUpcomingMatches,
} from '../api/endpoints.ts'
import type { Freshness } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { HealthPanel } from '../components/HealthPanel.tsx'
import { MatchList } from '../components/MatchList.tsx'
import { StatusBadge } from '../components/StatusBadge.tsx'
import { Card, Empty } from '../components/states.tsx'
import { Stat, StatGrid } from '../components/Stat.tsx'
import { config } from '../config.ts'
import { dateOnly, dateTime, percent, relative } from '../format.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { useApi, type AsyncState } from '../hooks/useApi.ts'
import { PageHeader } from './PageHeader.tsx'

/**
 * Why there is nothing to show in "Next matches".
 *
 * Read from the freshness panel this page has already loaded, so the sentence
 * stays true as the data changes: a hard-coded season range would become the next
 * misleading message the moment more data arrives. Nothing is claimed that the
 * response cannot support — in particular it says what *is* loaded, not why a
 * provider stopped there.
 */
function noUpcomingHint(state: AsyncState<Freshness>): string {
  if (state.kind !== 'success') {
    return 'Nothing is scheduled in the data loaded so far.'
  }
  const latest = state.data.latest_finished_match_at
  if (latest === null) {
    return 'No matches have been imported yet, so there is nothing ahead to predict.'
  }
  return (
    `The data loaded ends with the match on ${dateOnly(latest)}; no later fixtures ` +
    'have been imported, so there is nothing ahead to predict.'
  )
}

export function Dashboard() {
  useDocumentTitle('Dashboard')
  const upcoming = useApi(
    useCallback((signal: AbortSignal) => getUpcomingMatches({ limit: 8 }, signal), []),
    [],
  )
  const predictions = useApi(
    useCallback(
      (signal: AbortSignal) => getStoredPredictions({ latest_per_match: true, limit: 8 }, signal),
      [],
    ),
    [],
  )
  const models = useApi(
    useCallback((signal: AbortSignal) => getModels({ active_only: true }, signal), []),
    [],
  )
  const freshness = useApi(
    useCallback((signal: AbortSignal) => getFreshness(signal), []),
    [],
    { refreshIntervalMs: config.healthPollIntervalMs },
  )

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        subtitle="What is coming up, what has been predicted, and how current the data behind it is."
      />

      <Card title="Data freshness">
        <Async state={freshness.state} onRetry={freshness.reload} loadingLabel="Loading freshness…">
          {(data) => (
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
                label="Features built"
                value={relative(data.latest_feature_snapshot_at)}
                note={`Feature set ${data.feature_set_version}`}
              />
              <Stat label="Upcoming matches" value={data.upcoming_matches} />
              <Stat label="Predictions (24h)" value={data.predictions_last_24h} />
              <Stat label="Active models" value={data.active_models} />
            </StatGrid>
          )}
        </Async>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Next matches"
          action={
            <Link to="/data" className="text-xs font-medium text-pitch-700 hover:underline">
              Data status
            </Link>
          }
        >
          <Async state={upcoming.state} onRetry={upcoming.reload} loadingLabel="Loading fixtures…">
            {(page) =>
              page.items.length === 0 ? (
                <Empty title="No scheduled matches" hint={noUpcomingHint(freshness.state)} />
              ) : (
                <MatchList matches={page.items} />
              )
            }
          </Async>
        </Card>

        <Card
          title="Latest predictions"
          action={
            <span className="text-xs text-slate-500">
              Most likely score, with the outcome probabilities
            </span>
          }
        >
          <Async
            state={predictions.state}
            onRetry={predictions.reload}
            loadingLabel="Loading predictions…"
          >
            {(page) =>
              page.items.length === 0 ? (
                <Empty
                  title="No predictions yet"
                  hint="Open a match and ask for a prediction, or wait for the scheduled job."
                />
              ) : (
                <ul className="divide-y divide-slate-100">
                  {page.items.map((item) => (
                    <li key={item.id} className="py-2.5 text-sm">
                      <div className="flex items-center justify-between gap-3">
                        <Link
                          to={`/matches/${String(item.match_id)}`}
                          className="truncate text-slate-900 hover:underline"
                        >
                          {item.home_team} v {item.away_team}
                        </Link>
                        <span className="shrink-0 font-semibold tabular-nums text-slate-900">
                          {item.most_probable_score}
                        </span>
                      </div>
                      <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-slate-500">
                        <span>
                          H {percent(item.home_win_probability, 0)} · D{' '}
                          {percent(item.draw_probability, 0)} · A{' '}
                          {percent(item.away_win_probability, 0)}
                        </span>
                        <span>·</span>
                        <span>
                          {item.model_name} {item.model_version}
                        </span>
                        {item.actual_score && (
                          <>
                            <span>·</span>
                            <span>played {item.actual_score}</span>
                          </>
                        )}
                        {item.outcome_correct !== null && (
                          <StatusBadge
                            tone={item.outcome_correct ? 'ok' : 'error'}
                            label={item.outcome_correct ? 'Correct' : 'Wrong'}
                          />
                        )}
                      </p>
                    </li>
                  ))}
                </ul>
              )
            }
          </Async>
        </Card>

        <Card title="Active models">
          <Async state={models.state} onRetry={models.reload} loadingLabel="Loading models…">
            {(rows) =>
              rows.length === 0 ? (
                <Empty
                  title="No active model"
                  hint="Train and activate a version before predictions can be made."
                />
              ) : (
                <ul className="divide-y divide-slate-100">
                  {rows.map((row) => (
                    <li key={row.id} className="flex items-center justify-between gap-3 py-2.5">
                      <div className="min-w-0">
                        <Link
                          to={`/models?model=${String(row.id)}`}
                          className="text-sm font-medium text-slate-900 hover:underline"
                        >
                          {row.name} {row.version}
                        </Link>
                        <p className="truncate text-xs text-slate-500">
                          {row.algorithm} · trained {relative(row.trained_at ?? row.created_at)}
                        </p>
                      </div>
                      <StatusBadge tone="ok" label="Active" />
                    </li>
                  ))}
                </ul>
              )
            }
          </Async>
        </Card>

        <HealthPanel />
      </div>
    </div>
  )
}
