/**
 * Prediction history: what was predicted, what happened, and how wrong it was.
 *
 * Filters live in the URL, so a filtered view can be linked to and reloaded. The
 * summary comes from the backend and covers every row matching the filters, not
 * the page on screen.
 */
import { useCallback } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  getCompetitions,
  getHistory,
  getModels,
  getSeasons,
  getTeams,
  type HistoryQuery,
} from '../api/endpoints.ts'
import type { HistoryRow } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { Stat, StatGrid } from '../components/Stat.tsx'
import { StatusBadge } from '../components/StatusBadge.tsx'
import { Card, Empty } from '../components/states.tsx'
import { dateTime, decimal, percent, scoreline } from '../format.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { useApi } from '../hooks/useApi.ts'
import { PageHeader } from './PageHeader.tsx'

const PAGE_SIZE = 25
const OUTCOME_LABELS: Record<string, string> = { home: 'Home', draw: 'Draw', away: 'Away' }

/** A filter's value, or undefined when it is not set. */
const numberOr = (value: string | null): number | undefined =>
  value === null || value === '' ? undefined : Number(value)

function Probabilities({ row }: { row: HistoryRow }) {
  const cells: [string, number][] = [
    ['home', row.home_win_probability],
    ['draw', row.draw_probability],
    ['away', row.away_win_probability],
  ]
  return (
    <span className="flex gap-1.5 tabular-nums">
      {cells.map(([key, value]) => (
        <span
          key={key}
          title={`${OUTCOME_LABELS[key]}: ${percent(value)}`}
          className={
            key === row.predicted_outcome
              ? 'font-semibold text-slate-900'
              : key === row.actual_outcome
                ? 'text-slate-700 underline decoration-dotted'
                : 'text-slate-400'
          }
        >
          {percent(value, 0)}
        </span>
      ))}
    </span>
  )
}

function ResultCell({ row }: { row: HistoryRow }) {
  if (row.home_score === null || row.away_score === null)
    return <span className="text-xs text-slate-500">{row.status}</span>
  return (
    <span className="flex items-center gap-2">
      <span className="font-semibold tabular-nums">
        {scoreline(row.home_score, row.away_score)}
      </span>
      {row.outcome_correct !== null && (
        <StatusBadge
          tone={row.outcome_correct ? 'ok' : 'error'}
          label={row.outcome_correct ? 'Hit' : 'Miss'}
        />
      )}
      {row.outcome_correct === null && <StatusBadge tone="pending" label="Not scored" />}
    </span>
  )
}

function ErrorCell({ row }: { row: HistoryRow }) {
  if (row.evaluated_at === null) return <span className="text-slate-400">—</span>
  return (
    <span className="flex flex-col text-xs tabular-nums text-slate-600">
      <span title="Log loss of the outcome probabilities">log loss {decimal(row.log_loss, 3)}</span>
      <span title="Expected goals minus actual goals, home and away">
        goals {signed(row.home_goals_error)} / {signed(row.away_goals_error)}
      </span>
    </span>
  )
}

const signed = (value: number | null): string =>
  value === null ? '—' : `${value >= 0 ? '+' : ''}${value.toFixed(2)}`

export function History() {
  useDocumentTitle('Prediction history')
  const [params, setParams] = useSearchParams()
  const competitionId = numberOr(params.get('competition'))
  const seasonId = numberOr(params.get('season'))
  const teamId = numberOr(params.get('team'))
  const modelName = params.get('model') ?? undefined
  const fromDate = params.get('from') ?? undefined
  const toDate = params.get('to') ?? undefined
  const status = params.get('status') ?? 'all'
  const offset = numberOr(params.get('offset')) ?? 0

  const query: HistoryQuery = {
    competition_id: competitionId,
    season_id: seasonId,
    team_id: teamId,
    model_name: modelName,
    from_date: fromDate,
    to_date: toDate,
    evaluated: status === 'all' ? undefined : status === 'evaluated',
    limit: PAGE_SIZE,
    offset,
  }

  // One key for the whole filter set: the request is re-sent when any of it changes.
  const queryKey = JSON.stringify(query)
  const history = useApi(
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useCallback((signal: AbortSignal) => getHistory(query, signal), [queryKey]),
    [queryKey],
  )
  const competitions = useApi(
    useCallback((signal: AbortSignal) => getCompetitions(signal), []),
    [],
  )
  const seasons = useApi(
    useCallback((signal: AbortSignal) => getSeasons(competitionId, signal), [competitionId]),
    [competitionId],
  )
  const teams = useApi(
    useCallback(
      (signal: AbortSignal) => getTeams({ competition_id: competitionId, limit: 100 }, signal),
      [competitionId],
    ),
    [competitionId],
  )
  const models = useApi(
    useCallback((signal: AbortSignal) => getModels({}, signal), []),
    [],
  )

  /** Setting any filter returns to the first page; the rest of the filters stay. */
  function update(changes: Record<string, string | undefined>) {
    const next = new URLSearchParams(params)
    for (const [key, value] of Object.entries(changes)) {
      if (value === undefined || value === '') next.delete(key)
      else next.set(key, value)
    }
    if (!('offset' in changes)) next.delete('offset')
    setParams(next)
  }

  const competitionOptions =
    competitions.state.kind === 'success' ? competitions.state.data.items : []
  const seasonOptions = seasons.state.kind === 'success' ? seasons.state.data.items : []
  const teamOptions = teams.state.kind === 'success' ? teams.state.data.items : []
  const modelNames =
    models.state.kind === 'success' ? [...new Set(models.state.data.map((row) => row.name))] : []
  const hasFilters = ['competition', 'season', 'team', 'model', 'from', 'to', 'status'].some(
    (key) => params.get(key),
  )

  return (
    <div className="space-y-6">
      <PageHeader
        title="Prediction history"
        subtitle="Every stored prediction with the result it was scored against. Predictions are kept as they were served; nothing is recalculated here."
      />

      <Card title="Filters">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-xs text-slate-600">
            Competition
            <select
              value={competitionId === undefined ? '' : String(competitionId)}
              onChange={(event) => {
                update({ competition: event.target.value, season: undefined, team: undefined })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            >
              <option value="">All competitions</option>
              {competitionOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-600">
            Season
            <select
              value={seasonId === undefined ? '' : String(seasonId)}
              onChange={(event) => {
                update({ season: event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            >
              <option value="">All seasons</option>
              {seasonOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                  {competitionId === undefined && option.competition
                    ? ` · ${option.competition}`
                    : ''}
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-600">
            Team
            <select
              value={teamId === undefined ? '' : String(teamId)}
              onChange={(event) => {
                update({ team: event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            >
              <option value="">All teams</option>
              {teamOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-600">
            Model
            <select
              value={modelName ?? ''}
              onChange={(event) => {
                update({ model: event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            >
              <option value="">All models</option>
              {modelNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs text-slate-600">
            Kickoff from
            <input
              type="date"
              value={fromDate ?? ''}
              onChange={(event) => {
                update({ from: event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            />
          </label>

          <label className="text-xs text-slate-600">
            Kickoff to
            <input
              type="date"
              value={toDate ?? ''}
              onChange={(event) => {
                update({ to: event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            />
          </label>

          <label className="text-xs text-slate-600">
            Status
            <select
              value={status}
              onChange={(event) => {
                update({ status: event.target.value === 'all' ? undefined : event.target.value })
              }}
              className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            >
              <option value="all">All predictions</option>
              <option value="evaluated">Scored</option>
              <option value="pending">Not scored yet</option>
            </select>
          </label>

          <div className="flex items-end">
            <button
              type="button"
              disabled={!hasFilters}
              onClick={() => {
                setParams(new URLSearchParams())
              }}
              className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Clear filters
            </button>
          </div>
        </div>
      </Card>

      <Async state={history.state} onRetry={history.reload} loadingLabel="Loading history…">
        {(page) => (
          <div className="space-y-6">
            <Card title="Summary">
              <StatGrid>
                <Stat
                  label="Predictions"
                  value={page.summary.predictions}
                  note={`${String(page.summary.evaluated)} scored · ${String(page.summary.pending)} pending`}
                />
                <Stat
                  label="Accuracy"
                  value={page.summary.accuracy === null ? '—' : percent(page.summary.accuracy)}
                  note="Outcome, over scored predictions"
                />
                <Stat
                  label="Exact score"
                  value={
                    page.summary.exact_score_accuracy === null
                      ? '—'
                      : percent(page.summary.exact_score_accuracy)
                  }
                />
                <Stat label="Log loss" value={decimal(page.summary.log_loss, 4)} />
                <Stat label="Brier" value={decimal(page.summary.brier_score, 4)} />
                <Stat
                  label="RPS"
                  value={decimal(page.summary.ranked_probability_score, 4)}
                  note="Ranked probability score"
                />
                <Stat
                  label="Goal error"
                  value={decimal(page.summary.mean_absolute_goals_error, 2)}
                  note="Mean absolute, per side"
                />
                <Stat
                  label="Covering"
                  value={
                    page.summary.evaluated === 0 ? '—' : dateTime(page.summary.first_kickoff_at)
                  }
                  note={
                    page.summary.last_kickoff_at
                      ? `to ${dateTime(page.summary.last_kickoff_at)}`
                      : undefined
                  }
                />
              </StatGrid>
            </Card>

            <Card
              title="Predictions"
              action={
                <span className="text-xs text-slate-500">
                  {page.total === 0
                    ? 'Nothing to show'
                    : `${String(offset + 1)}–${String(Math.min(offset + PAGE_SIZE, page.total))} of ${String(page.total)}`}
                </span>
              }
            >
              {page.items.length === 0 ? (
                <Empty
                  title="No predictions match these filters"
                  hint={
                    hasFilters
                      ? 'Clear a filter to widen the search.'
                      : 'Predictions appear here as soon as any are made.'
                  }
                />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[56rem] text-sm">
                    <thead>
                      <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                        <th scope="col" className="pb-2 pr-3">
                          Kickoff
                        </th>
                        <th scope="col" className="pb-2 pr-3">
                          Match
                        </th>
                        <th
                          scope="col"
                          className="pb-2 pr-3"
                          title="The most likely of every possible score"
                        >
                          Most likely score
                        </th>
                        <th scope="col" className="pb-2 pr-3">
                          H / D / A
                        </th>
                        <th scope="col" className="pb-2 pr-3">
                          Result
                        </th>
                        <th scope="col" className="pb-2 pr-3">
                          Error
                        </th>
                        <th scope="col" className="pb-2">
                          Model
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {page.items.map((row) => (
                        <tr key={row.prediction_id} className="border-t border-slate-100 align-top">
                          <td className="py-2 pr-3 text-xs text-slate-600">
                            {dateTime(row.kickoff_at)}
                            {row.season && <div className="text-slate-400">{row.season}</div>}
                          </td>
                          <td className="py-2 pr-3">
                            <Link
                              to={`/matches/${String(row.match_id)}`}
                              className="text-slate-900 hover:underline"
                            >
                              {row.home_team} v {row.away_team}
                            </Link>
                            {row.competition && (
                              <div className="text-xs text-slate-500">{row.competition}</div>
                            )}
                          </td>
                          <td className="py-2 pr-3">
                            <span className="font-semibold tabular-nums">
                              {row.predicted_home_goals}–{row.predicted_away_goals}
                            </span>
                            <div className="text-xs text-slate-500">
                              {OUTCOME_LABELS[row.predicted_outcome]} · xG{' '}
                              {decimal(row.expected_home_goals, 1)}–
                              {decimal(row.expected_away_goals, 1)}
                            </div>
                          </td>
                          <td className="py-2 pr-3">
                            <Probabilities row={row} />
                          </td>
                          <td className="py-2 pr-3">
                            <ResultCell row={row} />
                          </td>
                          <td className="py-2 pr-3">
                            <ErrorCell row={row} />
                          </td>
                          <td className="py-2 text-xs text-slate-600">
                            <Link
                              to={`/models?model=${String(row.model_id)}`}
                              className="hover:underline"
                            >
                              {row.model_name} {row.model_version}
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {page.total > PAGE_SIZE && (
                <div className="mt-4 flex items-center justify-between gap-3">
                  <button
                    type="button"
                    disabled={offset === 0}
                    onClick={() => {
                      update({ offset: String(Math.max(offset - PAGE_SIZE, 0)) })
                    }}
                    className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <button
                    type="button"
                    disabled={offset + PAGE_SIZE >= page.total}
                    onClick={() => {
                      update({ offset: String(offset + PAGE_SIZE) })
                    }}
                    className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              )}
            </Card>
          </div>
        )}
      </Async>
    </div>
  )
}
