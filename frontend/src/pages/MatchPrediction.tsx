import { useCallback, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getExplanation, getMatch, getModels, getTeam, predictMatch } from '../api/endpoints.ts'
import type { CompetitionReliability, ModelVersion, Prediction, TeamDetail } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { ExplanationCard } from '../components/Explanation.tsx'
import { BarChart, ColumnChart } from '../components/charts/Charts.tsx'
import { FormPills, Stat, StatGrid } from '../components/Stat.tsx'
import { StatusBadge } from '../components/StatusBadge.tsx'
import { Card, Empty } from '../components/states.tsx'
import { dateTime, decimal, percent, relative, scoreline } from '../format.ts'
import { useApi } from '../hooks/useApi.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { PageHeader } from './PageHeader.tsx'

/** Total-goals distribution, summed off the score grid the model produced. */
function totalGoals(matrix: number[][]): { label: string; value: number }[] {
  const totals = new Map<number, number>()
  matrix.forEach((row, home) => {
    row.forEach((probability, away) => {
      totals.set(home + away, (totals.get(home + away) ?? 0) + probability)
    })
  })
  return [...totals.entries()]
    .filter(([goals]) => goals <= 6)
    .sort((a, b) => a[0] - b[0])
    .map(([goals, probability]) => ({ label: String(goals), value: probability }))
}

/** The figures recorded for a competition, for models that have any. */
function reliabilityIn(
  model: ModelVersion | undefined,
  competitionId: number,
): CompetitionReliability | undefined {
  const measured = model?.metrics.test_by_competition as
    Record<string, CompetitionReliability> | undefined
  return measured?.[String(competitionId)]
}

const monthYear = (value: string) =>
  new Date(value).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })

/** What the model's record in this league says about how far to trust it. */
function ReliabilityCard({
  competition,
  figures,
}: {
  competition: string
  figures: CompetitionReliability
}) {
  return (
    <Card title={`How reliable is this model in ${competition}?`}>
      <StatGrid>
        <Stat
          label="Right result"
          value={percent(figures.accuracy)}
          note={`Always picking the home team: ${percent(figures.home_win_rate)}`}
        />
        <Stat
          label="Exact score"
          value={percent(figures.exact_score_accuracy)}
          note="Its most likely score was exactly right"
        />
        <Stat
          label="Tested on"
          value={`${String(figures.matches)} matches`}
          note={`${monthYear(figures.start)} – ${monthYear(figures.end)}`}
        />
      </StatGrid>
      <p className="mt-3 text-xs text-slate-500">
        Measured on matches this model had not been trained on. This is its record across the
        league, not the chance that this prediction is right.
      </p>
    </Card>
  )
}

function ScoreGrid({ prediction }: { prediction: Prediction }) {
  const size = Math.min(prediction.max_goals + 1, 6)
  const rows = prediction.score_probability_matrix.slice(0, size)
  const largest = Math.max(...rows.flatMap((row) => row.slice(0, size)), 0.0001)
  const best = prediction.most_probable_score

  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <caption className="mb-2 text-left text-xs text-slate-500">
          Probability of each exact score — {prediction.home_team} down, {prediction.away_team}{' '}
          across.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="p-1" />
            {rows.map((_, away) => (
              <th key={away} scope="col" className="p-1 font-medium text-slate-500">
                {away}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, home) => (
            <tr key={home}>
              <th scope="row" className="p-1 font-medium text-slate-500">
                {home}
              </th>
              {row.slice(0, size).map((probability, away) => {
                const isBest = best.home_goals === home && best.away_goals === away
                return (
                  <td key={away} className="p-0.5">
                    <div
                      title={`${String(home)}–${String(away)}: ${percent(probability)}`}
                      className={`grid size-10 place-items-center rounded tabular-nums ${
                        isBest ? 'ring-2 ring-pitch-600' : ''
                      }`}
                      style={{
                        backgroundColor: `rgba(22, 163, 74, ${String(
                          Math.min(probability / largest, 1) * 0.85,
                        )})`,
                        color: probability / largest > 0.55 ? 'white' : '#0f172a',
                      }}
                    >
                      {(probability * 100).toFixed(1)}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TeamForm({ team, side }: { team: TeamDetail; side: 'Home' | 'Away' }) {
  const record = team.record
  const venue = side === 'Home' ? record.home : record.away
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <Link
          to={`/teams/${String(team.team.id)}`}
          className="text-sm font-semibold text-slate-900 hover:underline"
        >
          {team.team.name}
        </Link>
        <FormPills form={record.form} label={`${team.team.name} recent form`} />
      </div>
      <dl className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <dt className="text-xs text-slate-500">Scored per match</dt>
          <dd className="font-semibold tabular-nums">{decimal(record.goals_for_per_match)}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">Conceded per match</dt>
          <dd className="font-semibold tabular-nums">{decimal(record.goals_against_per_match)}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">{side} record</dt>
          <dd className="font-semibold tabular-nums">
            {venue.wins}W {venue.draws}D {venue.losses}L
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">Clean sheets</dt>
          <dd className="font-semibold tabular-nums">{record.clean_sheets}</dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">xG per match</dt>
          <dd className="font-semibold tabular-nums">
            {record.matches_with_expected_goals === 0
              ? 'not provided'
              : decimal(record.expected_goals_for_per_match)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500">xG against</dt>
          <dd className="font-semibold tabular-nums">
            {record.matches_with_expected_goals === 0
              ? 'not provided'
              : decimal(record.expected_goals_against_per_match)}
          </dd>
        </div>
      </dl>
    </div>
  )
}

export function MatchPrediction() {
  const matchId = Number(useParams().matchId)
  const [refreshToken, setRefreshToken] = useState(0)

  const match = useApi(
    useCallback((signal: AbortSignal) => getMatch(matchId, signal), [matchId]),
    [matchId],
  )
  const models = useApi(
    useCallback((signal: AbortSignal) => getModels({}, signal), []),
    [],
  )
  const prediction = useApi(
    useCallback(
      (signal: AbortSignal) =>
        predictMatch({ match_id: matchId, refresh: refreshToken > 0 }, signal),
      [matchId, refreshToken],
    ),
    [matchId, refreshToken],
  )
  const predictionId = prediction.state.kind === 'success' ? prediction.state.data.id : null
  const explanation = useApi(
    useCallback(
      (signal: AbortSignal) =>
        predictionId === null ? Promise.resolve(null) : getExplanation(predictionId, 8, signal),
      [predictionId],
    ),
    [predictionId],
  )
  useDocumentTitle(
    match.state.kind === 'success'
      ? `${match.state.data.home_team.name} v ${match.state.data.away_team.name}`
      : 'Match prediction',
  )
  const homeTeamId = match.state.kind === 'success' ? match.state.data.home_team.id : null
  const awayTeamId = match.state.kind === 'success' ? match.state.data.away_team.id : null
  const home = useApi(
    useCallback(
      (signal: AbortSignal) =>
        homeTeamId === null ? Promise.resolve(null) : getTeam(homeTeamId, 10, signal),
      [homeTeamId],
    ),
    [homeTeamId],
  )
  const away = useApi(
    useCallback(
      (signal: AbortSignal) =>
        awayTeamId === null ? Promise.resolve(null) : getTeam(awayTeamId, 10, signal),
      [awayTeamId],
    ),
    [awayTeamId],
  )

  return (
    <div className="space-y-6">
      <Async state={match.state} onRetry={match.reload} loadingLabel="Loading match…">
        {(data) => (
          <PageHeader
            eyebrow={
              <Link
                to={`/matches?league=${String(data.competition.id)}`}
                className="text-xs font-medium text-pitch-700 hover:underline"
              >
                ← All {data.competition.name} matches
              </Link>
            }
            title={`${data.home_team.name} v ${data.away_team.name}`}
            subtitle={
              <>
                {data.competition.name}
                {data.season ? ` · ${data.season}` : ''} · {dateTime(data.kickoff_at)}
                {data.venue ? ` · ${data.venue}` : ''}
              </>
            }
            action={
              <div className="flex items-center gap-2">
                <StatusBadge
                  tone={data.status === 'finished' ? 'ok' : 'pending'}
                  label={
                    data.status === 'finished'
                      ? `Played ${scoreline(data.home_score, data.away_score)}`
                      : data.status
                  }
                />
                <button
                  type="button"
                  onClick={() => {
                    setRefreshToken((token) => token + 1)
                  }}
                  className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  Recompute
                </button>
              </div>
            }
          />
        )}
      </Async>

      <Async state={prediction.state} onRetry={prediction.reload} loadingLabel="Running the model…">
        {(forecast) => (
          <div className="space-y-6">
            <Card
              title="Prediction"
              action={
                <span className="text-xs text-slate-500">
                  {forecast.model.name} {forecast.model.version} · {forecast.model.algorithm} ·{' '}
                  {forecast.reused ? 'stored' : 'new'} {relative(forecast.predicted_at)}
                </span>
              }
            >
              <StatGrid>
                <Stat
                  label="Most probable score"
                  value={`${String(forecast.most_probable_score.home_goals)}–${String(
                    forecast.most_probable_score.away_goals,
                  )}`}
                  note={percent(forecast.most_probable_score.probability)}
                />
                <Stat
                  label="Expected goals"
                  value={`${decimal(forecast.expected_home_goals, 2)} – ${decimal(
                    forecast.expected_away_goals,
                    2,
                  )}`}
                  note="Mean of the score distribution"
                />
                <Stat
                  label="Both teams score"
                  value={percent(forecast.btts_probability.yes)}
                  note={`No ${percent(forecast.btts_probability.no)}`}
                />
                <Stat
                  label="Features built"
                  value={relative(forecast.data_freshness.feature_cutoff_at)}
                  note={
                    forecast.data_freshness.match_started
                      ? 'Cut off at kickoff'
                      : `Last result ${relative(forecast.data_freshness.latest_result_at)}`
                  }
                />
              </StatGrid>
            </Card>

            {(() => {
              if (match.state.kind !== 'success' || models.state.kind !== 'success') return null
              const competition = match.state.data.competition
              const figures = reliabilityIn(
                models.state.data.find((model) => model.id === forecast.model.id),
                competition.id,
              )
              return figures ? (
                <ReliabilityCard competition={competition.name} figures={figures} />
              ) : null
            })()}

            <div className="grid gap-6 lg:grid-cols-2">
              <Card title="Match result">
                <BarChart
                  ariaLabel="Result probabilities"
                  max={1}
                  bars={[
                    {
                      label: forecast.home_team,
                      value: forecast.home_win_probability,
                      highlight: true,
                    },
                    { label: 'Draw', value: forecast.draw_probability },
                    { label: forecast.away_team, value: forecast.away_win_probability },
                  ]}
                />
              </Card>

              <Card title="Total goals">
                <ColumnChart
                  ariaLabel="Total goals distribution"
                  bars={totalGoals(forecast.score_probability_matrix)}
                />
                <p className="mt-2 text-xs text-slate-500">
                  Summed from the same score grid as every other market below.
                </p>
              </Card>

              <Card title="Over / under">
                {forecast.over_under_probabilities.length === 0 ? (
                  <Empty title="No lines returned for this prediction" />
                ) : (
                  <ul className="divide-y divide-slate-100 text-sm">
                    {forecast.over_under_probabilities.map((line) => (
                      <li
                        key={line.line}
                        className="flex items-center justify-between gap-3 py-2 tabular-nums"
                      >
                        <span className="font-medium text-slate-900">
                          {line.line.toFixed(1)} goals
                        </span>
                        <span className="text-slate-600">
                          over {percent(line.over)} · under {percent(line.under)}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              <Card title="Most likely scores">
                <BarChart
                  ariaLabel="Most likely exact scores"
                  bars={forecast.top_scores.slice(0, 8).map((score, index) => ({
                    label: `${String(score.home_goals)}–${String(score.away_goals)}`,
                    value: score.probability,
                    highlight: index === 0,
                  }))}
                />
              </Card>
            </div>

            <Card
              title="Why this prediction"
              action={
                <span className="text-xs text-slate-500">
                  How the model weighed its inputs — not what causes a result
                </span>
              }
            >
              <ExplanationCard state={explanation.state} onRetry={explanation.reload} />
            </Card>

            <Card title="Exact score distribution">
              <ScoreGrid prediction={forecast} />
            </Card>
          </div>
        )}
      </Async>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Home team form">
          <Async state={home.state} onRetry={home.reload} loadingLabel="Loading form…">
            {(team) =>
              team === null ? (
                <Empty title="Team not loaded" />
              ) : (
                <TeamForm team={team} side="Home" />
              )
            }
          </Async>
        </Card>
        <Card title="Away team form">
          <Async state={away.state} onRetry={away.reload} loadingLabel="Loading form…">
            {(team) =>
              team === null ? (
                <Empty title="Team not loaded" />
              ) : (
                <TeamForm team={team} side="Away" />
              )
            }
          </Async>
        </Card>
      </div>
    </div>
  )
}
