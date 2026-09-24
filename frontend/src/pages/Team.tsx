import { useCallback, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getTeam } from '../api/endpoints.ts'
import type { SideRecord } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { LineChart } from '../components/charts/Charts.tsx'
import { MatchList } from '../components/MatchList.tsx'
import { FormPills, Stat, StatGrid } from '../components/Stat.tsx'
import { Card, Empty } from '../components/states.tsx'
import { decimal } from '../format.ts'
import { useApi } from '../hooks/useApi.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { PageHeader } from './PageHeader.tsx'

const windows = [10, 20, 40]

function SideRow({ record }: { record: SideRecord }) {
  return (
    <tr className="border-t border-slate-100">
      <th scope="row" className="py-2 pr-3 text-left font-medium capitalize text-slate-700">
        {record.side}
      </th>
      <td className="py-2 pr-3 tabular-nums">{record.played}</td>
      <td className="py-2 pr-3 tabular-nums">
        {record.wins}–{record.draws}–{record.losses}
      </td>
      <td className="py-2 pr-3 tabular-nums">{record.scored}</td>
      <td className="py-2 pr-3 tabular-nums">{record.conceded}</td>
      <td className="py-2 pr-3 tabular-nums">{decimal(record.goals_for_per_match)}</td>
      <td className="py-2 pr-3 tabular-nums">{decimal(record.goals_against_per_match)}</td>
      <td className="py-2 tabular-nums">{record.clean_sheets}</td>
    </tr>
  )
}

export function Team() {
  const teamId = Number(useParams().teamId)
  const [window, setWindow] = useState(20)
  const team = useApi(
    useCallback((signal: AbortSignal) => getTeam(teamId, window, signal), [teamId, window]),
    [teamId, window],
  )
  useDocumentTitle(team.state.kind === 'success' ? team.state.data.team.name : 'Team')

  return (
    <div className="space-y-6">
      <Async state={team.state} onRetry={team.reload} loadingLabel="Loading team…">
        {(data) => {
          const record = data.record
          return (
            <div className="space-y-6">
              <PageHeader
                title={data.team.name}
                subtitle={
                  <>
                    {data.team.country ?? 'Unknown country'} · last {record.played} finished{' '}
                    {record.played === 1 ? 'match' : 'matches'}
                  </>
                }
                action={
                  <label className="flex items-center gap-2 text-xs text-slate-600">
                    Window
                    <select
                      value={window}
                      onChange={(event) => {
                        setWindow(Number(event.target.value))
                      }}
                      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs"
                    >
                      {windows.map((size) => (
                        <option key={size} value={size}>
                          {size} matches
                        </option>
                      ))}
                    </select>
                  </label>
                }
              />

              {record.played === 0 ? (
                <Empty
                  title="No finished matches for this team"
                  hint="Results appear once an import has brought in played matches."
                />
              ) : (
                <>
                  <Card
                    title="Record"
                    action={<FormPills form={record.form} label={`${data.team.name} form`} />}
                  >
                    <StatGrid>
                      <Stat
                        label="Won / drawn / lost"
                        value={`${String(record.wins)}–${String(record.draws)}–${String(record.losses)}`}
                        note={`${String(record.played)} matches`}
                      />
                      <Stat
                        label="Scored"
                        value={record.goals_for}
                        note={`${decimal(record.goals_for_per_match)} per match`}
                      />
                      <Stat
                        label="Conceded"
                        value={record.goals_against}
                        note={`${decimal(record.goals_against_per_match)} per match`}
                      />
                      <Stat
                        label="Clean sheets"
                        value={record.clean_sheets}
                        note={`${((record.clean_sheets / record.played) * 100).toFixed(0)}% of matches`}
                      />
                      <Stat
                        label="xG for"
                        value={
                          record.matches_with_expected_goals === 0
                            ? '—'
                            : decimal(record.expected_goals_for_per_match)
                        }
                        note={
                          record.matches_with_expected_goals === 0
                            ? 'No provider supplied xG'
                            : `From ${String(record.matches_with_expected_goals)} matches`
                        }
                      />
                      <Stat
                        label="xG against"
                        value={
                          record.matches_with_expected_goals === 0
                            ? '—'
                            : decimal(record.expected_goals_against_per_match)
                        }
                        note={
                          record.matches_with_expected_goals === 0
                            ? 'No provider supplied xG'
                            : `From ${String(record.matches_with_expected_goals)} matches`
                        }
                      />
                    </StatGrid>
                  </Card>

                  <Card title="Home and away">
                    <div className="overflow-x-auto">
                      <table className="w-full min-w-[34rem] text-sm">
                        <thead>
                          <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
                            <th scope="col" className="pb-2 pr-3">
                              Venue
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              Played
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              W–D–L
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              For
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              Against
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              For / match
                            </th>
                            <th scope="col" className="pb-2 pr-3">
                              Against / match
                            </th>
                            <th scope="col" className="pb-2">
                              Clean sheets
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          <SideRow record={record.home} />
                          <SideRow record={record.away} />
                        </tbody>
                      </table>
                    </div>
                  </Card>

                  <Card title="Goal trend">
                    {data.trend.length < 2 ? (
                      <Empty title="Not enough matches for a trend" />
                    ) : (
                      <>
                        <LineChart
                          ariaLabel="Rolling goals scored and conceded, oldest match first"
                          series={[
                            {
                              name: 'Scored (5-match average)',
                              colour: '#16a34a',
                              points: data.trend.map((entry, index) => ({
                                x: index,
                                y: entry.rolling_goals_for,
                              })),
                            },
                            {
                              name: 'Conceded (5-match average)',
                              colour: '#dc2626',
                              points: data.trend.map((entry, index) => ({
                                x: index,
                                y: entry.rolling_goals_against,
                              })),
                            },
                          ]}
                        />
                        <p className="mt-1 text-xs text-slate-500">
                          Oldest match on the left. {data.trend.length} matches.
                        </p>
                      </>
                    )}
                  </Card>

                  <Card title="Recent matches">
                    {data.recent_matches.length === 0 ? (
                      <Empty title="No recent matches" />
                    ) : (
                      <MatchList matches={data.recent_matches} showResult />
                    )}
                  </Card>
                </>
              )}
            </div>
          )
        }}
      </Async>
    </div>
  )
}
