import { useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getCompetitions, getUpcomingMatches } from '../api/endpoints.ts'
import type { MatchRead } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { MatchList } from '../components/MatchList.tsx'
import { Card, Empty } from '../components/states.tsx'
import { dateOnly } from '../format.ts'
import { useApi } from '../hooks/useApi.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { PageHeader } from './PageHeader.tsx'

const PAGE_SIZE = 25

/** A whole number from the address, or null for anything else. */
function wholeNumber(value: string | null): number | null {
  return value !== null && /^\d+$/.test(value) ? Number(value) : null
}

/** Matches under the day they are played, in the order they arrived (earliest first). */
function byDay(matches: MatchRead[]): { day: string; matches: MatchRead[] }[] {
  const days: { day: string; matches: MatchRead[] }[] = []
  for (const match of matches) {
    const day = dateOnly(match.kickoff_at)
    const current = days.at(-1)
    if (current?.day === day) {
      current.matches.push(match)
    } else {
      days.push({ day, matches: [match] })
    }
  }
  return days
}

/**
 * Choose a league, see what it has coming up, and open a match's prediction.
 *
 * The league and page live in the address, so going back from a prediction returns
 * to the same list, and a list can be bookmarked or shared.
 */
export function Matches() {
  useDocumentTitle('Matches')
  const [params, setParams] = useSearchParams()
  const leagueId = wholeNumber(params.get('league'))
  const pageNumber = Math.max(wholeNumber(params.get('page')) ?? 1, 1)
  const offset = (pageNumber - 1) * PAGE_SIZE

  const competitions = useApi(
    useCallback((signal: AbortSignal) => getCompetitions(signal), []),
    [],
  )
  const upcoming = useApi(
    useCallback(
      (signal: AbortSignal) =>
        leagueId === null
          ? Promise.resolve(null)
          : getUpcomingMatches({ competition_id: leagueId, limit: PAGE_SIZE, offset }, signal),
      [leagueId, offset],
    ),
    [leagueId, offset],
  )

  const leagueName =
    competitions.state.kind === 'success'
      ? competitions.state.data.items.find((item) => item.id === leagueId)?.name
      : undefined

  const goToPage = (next: number) => {
    setParams({ league: String(leagueId), page: String(next) })
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Matches"
        subtitle="Choose a league, then press Predict on any match it has coming up."
      />

      <Card title="League">
        <Async
          state={competitions.state}
          onRetry={competitions.reload}
          loadingLabel="Loading leagues…"
        >
          {(page) => (
            <label className="flex flex-col gap-1 text-sm sm:max-w-sm">
              <span className="sr-only">League</span>
              <select
                value={leagueId ?? ''}
                onChange={(event) => {
                  setParams(event.target.value ? { league: event.target.value } : {})
                }}
                className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pitch-600"
              >
                <option value="">Select a league…</option>
                {page.items.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.country ? `${item.name} · ${item.country}` : item.name}
                  </option>
                ))}
              </select>
            </label>
          )}
        </Async>
      </Card>

      <Card title={leagueName ? `Upcoming in ${leagueName}` : 'Upcoming matches'}>
        <Async state={upcoming.state} onRetry={upcoming.reload} loadingLabel="Loading fixtures…">
          {(page) => {
            if (page === null) {
              return (
                <Empty
                  title="Choose a league"
                  hint="Its upcoming matches will appear here, ready to predict."
                />
              )
            }
            if (page.items.length === 0 && pageNumber === 1) {
              return (
                <Empty
                  title="No upcoming matches"
                  hint={`${leagueName ?? 'This league'} has nothing scheduled in the data loaded so far.`}
                />
              )
            }
            const pages = Math.max(Math.ceil(page.total / PAGE_SIZE), 1)
            return (
              <div className="space-y-5">
                {byDay(page.items).map(({ day, matches }) => (
                  <section key={day}>
                    <h3 className="text-xs font-semibold tracking-wide text-slate-500 uppercase">
                      {day}
                    </h3>
                    <MatchList matches={matches} actionLabel="Predict" />
                  </section>
                ))}
                {pages > 1 && (
                  <nav
                    aria-label="Pages"
                    className="flex items-center justify-between gap-3 border-t border-slate-100 pt-4 text-sm"
                  >
                    <button
                      type="button"
                      disabled={pageNumber <= 1}
                      onClick={() => {
                        goToPage(pageNumber - 1)
                      }}
                      className="rounded-md border border-slate-300 px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Previous
                    </button>
                    <span className="text-slate-500">
                      Page {pageNumber} of {pages} · {page.total} matches
                    </span>
                    <button
                      type="button"
                      disabled={pageNumber >= pages}
                      onClick={() => {
                        goToPage(pageNumber + 1)
                      }}
                      className="rounded-md border border-slate-300 px-3 py-1.5 font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      Next
                    </button>
                  </nav>
                )}
              </div>
            )
          }}
        </Async>
      </Card>
    </div>
  )
}
