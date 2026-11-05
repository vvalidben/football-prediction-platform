import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { getTeams } from '../api/endpoints.ts'
import { Async } from '../components/Async.tsx'
import { Card, Empty } from '../components/states.tsx'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { useApi } from '../hooks/useApi.ts'
import { PageHeader } from './PageHeader.tsx'

/** A way into the team pages: search by name, then follow the link. */
export function Teams() {
  useDocumentTitle('Teams')
  const [search, setSearch] = useState('')
  const teams = useApi(
    useCallback(
      (signal: AbortSignal) => getTeams({ search: search || undefined, limit: 50 }, signal),
      [search],
    ),
    [search],
  )

  return (
    <div className="space-y-6">
      <PageHeader title="Teams" subtitle="Every team the platform has matches for." />
      <Card
        action={
          <label className="flex w-full items-center gap-2 text-sm">
            <span className="sr-only">Search teams</span>
            <input
              type="search"
              value={search}
              placeholder="Search teams"
              onChange={(event) => {
                setSearch(event.target.value)
              }}
              className="w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pitch-600 sm:w-64"
            />
          </label>
        }
      >
        <Async state={teams.state} onRetry={teams.reload} loadingLabel="Loading teams…">
          {(page) =>
            page.items.length === 0 ? (
              <Empty
                title={search ? `No team matches “${search}”` : 'No teams imported yet'}
                hint={search ? undefined : 'Teams arrive with the first provider import.'}
              />
            ) : (
              <>
                <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
                  {page.items.map((team) => (
                    <li key={team.id}>
                      <Link
                        to={`/teams/${String(team.id)}`}
                        className="block truncate py-1 text-sm text-slate-900 hover:underline"
                      >
                        {team.name}
                        {team.country && <span className="text-slate-500"> · {team.country}</span>}
                      </Link>
                    </li>
                  ))}
                </ul>
                {page.total > page.items.length && (
                  <p className="mt-4 text-sm text-slate-500">
                    Showing {page.items.length} of {page.total} teams. Search by name to find the
                    others.
                  </p>
                )}
              </>
            )
          }
        </Async>
      </Card>
    </div>
  )
}
