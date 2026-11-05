import { Link } from 'react-router-dom'
import type { MatchRead } from '../api/types.ts'
import { dateTime, scoreline } from '../format.ts'

/**
 * Matches as rows. Each links to its prediction page, and each team to its own
 * page, so the three views are reachable from one another.
 */
export function MatchList({
  matches,
  showResult,
  actionLabel,
}: {
  matches: MatchRead[]
  showResult?: boolean
  /** Makes the link to the prediction page the row's main action, under this name. */
  actionLabel?: string
}) {
  return (
    <ul className="divide-y divide-slate-100">
      {matches.map((match) => (
        <li key={match.id} className="flex items-center justify-between gap-3 py-2.5 text-sm">
          <div className="min-w-0">
            <p className="truncate text-slate-900">
              <Link className="hover:underline" to={`/teams/${String(match.home_team.id)}`}>
                {match.home_team.name}
              </Link>
              <span className="px-1.5 text-slate-400">v</span>
              <Link className="hover:underline" to={`/teams/${String(match.away_team.id)}`}>
                {match.away_team.name}
              </Link>
            </p>
            <p className="truncate text-xs text-slate-500">
              {match.competition.name} · {dateTime(match.kickoff_at)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {showResult && (
              <span className="font-semibold tabular-nums text-slate-900">
                {scoreline(match.home_score, match.away_score)}
              </span>
            )}
            <Link
              to={`/matches/${String(match.id)}`}
              className={
                actionLabel
                  ? 'rounded-md bg-pitch-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-pitch-700'
                  : 'rounded-md border border-slate-300 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50'
              }
            >
              {actionLabel ?? 'Prediction'}
            </Link>
          </div>
        </li>
      ))}
    </ul>
  )
}
