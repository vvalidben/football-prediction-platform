import type { DependencyCheck } from '../api/health.ts'
import { config } from '../config.ts'
import { useHealth } from '../hooks/useHealth.ts'
import { StatusBadge } from './StatusBadge.tsx'

const dependencyLabels: Record<string, string> = {
  database: 'PostgreSQL',
  redis: 'Redis',
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function DependencyRow({ name, check }: { name: string; check: DependencyCheck }) {
  return (
    <li className="flex items-center justify-between gap-4 py-3">
      <div className="min-w-0">
        <p className="text-sm font-medium text-slate-900">{dependencyLabels[name] ?? name}</p>
        {check.error && <p className="truncate text-xs text-red-600">{check.error}</p>}
      </div>
      <div className="flex shrink-0 items-center gap-3">
        {check.latency_ms !== null && (
          <span className="font-mono text-xs text-slate-500">{check.latency_ms.toFixed(1)} ms</span>
        )}
        <StatusBadge
          tone={check.status === 'ok' ? 'ok' : 'error'}
          label={check.status === 'ok' ? 'Healthy' : 'Unavailable'}
        />
      </div>
    </li>
  )
}

export function HealthPanel() {
  const { state, isRefreshing, refresh } = useHealth(config.healthPollIntervalMs)

  return (
    <section
      aria-labelledby="system-status-heading"
      className="rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-5 py-4">
        <div>
          <h2 id="system-status-heading" className="text-sm font-semibold text-slate-900">
            System status
          </h2>
          <p className="text-xs text-slate-500">
            {state.kind === 'loading'
              ? 'Checking backend…'
              : `Last checked ${formatTime(state.checkedAt)}`}
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          disabled={isRefreshing}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pitch-600 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isRefreshing ? 'Checking…' : 'Refresh'}
        </button>
      </div>

      <div className="px-5 py-2" aria-live="polite">
        {state.kind === 'loading' && (
          <div className="flex items-center justify-between py-3">
            <p className="text-sm text-slate-600">Backend API</p>
            <StatusBadge tone="pending" label="Checking" />
          </div>
        )}

        {state.kind === 'error' && (
          <div className="flex items-center justify-between gap-4 py-3">
            <div>
              <p className="text-sm font-medium text-slate-900">Backend API</p>
              <p className="text-xs text-red-600">{state.message}</p>
            </div>
            <StatusBadge tone="error" label="Offline" />
          </div>
        )}

        {state.kind === 'success' && (
          <ul className="divide-y divide-slate-100">
            <li className="flex items-center justify-between gap-4 py-3">
              <div>
                <p className="text-sm font-medium text-slate-900">Backend API</p>
                <p className="text-xs text-slate-500">
                  {state.data.service} v{state.data.version} · {state.data.environment}
                </p>
              </div>
              <StatusBadge
                tone={state.data.status === 'ok' ? 'ok' : 'warning'}
                label={state.data.status === 'ok' ? 'Operational' : 'Degraded'}
              />
            </li>
            {Object.entries(state.data.checks).map(([name, check]) => (
              <DependencyRow key={name} name={name} check={check} />
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
