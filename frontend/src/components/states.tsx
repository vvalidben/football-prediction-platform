/**
 * The three states every page can be in, shown the same way everywhere.
 *
 * `Empty` exists so a page can say "there is nothing here yet" instead of
 * inventing data: with no fixtures, no predictions or no evaluated matches, that
 * is the truthful answer.
 */
import type { ReactNode } from 'react'

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-600"
    >
      <span
        aria-hidden="true"
        className="size-4 animate-spin rounded-full border-2 border-slate-300 border-t-pitch-600"
      />
      {label}
    </div>
  )
}

interface ErrorStateProps {
  message: string
  status?: number | null
  onRetry?: () => void
}

export function ErrorState({ message, status, onRetry }: ErrorStateProps) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-6">
      <p className="text-sm font-medium text-red-800">
        {message}
        {status ? ` (HTTP ${String(status)})` : ''}
      </p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100"
        >
          Try again
        </button>
      )}
    </div>
  )
}

export function Empty({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-white p-6 text-center">
      <p className="text-sm font-medium text-slate-700">{title}</p>
      {hint && <p className="mt-1 text-sm text-slate-500">{hint}</p>}
    </div>
  )
}

export function Card({
  title,
  action,
  children,
}: {
  title?: ReactNode
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
      {(title ?? action) && (
        <header className="mb-3 flex items-center justify-between gap-3">
          {typeof title === 'string' ? (
            <h2 className="text-sm font-semibold text-slate-900">{title}</h2>
          ) : (
            title
          )}
          {action}
        </header>
      )}
      {children}
    </section>
  )
}
