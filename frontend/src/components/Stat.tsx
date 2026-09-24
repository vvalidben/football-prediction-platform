import type { ReactNode } from 'react'

/** A single figure with its label, and a note for where the figure comes from. */
export function Stat({
  label,
  value,
  note,
}: {
  label: string
  value: ReactNode
  note?: ReactNode
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 text-lg font-semibold tabular-nums text-slate-900">{value}</dd>
      {note && <p className="mt-0.5 text-xs text-slate-500">{note}</p>}
    </div>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">{children}</dl>
}

const resultClasses: Record<string, string> = {
  W: 'bg-pitch-600 text-white',
  D: 'bg-slate-300 text-slate-800',
  L: 'bg-red-500 text-white',
}

/** The last few results, most recent last, as W/D/L pills. */
export function FormPills({ form, label = 'Recent form' }: { form: string[]; label?: string }) {
  if (form.length === 0) return <span className="text-sm text-slate-500">No matches yet</span>
  return (
    <ul aria-label={label} className="flex gap-1">
      {form.map((result, index) => (
        <li
          key={`${result}-${String(index)}`}
          className={`grid size-6 place-items-center rounded text-xs font-bold ${resultClasses[result] ?? 'bg-slate-200'}`}
        >
          {result}
        </li>
      ))}
    </ul>
  )
}
