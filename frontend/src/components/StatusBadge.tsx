type Tone = 'ok' | 'warning' | 'error' | 'pending'

const toneClasses: Record<Tone, string> = {
  ok: 'bg-pitch-50 text-pitch-700 ring-pitch-500/30',
  warning: 'bg-amber-50 text-amber-800 ring-amber-500/30',
  error: 'bg-red-50 text-red-700 ring-red-500/30',
  pending: 'bg-slate-100 text-slate-600 ring-slate-400/30',
}

const dotClasses: Record<Tone, string> = {
  ok: 'bg-pitch-500',
  warning: 'bg-amber-500',
  error: 'bg-red-500',
  pending: 'bg-slate-400',
}

interface StatusBadgeProps {
  tone: Tone
  label: string
}

export function StatusBadge({ tone, label }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${toneClasses[tone]}`}
    >
      <span aria-hidden="true" className={`size-1.5 rounded-full ${dotClasses[tone]}`} />
      {label}
    </span>
  )
}
