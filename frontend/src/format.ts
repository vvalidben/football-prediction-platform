/** Shared formatting, so dates and numbers read the same on every page. */

export const percent = (value: number, digits = 1): string => `${(value * 100).toFixed(digits)}%`

export const decimal = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? '—' : value.toFixed(digits)

export function dateTime(value: string | null): string {
  if (!value) return '—'
  return new Date(value).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

export function dateOnly(value: string | null): string {
  if (!value) return '—'
  return new Date(value).toLocaleDateString(undefined, { dateStyle: 'medium' })
}

/** "3 hours ago", "in 2 days" — how current something is, in plain words. */
export function relative(value: string | null, now: Date = new Date()): string {
  if (!value) return 'never'
  const difference = new Date(value).getTime() - now.getTime()
  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ['day', 86_400_000],
    ['hour', 3_600_000],
    ['minute', 60_000],
  ]
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  for (const [unit, size] of units) {
    if (Math.abs(difference) >= size) return formatter.format(Math.round(difference / size), unit)
  }
  return formatter.format(Math.round(difference / 1000), 'second')
}

export const scoreline = (home: number | null, away: number | null): string =>
  home === null || away === null ? '—' : `${String(home)}–${String(away)}`
