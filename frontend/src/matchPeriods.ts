/** The kickoff dates behind the "When" filter on the Matches page. */

/** A date as YYYY-MM-DD, by the calendar the viewer is on. */
function isoDate(value: Date): string {
  const month = String(value.getMonth() + 1).padStart(2, '0')
  const day = String(value.getDate()).padStart(2, '0')
  return `${String(value.getFullYear())}-${month}-${day}`
}

function plusDays(value: Date, days: number): Date {
  return new Date(value.getFullYear(), value.getMonth(), value.getDate() + days)
}

/** The kickoff dates a period covers. An unknown period leaves them open. */
export function windowFor(
  period: string,
  now: Date = new Date(),
): { from_date?: string; to_date?: string } {
  if (period === 'today') {
    return { from_date: isoDate(now), to_date: isoDate(now) }
  }
  if (period === 'week') {
    return { from_date: isoDate(now), to_date: isoDate(plusDays(now, 6)) }
  }
  if (period === 'weekend') {
    const weekday = now.getDay() // Sunday is 0, Saturday is 6
    const sunday = plusDays(now, weekday === 0 ? 0 : 7 - weekday)
    const saturday = weekday === 0 ? now : plusDays(sunday, -1)
    return { from_date: isoDate(saturday), to_date: isoDate(sunday) }
  }
  return {}
}
