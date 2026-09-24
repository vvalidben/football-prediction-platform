/** Formatting is where "no data" quietly becomes "0" if nobody is watching. */
import { describe, expect, it } from 'vitest'
import { dateOnly, dateTime, decimal, percent, relative, scoreline } from './format.ts'

describe('percent', () => {
  it('shows one decimal by default', () => {
    expect(percent(0.5134)).toBe('51.3%')
  })

  it('can be asked for whole numbers', () => {
    expect(percent(0.5134, 0)).toBe('51%')
  })

  it('keeps the extremes honest', () => {
    expect(percent(0)).toBe('0.0%')
    expect(percent(1)).toBe('100.0%')
  })
})

describe('decimal', () => {
  it('formats a number to two places', () => {
    expect(decimal(1.8)).toBe('1.80')
  })

  it('shows a dash rather than a zero when there is no number', () => {
    expect(decimal(null)).toBe('—')
    expect(decimal(undefined)).toBe('—')
  })

  it('does not mistake zero for missing', () => {
    expect(decimal(0)).toBe('0.00')
  })
})

describe('scoreline', () => {
  it('joins the two scores', () => {
    expect(scoreline(2, 1)).toBe('2–1')
    expect(scoreline(0, 0)).toBe('0–0')
  })

  it('is a dash while a match has no result', () => {
    expect(scoreline(null, null)).toBe('—')
    expect(scoreline(2, null)).toBe('—')
  })
})

describe('dateTime and dateOnly', () => {
  it('render a timestamp', () => {
    const formatted = dateTime('2026-09-26T14:00:00Z')

    expect(formatted).toMatch(/2026/)
    expect(formatted).not.toBe('—')
  })

  it('are a dash when there is no timestamp', () => {
    expect(dateTime(null)).toBe('—')
    expect(dateOnly(null)).toBe('—')
  })
})

describe('relative', () => {
  const now = new Date('2026-09-23T12:00:00Z')

  it('says never when something has not happened', () => {
    expect(relative(null, now)).toBe('never')
  })

  it('counts backwards in the largest sensible unit', () => {
    expect(relative('2026-09-23T09:00:00Z', now)).toMatch(/3 hours ago/)
    expect(relative('2026-09-21T12:00:00Z', now)).toMatch(/2 days ago/)
    expect(relative('2026-09-23T11:45:00Z', now)).toMatch(/15 minutes ago/)
  })

  it('counts forwards for something still to come', () => {
    expect(relative('2026-09-26T12:00:00Z', now)).toMatch(/in 3 days/)
  })

  it('falls back to seconds for something that just happened', () => {
    expect(relative('2026-09-23T11:59:30Z', now)).toMatch(/30 seconds ago/)
  })
})
