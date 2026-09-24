import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { HealthResponse } from '../api/health.ts'
import { HealthPanel } from './HealthPanel.tsx'

const healthyReport: HealthResponse = {
  status: 'ok',
  service: 'football-prediction-api',
  version: '0.1.0',
  environment: 'test',
  checks: {
    database: { status: 'ok', latency_ms: 2.25, error: null },
    redis: { status: 'ok', latency_ms: 0.5, error: null },
  },
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('HealthPanel', () => {
  it('shows every dependency as healthy when the backend is up', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(healthyReport))

    render(<HealthPanel />)

    expect(await screen.findByText('Operational')).toBeInTheDocument()
    expect(screen.getByText('PostgreSQL')).toBeInTheDocument()
    expect(screen.getByText('Redis')).toBeInTheDocument()
    expect(screen.getAllByText('Healthy')).toHaveLength(2)
    expect(screen.getByText('2.3 ms')).toBeInTheDocument()
  })

  it('shows a degraded dependency with its error', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse(
        {
          ...healthyReport,
          status: 'degraded',
          checks: {
            ...healthyReport.checks,
            database: { status: 'error', latency_ms: null, error: 'OperationalError' },
          },
        },
        503,
      ),
    )

    render(<HealthPanel />)

    expect(await screen.findByText('Degraded')).toBeInTheDocument()
    expect(screen.getByText('Unavailable')).toBeInTheDocument()
    expect(screen.getByText('OperationalError')).toBeInTheDocument()
  })

  it('shows the backend as offline when it cannot be reached, and recovers on refresh', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValue(jsonResponse(healthyReport))

    render(<HealthPanel />)

    expect(await screen.findByText('Offline')).toBeInTheDocument()
    expect(screen.getByText('Backend is unreachable')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }))

    expect(await screen.findByText('Operational')).toBeInTheDocument()
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })
})
