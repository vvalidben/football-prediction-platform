import { describe, expect, it, vi } from 'vitest'
import { ApiError, fetchHealth, isHealthResponse, type HealthResponse } from './health.ts'

const healthyReport: HealthResponse = {
  status: 'ok',
  service: 'football-prediction-api',
  version: '0.1.0',
  environment: 'test',
  checks: {
    database: { status: 'ok', latency_ms: 1.5, error: null },
    redis: { status: 'ok', latency_ms: 0.4, error: null },
  },
}

function mockFetch(response: Response | Error) {
  return vi
    .spyOn(globalThis, 'fetch')
    .mockImplementation(() =>
      response instanceof Error ? Promise.reject(response) : Promise.resolve(response),
    )
}

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('isHealthResponse', () => {
  it('accepts a well-formed report', () => {
    expect(isHealthResponse(healthyReport)).toBe(true)
  })

  it('rejects a report with an invalid check', () => {
    expect(
      isHealthResponse({ ...healthyReport, checks: { database: { status: 'unknown' } } }),
    ).toBe(false)
  })
})

describe('fetchHealth', () => {
  it('calls the health endpoint under the API base URL', async () => {
    const fetchSpy = mockFetch(jsonResponse(healthyReport, 200))

    await expect(fetchHealth()).resolves.toEqual(healthyReport)
    expect(fetchSpy).toHaveBeenCalledWith('/api/health', expect.any(Object))
  })

  it('resolves a degraded report returned with HTTP 503', async () => {
    const degraded: HealthResponse = {
      ...healthyReport,
      status: 'degraded',
      checks: {
        ...healthyReport.checks,
        redis: { status: 'error', latency_ms: null, error: 'ConnectionError' },
      },
    }
    mockFetch(jsonResponse(degraded, 503))

    await expect(fetchHealth()).resolves.toEqual(degraded)
  })

  it('rejects with ApiError when the backend is unreachable', async () => {
    mockFetch(new TypeError('Failed to fetch'))

    await expect(fetchHealth()).rejects.toThrow(new ApiError('Backend is unreachable'))
  })

  it('rejects with ApiError on unexpected status codes', async () => {
    mockFetch(new Response('Bad gateway', { status: 502 }))

    await expect(fetchHealth()).rejects.toMatchObject({ name: 'ApiError', status: 502 })
  })

  it('rejects with ApiError on a malformed body', async () => {
    mockFetch(jsonResponse({ status: 'ok' }, 200))

    await expect(fetchHealth()).rejects.toThrow('malformed health report')
  })
})
