import { config } from '../config.ts'

export type CheckStatus = 'ok' | 'error'
export type OverallStatus = 'ok' | 'degraded'

export interface DependencyCheck {
  status: CheckStatus
  latency_ms: number | null
  error: string | null
}

export interface HealthResponse {
  status: OverallStatus
  service: string
  version: string
  environment: string
  checks: Record<string, DependencyCheck>
}

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isDependencyCheck(value: unknown): value is DependencyCheck {
  if (typeof value !== 'object' || value === null) return false
  const check = value as Record<string, unknown>
  return (
    (check.status === 'ok' || check.status === 'error') &&
    (check.latency_ms === null || typeof check.latency_ms === 'number') &&
    (check.error === null || typeof check.error === 'string')
  )
}

export function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== 'object' || value === null) return false
  const body = value as Record<string, unknown>
  return (
    (body.status === 'ok' || body.status === 'degraded') &&
    typeof body.service === 'string' &&
    typeof body.version === 'string' &&
    typeof body.environment === 'string' &&
    typeof body.checks === 'object' &&
    body.checks !== null &&
    Object.values(body.checks).every(isDependencyCheck)
  )
}

/**
 * Fetch the backend health report.
 *
 * The backend answers 200 when every dependency is healthy and 503 when one is
 * degraded; both carry a valid report, so both resolve. Anything else rejects
 * with an {@link ApiError}.
 */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  let response: Response
  try {
    response = await fetch(`${config.apiBaseUrl}/health`, {
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('Backend is unreachable')
  }

  if (response.status !== 200 && response.status !== 503) {
    throw new ApiError(
      `Unexpected response from backend (HTTP ${String(response.status)})`,
      response.status,
    )
  }

  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new ApiError('Backend returned a non-JSON response', response.status)
  }

  if (!isHealthResponse(body)) {
    throw new ApiError('Backend returned a malformed health report', response.status)
  }
  return body
}
