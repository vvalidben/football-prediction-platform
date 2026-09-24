/**
 * The shared API client.
 *
 * Every page loads its data through `apiGet`, so failures look the same
 * everywhere: an {@link ApiError} with a message worth showing to a person.
 * Responses are returned as-is; each page's types describe what it reads.
 */
import { config } from '../config.ts'

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export interface Page<T> {
  total: number
  limit: number
  offset: number
  items: T[]
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

interface Options {
  method: 'GET' | 'POST'
  headers?: Record<string, string>
  body?: string
}

async function request<T>(path: string, init: Options, signal?: AbortSignal): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${config.apiBaseUrl}${path}`, {
      method: init.method,
      body: init.body,
      headers: { Accept: 'application/json', ...init.headers },
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('The backend is unreachable')
  }

  let body: unknown = null
  const text = await response.text()
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      throw new ApiError('The backend returned a response that is not JSON', response.status)
    }
  }

  if (!response.ok) {
    throw new ApiError(
      detailOf(body) ?? `Request failed (HTTP ${String(response.status)})`,
      response.status,
    )
  }
  return body as T
}

/** The `detail` a FastAPI error carries, when it is a readable string. */
function detailOf(body: unknown): string | null {
  if (typeof body !== 'object' || body === null) return null
  const detail = (body as { detail?: unknown }).detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        typeof item === 'object' &&
        item !== null &&
        typeof (item as { msg?: unknown }).msg === 'string'
          ? (item as { msg: string }).msg
          : null,
      )
      .filter((message): message is string => message !== null)
    if (messages.length > 0) return messages.join('; ')
  }
  return null
}

export function apiGet<T>(
  path: string,
  params: Record<string, string | number | boolean | undefined> = {},
  signal?: AbortSignal,
): Promise<T> {
  return request<T>(`${path}${query(params)}`, { method: 'GET' }, signal)
}

export function apiPost<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(
    path,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    signal,
  )
}
