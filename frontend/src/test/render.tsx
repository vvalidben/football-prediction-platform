import { render, type RenderResult } from '@testing-library/react'
import type { ReactElement } from 'react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'

export function renderAt(ui: ReactElement, path = '/'): RenderResult {
  return render(<MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter>)
}

export interface ApiCall {
  url: string
  body: unknown
}

type Handler = ((request: ApiCall) => unknown) | object

/** The URL of a recorded `fetch` call, whichever form the caller used. */
export function urlOf(input: RequestInfo | URL): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.href
  return input.url
}

/** The JSON a recorded `fetch` call sent, or null when it sent nothing. */
export function bodyOf(init: RequestInit | undefined): unknown {
  const body = init?.body
  return typeof body === 'string' ? JSON.parse(body) : null
}

/**
 * Answer API calls from a table keyed by path prefix, so a test states only the
 * responses it cares about. An unlisted path returns a 404 rather than something
 * invented, which shows up as a visible error in the page under test.
 */
export function mockApi(routes: Record<string, Handler>) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = urlOf(input)
    const path = url.replace(/^\/api/, '').split('?')[0]
    const key = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((candidate) => path === candidate || path.startsWith(`${candidate}/`))
    if (key === undefined) {
      return Promise.resolve(
        new Response(JSON.stringify({ detail: `no mock for ${path}` }), { status: 404 }),
      )
    }
    const handler = routes[key]
    const body =
      typeof handler === 'function'
        ? (handler as (request: ApiCall) => unknown)({ url, body: bodyOf(init) })
        : handler
    if (body instanceof Response) return Promise.resolve(body)
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
  })
}

export const page = <T,>(items: T[]) => ({
  total: items.length,
  limit: 20,
  offset: 0,
  items,
})
