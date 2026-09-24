import { describe, expect, it, vi } from 'vitest'
import { bodyOf, mockApi, urlOf } from '../test/render.tsx'
import { apiGet, apiPost, ApiError } from './client.ts'

describe('apiGet', () => {
  it('sends only the parameters that have a value', async () => {
    const fetchMock = mockApi({ '/matches': { total: 0, limit: 20, offset: 0, items: [] } })

    await apiGet('/matches', { limit: 20, status: undefined, team_id: '' })

    const url = urlOf(fetchMock.mock.calls[0][0])
    expect(url).toContain('limit=20')
    expect(url).not.toContain('status')
    expect(url).not.toContain('team_id')
  })

  it('raises the detail a FastAPI error carries', async () => {
    mockApi({
      '/teams': new Response(JSON.stringify({ detail: 'team 9 was not found' }), { status: 404 }),
    })

    await expect(apiGet('/teams/9')).rejects.toMatchObject({
      message: 'team 9 was not found',
      status: 404,
    })
  })

  it('joins the messages of a validation error', async () => {
    mockApi({
      '/matches': new Response(
        JSON.stringify({ detail: [{ msg: 'limit must be positive' }, { msg: 'unknown status' }] }),
        { status: 422 },
      ),
    })

    await expect(apiGet('/matches')).rejects.toThrow('limit must be positive; unknown status')
  })

  it('says the backend is unreachable when the request cannot be made', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('network error'))

    await expect(apiGet('/health')).rejects.toThrow('The backend is unreachable')
    await expect(apiGet('/health')).rejects.toBeInstanceOf(ApiError)
  })

  it('falls back to the status code when the body carries no detail', async () => {
    mockApi({ '/data/status': new Response('', { status: 500 }) })

    await expect(apiGet('/data/status')).rejects.toThrow('Request failed (HTTP 500)')
  })
})

describe('apiPost', () => {
  it('sends the body as JSON', async () => {
    const fetchMock = mockApi({ '/predictions/match': { id: 1 } })

    await apiPost('/predictions/match', { match_id: 4, refresh: true })

    const init = fetchMock.mock.calls[0][1]
    expect(init?.method).toBe('POST')
    expect(bodyOf(init)).toEqual({ match_id: 4, refresh: true })
  })
})
