import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { mockApi, page, renderAt, urlOf } from '../test/render.tsx'
import { Teams } from './Teams.tsx'

const teams = [
  { id: 11, name: 'Arsenal', short_name: 'ARS', country: 'England' },
  { id: 12, name: 'Chelsea', short_name: 'CHE', country: 'England' },
]

describe('Teams', () => {
  it('lists the teams, each linking to its own page', async () => {
    mockApi({ '/teams': page(teams) })
    renderAt(<Teams />, '/teams')

    expect(await screen.findByRole('link', { name: /Arsenal/ })).toHaveAttribute(
      'href',
      '/teams/11',
    )
    expect(screen.getByRole('link', { name: /Chelsea/ })).toHaveAttribute('href', '/teams/12')
  })

  it('names the page in the browser tab', async () => {
    mockApi({ '/teams': page(teams) })
    renderAt(<Teams />, '/teams')
    await screen.findByRole('link', { name: /Arsenal/ })

    expect(document.title).toBe('Teams · Football Prediction Platform')
  })

  it('searches by name', async () => {
    const fetchMock = mockApi({ '/teams': page(teams) })
    renderAt(<Teams />, '/teams')
    await screen.findByRole('link', { name: /Arsenal/ })

    await userEvent.type(screen.getByRole('searchbox', { name: 'Search teams' }), 'ars')

    await waitFor(() => {
      expect(fetchMock.mock.calls.map(([input]) => urlOf(input)).at(-1)).toContain('search=ars')
    })
  })

  it('says when a search matches nothing', async () => {
    mockApi({ '/teams': page([]) })
    renderAt(<Teams />, '/teams')
    await userEvent.type(screen.getByRole('searchbox', { name: 'Search teams' }), 'zzz')

    expect(await screen.findByText(/No team matches/)).toBeInTheDocument()
  })

  it('says when nothing has been imported at all', async () => {
    mockApi({ '/teams': page([]) })
    renderAt(<Teams />, '/teams')

    expect(await screen.findByText('No teams imported yet')).toBeInTheDocument()
    expect(screen.getByText(/first provider import/)).toBeInTheDocument()
  })

  it('reports a backend failure with a retry', async () => {
    mockApi({
      '/teams': new Response(JSON.stringify({ detail: 'database is down' }), { status: 503 }),
    })
    renderAt(<Teams />, '/teams')

    expect(await screen.findByRole('alert')).toHaveTextContent('database is down')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })
})
