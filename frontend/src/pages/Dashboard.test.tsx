import { screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { dateOnly } from '../format.ts'
import * as fixtures from '../test/fixtures.ts'
import { mockApi, page, renderAt } from '../test/render.tsx'
import { Dashboard } from './Dashboard.tsx'

const routes = {
  '/matches/upcoming': page([fixtures.match]),
  '/predictions': page([fixtures.storedPrediction]),
  '/models': [fixtures.modelVersion],
  '/data/freshness': fixtures.freshness,
  '/health': fixtures.health,
}

describe('Dashboard', () => {
  it('shows a loading state before anything has arrived', () => {
    mockApi(routes)
    renderAt(<Dashboard />)

    expect(screen.getAllByRole('status').length).toBeGreaterThan(0)
  })

  it('shows fixtures, predictions, models and freshness once loaded', async () => {
    mockApi(routes)
    renderAt(<Dashboard />)

    expect(await screen.findByText('Upcoming matches')).toBeInTheDocument() // freshness panel
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('Next matches')).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'Arsenal v Chelsea' })).toHaveAttribute(
      'href',
      '/matches/1',
    )
    expect(screen.getByText('1-1')).toBeInTheDocument()
    expect(screen.getByText(/H 51% · D 26% · A 23%/)).toBeInTheDocument()
    expect(await screen.findByRole('link', { name: 'elo 1.0.0' })).toBeInTheDocument()
  })

  it('says when there is nothing rather than inventing rows', async () => {
    mockApi({ ...routes, '/matches/upcoming': page([]), '/predictions': page([]), '/models': [] })
    renderAt(<Dashboard />)

    expect(await screen.findByText('No scheduled matches')).toBeInTheDocument()
    expect(await screen.findByText('No predictions yet')).toBeInTheDocument()
    expect(await screen.findByText('No active model')).toBeInTheDocument()
  })

  describe('when there are no upcoming fixtures', () => {
    /**
     * Imports have run; the data simply stops. Telling someone to run an import
     * sends them after a problem that is not there, and a hard-coded season range
     * would go stale the moment more data arrives — so the message is read from
     * the freshness panel this page has already loaded.
     */
    it('names the last match in the loaded data', async () => {
      mockApi({ ...routes, '/matches/upcoming': page([]) })
      renderAt(<Dashboard />)

      const hint = await screen.findByText(/The data loaded ends with the match on/)
      // Compared against the shared formatter rather than a literal: the rendered
      // date follows the viewer's locale.
      expect(hint).toHaveTextContent(dateOnly(fixtures.freshness.latest_finished_match_at))
      expect(hint).toHaveTextContent('nothing ahead to predict')
      expect(screen.queryByText(/once a provider import has run/)).not.toBeInTheDocument()
    })

    it('says nothing has been imported when the data is empty', async () => {
      mockApi({
        ...routes,
        '/matches/upcoming': page([]),
        '/data/freshness': { ...fixtures.freshness, latest_finished_match_at: null },
      })
      renderAt(<Dashboard />)

      expect(await screen.findByText(/No matches have been imported yet/)).toBeInTheDocument()
    })

    it('falls back to a plain statement when freshness cannot be read', async () => {
      mockApi({
        ...routes,
        '/matches/upcoming': page([]),
        '/data/freshness': new Response('', { status: 503 }),
      })
      renderAt(<Dashboard />)

      expect(
        await screen.findByText('Nothing is scheduled in the data loaded so far.'),
      ).toBeInTheDocument()
    })

    it('invents no date while the freshness panel is still loading', () => {
      mockApi({ ...routes, '/matches/upcoming': page([]) })
      renderAt(<Dashboard />)

      // Nothing has resolved yet: the hint must not claim a date it does not have.
      expect(screen.queryByText(/ends with the match on/)).not.toBeInTheDocument()
    })
  })

  it('reports a failing endpoint without hiding the rest of the page', async () => {
    mockApi({
      ...routes,
      '/data/freshness': new Response(JSON.stringify({ detail: 'database is down' }), {
        status: 503,
      }),
    })
    renderAt(<Dashboard />)

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('database is down')
    expect(alert).toHaveTextContent('HTTP 503')
    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Arsenal v Chelsea' })).toBeInTheDocument()
    })
  })
})
