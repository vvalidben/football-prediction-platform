/** Routing, navigation and the boundary around every page. */
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import App from './App.tsx'
import * as fixtures from './test/fixtures.ts'
import { mockApi, page, renderAt } from './test/render.tsx'

function routes() {
  return {
    '/matches': page([fixtures.match]),
    '/predictions': page([fixtures.storedPrediction]),
    '/models': [fixtures.modelVersion],
    '/data': fixtures.dataStatus,
    '/health': fixtures.health,
    '/competitions': page([{ id: 7, name: 'Premier League', country: 'England' }]),
    '/seasons': page([]),
    '/teams': page([{ id: 11, name: 'Arsenal', short_name: 'ARS', country: 'England' }]),
  }
}

describe('App', () => {
  it('shows the dashboard at the root', async () => {
    mockApi(routes())
    renderAt(<App />, '/')

    expect(await screen.findByRole('heading', { name: 'Dashboard', level: 1 })).toBeInTheDocument()
  })

  it('offers every page in the navigation', () => {
    mockApi(routes())
    renderAt(<App />, '/')

    const nav = screen.getByRole('navigation', { name: 'Main' })
    for (const label of ['Dashboard', 'History', 'Teams', 'Models', 'Data']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
    expect(nav).toBeInTheDocument()
  })

  it('navigates without reloading the application', async () => {
    mockApi(routes())
    renderAt(<App />, '/')
    await screen.findByRole('heading', { name: 'Dashboard', level: 1 })

    await userEvent.click(screen.getByRole('link', { name: 'Teams' }))

    expect(await screen.findByRole('heading', { name: 'Teams', level: 1 })).toBeInTheDocument()
  })

  it('marks the page you are on', async () => {
    mockApi(routes())
    renderAt(<App />, '/data')

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Data' })).toHaveAttribute('aria-current', 'page')
    })
    expect(screen.getByRole('link', { name: 'History' })).not.toHaveAttribute('aria-current')
  })

  it('sends an unknown path back to the dashboard', async () => {
    mockApi(routes())
    renderAt(<App />, '/somewhere-that-does-not-exist')

    expect(await screen.findByRole('heading', { name: 'Dashboard', level: 1 })).toBeInTheDocument()
  })

  it('keeps the shell usable when an endpoint fails', async () => {
    mockApi({ ...routes(), '/data': new Response('', { status: 500 }) })
    renderAt(<App />, '/data')

    expect(await screen.findByRole('alert')).toHaveTextContent('Request failed (HTTP 500)')
    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
  })

  it('states what the figures are, in the footer', () => {
    mockApi(routes())
    renderAt(<App />, '/')

    expect(screen.getByText(/Nothing is filled in when data is missing/)).toBeInTheDocument()
  })
})
