import { screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import * as fixtures from '../test/fixtures.ts'
import { mockApi, renderAt } from '../test/render.tsx'
import { DataStatus } from './DataStatus.tsx'

describe('DataStatus', () => {
  it('shows synchronisation, coverage, providers and jobs', async () => {
    mockApi({ '/data/status': fixtures.dataStatus })
    renderAt(<DataStatus />, '/data')

    expect(await screen.findByText('Premier League')).toBeInTheDocument()
    expect(screen.getByText('1140 matches · 3 seasons')).toBeInTheDocument()
    expect(screen.getByText('1100 / 1140')).toBeInTheDocument()
    expect(screen.getByText('football-data.co.uk')).toBeInTheDocument()
    expect(screen.getByText(/12 runs ·/)).toBeInTheDocument()
    expect(screen.getByText('app.tasks.sync_results')).toBeInTheDocument()
    expect(screen.getByText('missing_lineup')).toBeInTheDocument()
  })

  it('shows the gaps as plainly as the coverage', async () => {
    mockApi({ '/data/status': fixtures.dataStatus })
    renderAt(<DataStatus />, '/data')

    expect(await screen.findByText('Matches without features')).toBeInTheDocument()
    expect(screen.getByText('20')).toBeInTheDocument()
    expect(screen.getByText('0 / 1100')).toBeInTheDocument() // no statistics imported
  })

  it('says nothing has been imported rather than showing an empty chart', async () => {
    mockApi({
      '/data/status': {
        ...fixtures.dataStatus,
        coverage: [],
        providers: [],
        ingestion: [],
        jobs: [],
        issues: [],
      },
    })
    renderAt(<DataStatus />, '/data')

    expect(await screen.findByText('Nothing imported yet')).toBeInTheDocument()
    expect(screen.getByText('No provider configured')).toBeInTheDocument()
    expect(screen.getByText('No import has run yet')).toBeInTheDocument()
    expect(screen.getByText('No open issues')).toBeInTheDocument()
  })

  it('reports a backend failure with a retry', async () => {
    mockApi({ '/data/status': new Response('', { status: 500 }) })
    renderAt(<DataStatus />, '/data')

    expect(await screen.findByRole('alert')).toHaveTextContent('Request failed (HTTP 500)')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })
})
