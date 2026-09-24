import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import type { HistoryPage } from '../api/types.ts'
import * as fixtures from '../test/fixtures.ts'
import { mockApi, page, renderAt, urlOf } from '../test/render.tsx'
import { History } from './History.tsx'

function routes(overrides: Record<string, unknown> = {}) {
  return {
    '/predictions/history': fixtures.historyPage,
    '/competitions': page([{ id: 7, name: 'Premier League', country: 'England' }]),
    '/seasons': page([
      {
        id: 3,
        competition_id: 7,
        competition: 'Premier League',
        name: '2026/2027',
        start_date: '2026-08-01',
        end_date: '2027-05-31',
        is_current: true,
      },
    ]),
    '/teams': page([{ id: 11, name: 'Arsenal', short_name: 'ARS', country: 'England' }]),
    '/models': [fixtures.modelVersion],
    ...overrides,
  }
}

const renderPage = (overrides: Record<string, unknown> = {}, path = '/history') => {
  const fetchMock = mockApi(routes(overrides))
  renderAt(<History />, path)
  return fetchMock
}

const historyUrls = (fetchMock: ReturnType<typeof mockApi>) =>
  fetchMock.mock.calls.map(([input]) => urlOf(input)).filter((url) => url.includes('/history'))

describe('History', () => {
  it('shows what was predicted beside what happened', async () => {
    renderPage()

    expect(await screen.findByRole('heading', { name: 'Prediction history' })).toBeInTheDocument()
    const table = await screen.findByRole('table')
    const scored = within(table).getAllByRole('row')[1]
    expect(within(scored).getByRole('link', { name: 'Arsenal v Chelsea' })).toBeInTheDocument()
    expect(within(scored).getByText('2–1')).toBeInTheDocument() // predicted
    expect(within(scored).getByText('1–1')).toBeInTheDocument() // actual
    expect(within(scored).getByText('Miss')).toBeInTheDocument()
    expect(within(scored).getByText(/log loss 1\.204/)).toBeInTheDocument()
    expect(within(scored).getByText(/goals \+0\.35 \/ -0\.10/)).toBeInTheDocument()
    expect(within(scored).getByRole('link', { name: 'elo 1.0.0' })).toBeInTheDocument()
  })

  it('shows the probabilities that were served', async () => {
    renderPage()

    const table = await screen.findByRole('table')
    const scored = within(table).getAllByRole('row')[1]
    expect(within(scored).getByTitle('Home: 51.0%')).toHaveTextContent('51%')
    expect(within(scored).getByTitle('Draw: 26.0%')).toHaveTextContent('26%')
    expect(within(scored).getByTitle('Away: 23.0%')).toHaveTextContent('23%')
  })

  it('lists a prediction whose match has not been played, with no result or error', async () => {
    renderPage()

    const table = await screen.findByRole('table')
    const pending = within(table).getAllByRole('row')[2]
    expect(within(pending).getByText('scheduled')).toBeInTheDocument()
    expect(within(pending).getByText('—')).toBeInTheDocument() // no error to report
    expect(within(pending).getByTitle('Home: 51.0%')).toBeInTheDocument() // still predicted
  })

  it('marks a played match that has not been scored yet', async () => {
    const played = {
      ...fixtures.historyPage,
      items: [
        {
          ...fixtures.historyPage.items[1],
          status: 'finished',
          home_score: 2,
          away_score: 0,
          actual_outcome: 'home' as const,
        },
      ],
    }
    renderPage({ '/predictions/history': played })

    const table = await screen.findByRole('table')
    expect(within(table).getByText('Not scored')).toBeInTheDocument()
    expect(within(table).getByText('2–0')).toBeInTheDocument()
  })

  it('summarises every row matching the filters, not just the page', async () => {
    renderPage()

    expect(await screen.findByText('Summary')).toBeInTheDocument()
    expect(screen.getByText('40')).toBeInTheDocument() // predictions
    expect(screen.getByText('32 scored · 8 pending')).toBeInTheDocument()
    expect(screen.getByText('53.1%')).toBeInTheDocument() // accuracy
    expect(screen.getByText('1.0210')).toBeInTheDocument() // log loss
    expect(screen.getByText('0.88')).toBeInTheDocument() // mean absolute goal error
  })

  it('filters by competition, season, team, model, date and status', async () => {
    const fetchMock = renderPage()
    await screen.findByRole('table')

    await userEvent.selectOptions(screen.getByLabelText('Competition'), '7')
    await waitFor(() => {
      expect(historyUrls(fetchMock).at(-1)).toContain('competition_id=7')
    })

    await userEvent.selectOptions(screen.getByLabelText('Season'), '3')
    await userEvent.selectOptions(screen.getByLabelText('Team'), '11')
    await userEvent.selectOptions(screen.getByLabelText('Model'), 'elo')
    await userEvent.selectOptions(screen.getByLabelText('Status'), 'evaluated')

    await waitFor(() => {
      const latest = historyUrls(fetchMock).at(-1) ?? ''
      expect(latest).toContain('season_id=3')
      expect(latest).toContain('team_id=11')
      expect(latest).toContain('model_name=elo')
      expect(latest).toContain('evaluated=true')
    })
  })

  it('keeps the filters in the URL so a view can be linked to', async () => {
    const fetchMock = renderPage({}, '/history?competition=7&team=11&status=pending')

    await waitFor(() => {
      const latest = historyUrls(fetchMock).at(-1) ?? ''
      expect(latest).toContain('competition_id=7')
      expect(latest).toContain('team_id=11')
      expect(latest).toContain('evaluated=false')
    })
    expect(await screen.findByLabelText('Status')).toHaveValue('pending')
  })

  it('clears the filters again', async () => {
    const fetchMock = renderPage({}, '/history?competition=7')
    await screen.findByRole('table')

    await userEvent.click(screen.getByRole('button', { name: 'Clear filters' }))

    await waitFor(() => {
      expect(historyUrls(fetchMock).at(-1)).not.toContain('competition_id')
    })
  })

  it('pages through a long history', async () => {
    const long: HistoryPage = { ...fixtures.historyPage, total: 60 }
    const fetchMock = renderPage({ '/predictions/history': long })
    await screen.findByRole('table')

    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => {
      expect(historyUrls(fetchMock).at(-1)).toContain('offset=25')
    })
  })

  it('says when nothing matches rather than showing an empty table', async () => {
    renderPage({
      '/predictions/history': {
        total: 0,
        limit: 25,
        offset: 0,
        items: [],
        summary: {
          predictions: 0,
          evaluated: 0,
          pending: 0,
          accuracy: null,
          exact_score_accuracy: null,
          btts_accuracy: null,
          log_loss: null,
          brier_score: null,
          ranked_probability_score: null,
          mean_home_goals_error: null,
          mean_away_goals_error: null,
          mean_absolute_goals_error: null,
          first_kickoff_at: null,
          last_kickoff_at: null,
        },
      },
    })

    expect(await screen.findByText('No predictions match these filters')).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0) // no invented averages
  })

  it('reports a backend failure with a retry', async () => {
    renderPage({
      '/predictions/history': new Response(JSON.stringify({ detail: 'database is down' }), {
        status: 503,
      }),
    })

    expect(await screen.findByRole('alert')).toHaveTextContent('database is down')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })
})
