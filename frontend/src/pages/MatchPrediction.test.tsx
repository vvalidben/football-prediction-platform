import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import * as fixtures from '../test/fixtures.ts'
import { bodyOf, mockApi, renderAt, urlOf, type ApiCall } from '../test/render.tsx'
import { MatchPrediction } from './MatchPrediction.tsx'

const chelsea = {
  ...fixtures.teamDetail,
  team: { id: 12, name: 'Chelsea', short_name: 'CHE', country: 'England' },
}

const routes = {
  '/matches': fixtures.matchDetail,
  '/predictions/match': fixtures.prediction,
  '/predictions': fixtures.explanation,
  '/teams': ({ url }: ApiCall) => (url.includes('/teams/12') ? chelsea : fixtures.teamDetail),
}

function renderPage(overrides: Record<string, unknown> = {}) {
  const fetchMock = mockApi({ ...routes, ...overrides })
  renderAt(
    <Routes>
      <Route path="/matches/:matchId" element={<MatchPrediction />} />
    </Routes>,
    '/matches/1',
  )
  return fetchMock
}

const measuredModel = {
  ...fixtures.modelVersion,
  metrics: {
    test_by_competition: {
      '7': {
        matches: 380,
        accuracy: 0.476,
        log_loss: 1.04,
        exact_score_accuracy: 0.129,
        home_win_rate: 0.426,
        start: '2025-08-15T19:00:00+00:00',
        end: '2026-05-24T15:00:00+00:00',
      },
    },
  },
}

describe('reliability of the model in this league', () => {
  it('shows how the model did on matches it was not trained on', async () => {
    renderPage({ '/models': [measuredModel] })

    const card = (
      await screen.findByRole('heading', { name: /How reliable is this model in Premier League/ })
    ).closest('section')
    expect(card).not.toBeNull()
    const scope = within(card as HTMLElement)
    expect(scope.getByText('47.6%')).toBeInTheDocument()
    expect(scope.getByText('12.9%')).toBeInTheDocument()
    expect(scope.getByText('380 matches')).toBeInTheDocument()
    expect(scope.getByText(/Always picking the home team: 42.6%/)).toBeInTheDocument()
    expect(scope.getByText(/not been trained on/)).toBeInTheDocument()
  })

  it('stays out of the way when the league was not measured', async () => {
    renderPage({ '/models': [fixtures.modelVersion] })

    await screen.findByText('Most probable score')

    expect(screen.queryByText(/How reliable is this model/)).not.toBeInTheDocument()
  })

  it('does not disturb the prediction when the models cannot be read', async () => {
    renderPage({
      '/models': new Response(JSON.stringify({ detail: 'down' }), { status: 503 }),
    })

    expect(await screen.findByText('Most probable score')).toBeInTheDocument()
    expect(screen.queryByText(/How reliable is this model/)).not.toBeInTheDocument()
  })
})

describe('MatchPrediction', () => {
  it('links back to the upcoming matches of its league', async () => {
    renderPage()

    expect(await screen.findByRole('link', { name: /All Premier League matches/ })).toHaveAttribute(
      'href',
      '/matches?league=7',
    )
  })

  it('shows the teams, the predicted score and the expected goals', async () => {
    renderPage()

    expect(
      await screen.findByRole('heading', { name: 'Arsenal v Chelsea', level: 1 }),
    ).toBeInTheDocument()
    const headline = (await screen.findByText('Most probable score')).closest('div')
    expect(headline).not.toBeNull()
    expect(within(headline as HTMLElement).getByText('1–1')).toBeInTheDocument()
    expect(screen.getByText('1.62 – 1.08')).toBeInTheDocument()
    expect(screen.getByText('57.0%')).toBeInTheDocument() // both teams to score
  })

  it('derives every market from the returned score grid', async () => {
    renderPage()

    const result = await screen.findByRole('list', { name: 'Result probabilities' })
    expect(within(result).getByText('51.0%')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Total goals distribution' })).toBeInTheDocument()
    expect(screen.getByText('2.5 goals')).toBeInTheDocument()
    expect(screen.getByText('over 52.0% · under 48.0%')).toBeInTheDocument()
    expect(
      screen.getByRole('table', { name: /Probability of each exact score/ }),
    ).toBeInTheDocument()
  })

  it('shows recent form for both teams', async () => {
    renderPage()

    expect(await screen.findByLabelText('Arsenal recent form')).toBeInTheDocument()
    expect(screen.getByLabelText('Chelsea recent form')).toBeInTheDocument()
    expect(screen.getAllByText('1.80').length).toBe(2) // one per team
  })

  it('says xG is not provided instead of showing a number', async () => {
    renderPage()

    expect((await screen.findAllByText('not provided')).length).toBeGreaterThan(0)
  })

  it('asks for a fresh prediction when recompute is pressed', async () => {
    const fetchMock = renderPage()
    await screen.findByText('Most probable score')

    await userEvent.click(screen.getByRole('button', { name: 'Recompute' }))

    const bodies = fetchMock.mock.calls
      .filter(([input]) => urlOf(input).includes('/predictions/match'))
      .map(([, init]) => bodyOf(init) as { refresh: boolean })
    expect(bodies.at(-1)?.refresh).toBe(true)
  })

  it('shows which features contributed to the prediction', async () => {
    renderPage()

    expect(await screen.findByText('Why this prediction')).toBeInTheDocument()
    const chart = await screen.findByRole('list', {
      name: 'Feature contributions to home win',
    })
    expect(within(chart).getByText('Home Elo + home advantage - away Elo')).toBeInTheDocument()
    expect(screen.getByText(/They are not causes/)).toBeInTheDocument()
  })

  it('says so when the model behind the prediction cannot be explained', async () => {
    renderPage({
      '/predictions': new Response(
        JSON.stringify({ detail: 'the historical baseline reads no features' }),
        { status: 409 },
      ),
    })

    expect(await screen.findByText('This model cannot be explained')).toBeInTheDocument()
    expect(screen.getByText(/reads no features/)).toBeInTheDocument()
  })

  it('reports a refused prediction', async () => {
    renderPage({
      '/predictions/match': new Response(
        JSON.stringify({ detail: 'features could not be computed' }),
        { status: 422 },
      ),
    })

    expect(await screen.findByRole('alert')).toHaveTextContent('features could not be computed')
  })
})
