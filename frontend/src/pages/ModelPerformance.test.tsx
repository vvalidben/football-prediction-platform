import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import * as fixtures from '../test/fixtures.ts'
import { mockApi, renderAt, urlOf } from '../test/render.tsx'
import { ModelPerformance } from './ModelPerformance.tsx'

const second = {
  ...fixtures.modelVersion,
  id: 4,
  name: 'poisson',
  version: '2.0.0',
  is_active: false,
}

function renderPage(overrides: Record<string, unknown> = {}) {
  const fetchMock = mockApi({
    '/models': (request: { url: string }) => {
      if (/\/models\/\d+\/performance/.test(request.url))
        return overrides.performance ?? fixtures.performance
      if (/\/models\/\d+\/importance/.test(request.url))
        return overrides.importance ?? fixtures.importance
      return overrides.models ?? [fixtures.modelVersion, second]
    },
  })
  renderAt(<ModelPerformance />, '/models')
  return fetchMock
}

describe('ModelPerformance', () => {
  it('reports the metrics of the active version by default', async () => {
    renderPage()

    expect(await screen.findByText('elo 1.0.0')).toBeInTheDocument()
    expect(screen.getByText('52.0%')).toBeInTheDocument()
    expect(screen.getByText('0.9871')).toBeInTheDocument()
    expect(screen.getByText('0.1932')).toBeInTheDocument()
    expect(screen.getByText('120')).toBeInTheDocument()
  })

  it('draws calibration and history', async () => {
    renderPage()

    expect(
      await screen.findByRole('img', {
        name: 'Reliability diagram: predicted probability against observed frequency',
      }),
    ).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Monthly accuracy and log loss' })).toBeInTheDocument()
    expect(screen.getByText('2026-09: 60 matches, 54%')).toBeInTheDocument()
  })

  it('keeps training metrics separate from measured performance', async () => {
    renderPage()

    expect(await screen.findByText('Recorded at registration')).toBeInTheDocument()
    expect(screen.getByText('validation · log_loss')).toBeInTheDocument()
    expect(screen.getByText('1.0123')).toBeInTheDocument()
  })

  it('says nothing has been evaluated rather than showing zeros as results', async () => {
    renderPage({ performance: fixtures.emptyPerformance })

    expect(await screen.findByText('No evaluated predictions yet')).toBeInTheDocument()
    expect(screen.getByText('Nothing to calibrate yet')).toBeInTheDocument()
    expect(screen.getByText('No history yet')).toBeInTheDocument()
  })

  it('switches to another version', async () => {
    const fetchMock = renderPage()
    await screen.findByText('elo 1.0.0')

    await userEvent.selectOptions(screen.getByLabelText('Model'), '4')

    expect(
      fetchMock.mock.calls.some(([input]) => urlOf(input).includes('/models/4/performance')),
    ).toBe(true)
  })

  it('shows what the model relies on, as model behaviour', async () => {
    renderPage()

    expect(await screen.findByText('What this model relies on')).toBeInTheDocument()
    const chart = screen.getByRole('list', { name: 'Feature importance' })
    expect(within(chart).getByText('Home Elo + home advantage - away Elo')).toBeInTheDocument()
    expect(within(chart).getByText('48.0%')).toBeInTheDocument()
    expect(screen.getByText(/other 9 features account for/)).toBeInTheDocument()
    expect(screen.getByText(/not a ranking of what matters in football/)).toBeInTheDocument()
  })

  it('says when a model has no importance measure', async () => {
    renderPage({
      importance: new Response(
        JSON.stringify({ detail: 'no importance measure is available for historical_baseline' }),
        { status: 409 },
      ),
    })

    expect(await screen.findByText('This model has no importance measure')).toBeInTheDocument()
  })

  it('says when no version has been registered', async () => {
    renderPage({ models: [] })

    expect(await screen.findByText('No model versions registered')).toBeInTheDocument()
  })
})
