import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { AsyncState } from '../hooks/useApi.ts'
import type { Explanation } from '../api/types.ts'
import * as fixtures from '../test/fixtures.ts'
import { ExplanationCard, ExplanationPanel } from './Explanation.tsx'

const loaded = (data: Explanation | null): AsyncState<Explanation | null> => ({
  kind: 'success',
  data,
  loadedAt: new Date(),
})

describe('ExplanationPanel', () => {
  it('names the features that contributed and which way each one pushed', () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    expect(
      screen.getByText(/These features contributed to the model.{1,3}s prediction\./),
    ).toBeInTheDocument()
    const chart = screen.getByRole('list', { name: 'Feature contributions to home win' })
    expect(within(chart).getByText('Home Elo + home advantage - away Elo')).toBeInTheDocument()
    expect(within(chart).getByText('+0.210')).toBeInTheDocument()
    expect(within(chart).getByText('-0.090')).toBeInTheDocument()
  })

  it('says what the model started from and where it arrived', () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    const summary = screen.getByText(/Starting from/)
    expect(summary).toHaveTextContent('41.0%')
    expect(summary).toHaveTextContent('51.0%')
    expect(summary).toHaveTextContent(/an average match in the model.{1,3}s training data/)
  })

  it('describes a contribution as the model pushing, never as a cause', () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    const chart = screen.getByRole('list', { name: 'Feature contributions to home win' })
    const described = within(chart).getAllByTitle(/pushed/)
    expect(described[0]).toHaveAttribute(
      'title',
      expect.stringContaining("pushed up the model's home win score"),
    )
    expect(screen.getByText(/They are not causes/)).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/caused|because the home team/i)
  })

  it('switches to another outcome', async () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    await userEvent.click(screen.getByRole('button', { name: 'Draw' }))

    expect(screen.getByRole('list', { name: 'Feature contributions to draw' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Draw' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('accounts for the features it did not list', () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    expect(screen.getByText(/other 8 features together contributed/)).toBeInTheDocument()
  })

  it('shows the method behind the figures', () => {
    render(<ExplanationPanel explanation={fixtures.explanation} />)

    expect(screen.getByText(/linear Shapley values/)).toBeInTheDocument()
  })

  it('reports a feature that had no value rather than showing a number for it', () => {
    const explanation: Explanation = {
      ...fixtures.explanation,
      targets: [
        {
          ...fixtures.explanation.targets[0],
          contributions: [{ ...fixtures.explanation.targets[0].contributions[0], value: null }],
        },
      ],
    }

    render(<ExplanationPanel explanation={explanation} />)

    expect(screen.getAllByTitle(/no value for this match/).length).toBeGreaterThan(0)
  })
})

describe('ExplanationCard', () => {
  it('shows a loading state while the contributions are computed', () => {
    render(<ExplanationCard state={{ kind: 'loading' }} onRetry={vi.fn()} />)

    expect(screen.getByRole('status')).toHaveTextContent('Working out the contributions')
  })

  it('states plainly when the model cannot be explained, without an error', () => {
    render(
      <ExplanationCard
        state={{
          kind: 'error',
          message: 'the historical baseline reads no features',
          status: 409,
        }}
        onRetry={vi.fn()}
      />,
    )

    expect(screen.getByText('This model cannot be explained')).toBeInTheDocument()
    expect(screen.getByText(/reads no features/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('reports a real failure as an error with a retry', async () => {
    const onRetry = vi.fn()
    render(
      <ExplanationCard
        state={{ kind: 'error', message: 'The backend is unreachable', status: null }}
        onRetry={onRetry}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(screen.getByRole('alert')).toHaveTextContent('The backend is unreachable')
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('waits for a prediction before explaining anything', () => {
    render(<ExplanationCard state={loaded(null)} onRetry={vi.fn()} />)

    expect(screen.getByText('No prediction to explain yet')).toBeInTheDocument()
  })

  it('renders the panel once the explanation arrives', () => {
    render(<ExplanationCard state={loaded(fixtures.explanation)} onRetry={vi.fn()} />)

    expect(
      screen.getByText(/These features contributed to the model.{1,3}s prediction\./),
    ).toBeInTheDocument()
  })
})
