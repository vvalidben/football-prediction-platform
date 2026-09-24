import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { AsyncState } from '../hooks/useApi.ts'
import { Async } from './Async.tsx'
import { Card, Empty, ErrorState, Loading } from './states.tsx'

describe('Loading', () => {
  it('announces itself to assistive technology rather than only spinning', () => {
    render(<Loading />)

    const status = screen.getByRole('status')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveTextContent('Loading…')
  })

  it('says what is being loaded when the page knows', () => {
    render(<Loading label="Loading fixtures…" />)

    expect(screen.getByRole('status')).toHaveTextContent('Loading fixtures…')
  })
})

describe('ErrorState', () => {
  it('is an alert carrying the backend message', () => {
    render(<ErrorState message="database is down" status={503} />)

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('database is down')
    expect(alert).toHaveTextContent('(HTTP 503)')
  })

  it('omits a status code it was not given', () => {
    render(<ErrorState message="The backend is unreachable" status={null} />)

    expect(screen.getByRole('alert')).not.toHaveTextContent('HTTP')
  })

  it('offers a retry only when there is something to retry', async () => {
    const onRetry = vi.fn()
    const { rerender } = render(<ErrorState message="failed" />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()

    rerender(<ErrorState message="failed" onRetry={onRetry} />)
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(onRetry).toHaveBeenCalledOnce()
  })
})

describe('Empty', () => {
  it('states what is missing instead of showing a figure', () => {
    render(<Empty title="No predictions yet" hint="Ask for one from a match page." />)

    expect(screen.getByText('No predictions yet')).toBeInTheDocument()
    expect(screen.getByText('Ask for one from a match page.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument() // nothing has failed
  })

  it('works without a hint', () => {
    render(<Empty title="No open issues" />)

    expect(screen.getByText('No open issues')).toBeInTheDocument()
  })
})

describe('Card', () => {
  it('gives its content a heading', () => {
    render(
      <Card title="Coverage">
        <p>body</p>
      </Card>,
    )

    expect(screen.getByRole('heading', { name: 'Coverage', level: 2 })).toBeInTheDocument()
    expect(screen.getByText('body')).toBeInTheDocument()
  })

  it('can carry an action beside the title', () => {
    render(
      <Card title="Models" action={<button type="button">Refresh</button>}>
        <p>body</p>
      </Card>,
    )

    expect(screen.getByRole('button', { name: 'Refresh' })).toBeInTheDocument()
  })

  it('renders nothing above the content when it has no title', () => {
    render(
      <Card>
        <p>plain</p>
      </Card>,
    )

    expect(screen.queryByRole('heading')).not.toBeInTheDocument()
    expect(screen.getByText('plain')).toBeInTheDocument()
  })
})

describe('Async', () => {
  const loaded = (data: string): AsyncState<string> => ({
    kind: 'success',
    data,
    loadedAt: new Date(),
  })

  it('shows the loading state first', () => {
    render(
      <Async state={{ kind: 'loading' }} loadingLabel="Loading teams…">
        {(data: string) => <p>{data}</p>}
      </Async>,
    )

    expect(screen.getByRole('status')).toHaveTextContent('Loading teams…')
  })

  it('shows the error state, with the retry wired through', async () => {
    const onRetry = vi.fn()
    render(
      <Async state={{ kind: 'error', message: 'nope', status: 500 }} onRetry={onRetry}>
        {(data: string) => <p>{data}</p>}
      </Async>,
    )

    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(screen.getByRole('alert')).toHaveTextContent('nope')
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('hands the data to its children once it has arrived', () => {
    render(<Async state={loaded('Arsenal')}>{(data: string) => <p>{data}</p>}</Async>)

    expect(screen.getByText('Arsenal')).toBeInTheDocument()
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})
