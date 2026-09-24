import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ErrorBoundary } from './ErrorBoundary.tsx'

function Boom({ fail }: { fail: boolean }) {
  if (fail) throw new Error('render exploded')
  return <p>Working view</p>
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // React logs the caught error itself; the test asserts on what the user sees.
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  it('shows the children while nothing is wrong', () => {
    render(
      <ErrorBoundary>
        <Boom fail={false} />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Working view')).toBeInTheDocument()
  })

  it('keeps the rest of the application usable when a view throws', () => {
    render(
      <ErrorBoundary>
        <Boom fail />
      </ErrorBoundary>,
    )

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('This page stopped working')
    expect(alert).toHaveTextContent('render exploded')
    expect(alert).toHaveTextContent('The rest of the application still works')
  })

  it('retries when asked', async () => {
    function Flaky() {
      const [fail, setFail] = useState(true)
      return (
        <>
          <button
            type="button"
            onClick={() => {
              setFail(false)
            }}
          >
            Fix it
          </button>
          <ErrorBoundary>
            <Boom fail={fail} />
          </ErrorBoundary>
        </>
      )
    }
    render(<Flaky />)
    expect(screen.getByRole('alert')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Fix it' }))
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))

    expect(screen.getByText('Working view')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('clears itself when the route changes', () => {
    const { rerender } = render(
      <ErrorBoundary resetKey="/broken">
        <Boom fail />
      </ErrorBoundary>,
    )
    expect(screen.getByRole('alert')).toBeInTheDocument()

    rerender(
      <ErrorBoundary resetKey="/somewhere-else">
        <Boom fail={false} />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Working view')).toBeInTheDocument()
  })

  it('reports the failure to the console rather than to a service that does not exist', () => {
    render(
      <ErrorBoundary>
        <Boom fail />
      </ErrorBoundary>,
    )

    expect(console.error).toHaveBeenCalled()
  })
})
