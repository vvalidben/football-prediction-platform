/**
 * The last line of defence in the browser.
 *
 * A thrown render error unmounts the whole React tree: without this the page goes
 * blank, which looks like the site is down rather than one view being broken. The
 * boundary keeps the layout and navigation usable and offers a way back.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Changing this resets the boundary — the router passes the current path. */
  resetKey?: string
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  override state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  override componentDidUpdate(previous: Props): void {
    if (previous.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    // Nothing is sent anywhere: there is no error-reporting service configured, and
    // inventing one would be worse than saying so.
    console.error('Unhandled error while rendering', error, info.componentStack)
  }

  override render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-6">
        <h2 className="text-base font-semibold text-red-900">This page stopped working</h2>
        <p className="mt-1 text-sm text-red-800">
          Something went wrong while showing this view. The rest of the application still works —
          use the navigation above, or try again.
        </p>
        <p className="mt-2 font-mono text-xs text-red-700">{error.message}</p>
        <button
          type="button"
          onClick={() => {
            this.setState({ error: null })
          }}
          className="mt-3 rounded-lg border border-red-300 bg-white px-3 py-1.5 text-sm font-medium text-red-700 hover:bg-red-100"
        >
          Try again
        </button>
      </div>
    )
  }
}
