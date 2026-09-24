/**
 * Renders the loading and error states of an {@link AsyncState} so a page only
 * has to describe what it looks like once the data has arrived.
 */
import type { ReactNode } from 'react'
import type { AsyncState } from '../hooks/useApi.ts'
import { ErrorState, Loading } from './states.tsx'

interface AsyncProps<T> {
  state: AsyncState<T>
  onRetry?: () => void
  loadingLabel?: string
  children: (data: T) => ReactNode
}

export function Async<T>({ state, onRetry, loadingLabel, children }: AsyncProps<T>) {
  if (state.kind === 'loading') return <Loading label={loadingLabel} />
  if (state.kind === 'error')
    return <ErrorState message={state.message} status={state.status} onRetry={onRetry} />
  return <>{children(state.data)}</>
}
