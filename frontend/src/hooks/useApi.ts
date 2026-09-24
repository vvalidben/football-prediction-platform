import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client.ts'

export type AsyncState<T> =
  | { kind: 'loading' }
  | { kind: 'success'; data: T; loadedAt: Date }
  | { kind: 'error'; message: string; status: number | null }

interface UseApiResult<T> {
  state: AsyncState<T>
  isRefreshing: boolean
  reload: () => void
}

/**
 * Load data on mount and whenever `deps` change, with an optional refresh interval.
 *
 * The request is aborted when the component unmounts or the dependencies change,
 * so a slow response can never overwrite a newer one.
 */
export function useApi<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  options: { refreshIntervalMs?: number } = {},
): UseApiResult<T> {
  const [state, setState] = useState<AsyncState<T>>({ kind: 'loading' })
  const [isRefreshing, setIsRefreshing] = useState(true)
  const controllerRef = useRef<AbortController | null>(null)
  // Holds the current loader so `run` keeps one identity: the interval and the
  // retry button call the same function after the dependencies change. It is
  // assigned in the effect below, never during render.
  const loadRef = useRef(load)

  const run = useCallback(() => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setIsRefreshing(true)
    loadRef
      .current(controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return
        setState({ kind: 'success', data, loadedAt: new Date() })
        setIsRefreshing(false)
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return
        if (error instanceof DOMException && error.name === 'AbortError') return
        setState({
          kind: 'error',
          message: error instanceof ApiError ? error.message : 'Something went wrong',
          status: error instanceof ApiError ? error.status : null,
        })
        setIsRefreshing(false)
      })
  }, [])

  useEffect(() => {
    loadRef.current = load
    // The reset belongs to the request, not to a render: what is on screen is
    // stale the moment the dependencies change, so it is replaced by a loading
    // state before the new request is sent.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({ kind: 'loading' })
    run()
    return () => {
      controllerRef.current?.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    const interval = options.refreshIntervalMs
    if (!interval) return
    const id = window.setInterval(run, interval)
    return () => {
      window.clearInterval(id)
    }
  }, [run, options.refreshIntervalMs])

  return { state, isRefreshing, reload: run }
}
