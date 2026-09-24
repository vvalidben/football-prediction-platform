import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, fetchHealth, type HealthResponse } from '../api/health.ts'

export type HealthState =
  | { kind: 'loading' }
  | { kind: 'success'; data: HealthResponse; checkedAt: Date }
  | { kind: 'error'; message: string; checkedAt: Date }

type SettledHealthState = Exclude<HealthState, { kind: 'loading' }>

interface UseHealthResult {
  state: HealthState
  isRefreshing: boolean
  refresh: () => void
}

async function checkHealth(signal: AbortSignal): Promise<SettledHealthState> {
  try {
    const data = await fetchHealth(signal)
    return { kind: 'success', data, checkedAt: new Date() }
  } catch (error) {
    const message =
      error instanceof ApiError ? error.message : 'Unexpected error while checking health'
    return { kind: 'error', message, checkedAt: new Date() }
  }
}

/** Load the backend health report on mount, on demand, and on an interval. */
export function useHealth(pollIntervalMs: number): UseHealthResult {
  const [state, setState] = useState<HealthState>({ kind: 'loading' })
  const [isRefreshing, setIsRefreshing] = useState(true)
  const controllerRef = useRef<AbortController | null>(null)

  // Starts a check, cancelling any in flight, and applies its result unless superseded.
  const startCheck = useCallback(() => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller

    void checkHealth(controller.signal).then((result) => {
      if (controller.signal.aborted) return
      setState(result)
      setIsRefreshing(false)
    })
  }, [])

  const refresh = useCallback(() => {
    setIsRefreshing(true)
    startCheck()
  }, [startCheck])

  useEffect(() => {
    startCheck()
    const intervalId = window.setInterval(refresh, pollIntervalMs)
    return () => {
      window.clearInterval(intervalId)
      controllerRef.current?.abort()
    }
  }, [startCheck, refresh, pollIntervalMs])

  return { state, isRefreshing, refresh }
}
