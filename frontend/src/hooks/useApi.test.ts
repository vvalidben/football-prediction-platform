import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client.ts'
import { useApi } from './useApi.ts'

/** A loader whose promise the test resolves by hand. */
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('useApi', () => {
  it('starts in a loading state and settles on the data', async () => {
    const { result } = renderHook(() => useApi(() => Promise.resolve({ value: 1 }), []))

    expect(result.current.state).toEqual({ kind: 'loading' })

    await waitFor(() => {
      expect(result.current.state.kind).toBe('success')
    })
    expect(result.current.state).toMatchObject({ data: { value: 1 } })
    expect(result.current.isRefreshing).toBe(false)
  })

  it('reports an ApiError with the status the backend sent', async () => {
    const { result } = renderHook(() =>
      useApi(() => Promise.reject(new ApiError('team 9 was not found', 404)), []),
    )

    await waitFor(() => {
      expect(result.current.state.kind).toBe('error')
    })
    expect(result.current.state).toEqual({
      kind: 'error',
      message: 'team 9 was not found',
      status: 404,
    })
  })

  it('does not show an unexpected failure as if the backend had spoken', async () => {
    const { result } = renderHook(() => useApi(() => Promise.reject(new TypeError('boom')), []))

    await waitFor(() => {
      expect(result.current.state.kind).toBe('error')
    })
    expect(result.current.state).toEqual({
      kind: 'error',
      message: 'Something went wrong',
      status: null,
    })
  })

  it('reloads on demand', async () => {
    let calls = 0
    const { result } = renderHook(() =>
      useApi(() => {
        calls += 1
        return Promise.resolve(calls)
      }, []),
    )
    await waitFor(() => {
      expect(result.current.state).toMatchObject({ data: 1 })
    })

    act(() => {
      result.current.reload()
    })

    await waitFor(() => {
      expect(result.current.state).toMatchObject({ data: 2 })
    })
  })

  it('goes back to loading when its dependencies change', async () => {
    let id = 1
    const { result, rerender } = renderHook(() =>
      useApi(() => Promise.resolve(`team ${String(id)}`), [id]),
    )
    await waitFor(() => {
      expect(result.current.state).toMatchObject({ data: 'team 1' })
    })

    id = 2
    rerender()

    await waitFor(() => {
      expect(result.current.state).toMatchObject({ data: 'team 2' })
    })
  })

  it('aborts the request in flight when the dependencies change', async () => {
    const signals: AbortSignal[] = []
    let id = 1
    const first = deferred<string>()
    const { rerender } = renderHook(() =>
      useApi(
        (signal) => {
          signals.push(signal)
          return id === 1 ? first.promise : Promise.resolve('second')
        },
        [id],
      ),
    )

    id = 2
    rerender()

    await waitFor(() => {
      expect(signals).toHaveLength(2)
    })
    expect(signals[0].aborted).toBe(true)
    expect(signals[1].aborted).toBe(false)
  })

  it('a slow response cannot overwrite a newer one', async () => {
    let id = 1
    const slow = deferred<string>()
    const { result, rerender } = renderHook(() =>
      useApi(() => (id === 1 ? slow.promise : Promise.resolve('newer')), [id]),
    )

    id = 2
    rerender()
    await waitFor(() => {
      expect(result.current.state).toMatchObject({ data: 'newer' })
    })
    await act(async () => {
      slow.resolve('stale')
      await slow.promise
    })

    expect(result.current.state).toMatchObject({ data: 'newer' })
  })

  it('aborts on unmount, so a closed page stops asking', async () => {
    const signals: AbortSignal[] = []
    const pending = deferred<string>()
    const { unmount } = renderHook(() =>
      useApi((signal) => {
        signals.push(signal)
        return pending.promise
      }, []),
    )
    await waitFor(() => {
      expect(signals).toHaveLength(1)
    })

    unmount()

    expect(signals[0].aborted).toBe(true)
  })

  it('refreshes on an interval when one is given', async () => {
    vi.useFakeTimers()
    let calls = 0
    renderHook(() =>
      useApi(
        () => {
          calls += 1
          return Promise.resolve(calls)
        },
        [],
        { refreshIntervalMs: 30_000 },
      ),
    )
    expect(calls).toBe(1)

    await act(async () => {
      vi.advanceTimersByTime(60_000)
      await Promise.resolve()
    })

    expect(calls).toBe(3)
  })

  it('does not refresh when no interval is given', async () => {
    vi.useFakeTimers()
    let calls = 0
    renderHook(() =>
      useApi(() => {
        calls += 1
        return Promise.resolve(calls)
      }, []),
    )

    await act(async () => {
      vi.advanceTimersByTime(600_000)
      await Promise.resolve()
    })

    expect(calls).toBe(1)
  })

  it('stops refreshing once the component is gone', async () => {
    vi.useFakeTimers()
    let calls = 0
    const { unmount } = renderHook(() =>
      useApi(
        () => {
          calls += 1
          return Promise.resolve(calls)
        },
        [],
        { refreshIntervalMs: 10_000 },
      ),
    )

    unmount()
    await act(async () => {
      vi.advanceTimersByTime(60_000)
      await Promise.resolve()
    })

    expect(calls).toBe(1)
  })
})
