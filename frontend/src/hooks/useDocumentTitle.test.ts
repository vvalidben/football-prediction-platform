import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { useDocumentTitle } from './useDocumentTitle.ts'

const ORIGINAL = 'Football Prediction Platform'

afterEach(() => {
  document.title = ORIGINAL
})

describe('useDocumentTitle', () => {
  it('names the page in the browser tab', () => {
    renderHook(() => {
      useDocumentTitle('Prediction history')
    })

    expect(document.title).toBe('Prediction history · Football Prediction Platform')
  })

  it('follows the page it is on', () => {
    const { rerender } = renderHook(
      ({ title }: { title: string }) => {
        useDocumentTitle(title)
      },
      { initialProps: { title: 'Dashboard' } },
    )

    rerender({ title: 'Data status' })

    expect(document.title).toBe('Data status · Football Prediction Platform')
  })

  it('restores the previous title when the page goes away', () => {
    document.title = ORIGINAL
    const { unmount } = renderHook(() => {
      useDocumentTitle('Teams')
    })

    unmount()

    expect(document.title).toBe(ORIGINAL)
  })

  it('leaves the title alone while the name is still unknown', () => {
    document.title = ORIGINAL

    renderHook(() => {
      useDocumentTitle(null)
    })

    expect(document.title).toBe(ORIGINAL)
  })
})
