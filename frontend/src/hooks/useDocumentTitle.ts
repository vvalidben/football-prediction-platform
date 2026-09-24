import { useEffect } from 'react'

const SITE = 'Football Prediction Platform'

/**
 * Name the page in the browser tab.
 *
 * Without this every view is called "Football Prediction Platform": bookmarks,
 * history entries and open tabs are indistinguishable. The title is restored when
 * the page unmounts, so a stale name never outlives the view that set it.
 */
export function useDocumentTitle(title: string | null): void {
  useEffect(() => {
    if (title === null) return
    const previous = document.title
    document.title = `${title} · ${SITE}`
    return () => {
      document.title = previous
    }
  }, [title])
}
