import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

interface LayoutProps {
  children: ReactNode
}

const pages = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/matches', label: 'Matches', end: false },
  { to: '/history', label: 'History', end: false },
  { to: '/teams', label: 'Teams', end: false },
  { to: '/models', label: 'Models', end: false },
  { to: '/data', label: 'Data', end: false },
]

export function Layout({ children }: LayoutProps) {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-3">
            <span
              aria-hidden="true"
              className="grid size-9 place-items-center rounded-lg bg-pitch-600 font-bold text-white"
            >
              FP
            </span>
            <div>
              <p className="text-base font-semibold text-slate-900">Football Prediction Platform</p>
              <p className="text-xs text-slate-500">Match forecasting from historical data</p>
            </div>
          </NavLink>

          <nav aria-label="Main" className="ml-auto">
            <ul className="flex flex-wrap gap-1">
              {pages.map((page) => (
                <li key={page.to}>
                  <NavLink
                    to={page.to}
                    end={page.end}
                    className={({ isActive }) =>
                      `rounded-md px-3 py-1.5 text-sm font-medium ${
                        isActive
                          ? 'bg-pitch-50 text-pitch-700'
                          : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
                      }`
                    }
                  >
                    {page.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-8 sm:px-6">{children}</main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-4 text-xs text-slate-500 sm:px-6">
          Every figure on these pages comes from the backend. Nothing is filled in when data is
          missing.
        </div>
      </footer>
    </div>
  )
}
