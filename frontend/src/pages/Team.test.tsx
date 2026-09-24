import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { TeamDetail } from '../api/types.ts'
import * as fixtures from '../test/fixtures.ts'
import { mockApi, renderAt, urlOf } from '../test/render.tsx'
import { Team } from './Team.tsx'

function renderPage(team: TeamDetail | Response = fixtures.teamDetail) {
  const fetchMock = mockApi({ '/teams': team })
  renderAt(
    <Routes>
      <Route path="/teams/:teamId" element={<Team />} />
    </Routes>,
    '/teams/11',
  )
  return fetchMock
}

describe('Team', () => {
  it('shows the record, the home and away split and the recent matches', async () => {
    renderPage()

    expect(await screen.findByRole('heading', { name: 'Arsenal', level: 1 })).toBeInTheDocument()
    expect(screen.getByText('6–2–2')).toBeInTheDocument()
    expect(screen.getByText('1.80 per match')).toBeInTheDocument()
    expect(screen.getByText('0.90 per match')).toBeInTheDocument()
    const table = screen.getByRole('table')
    expect(table).toHaveTextContent('home')
    expect(table).toHaveTextContent('away')
    expect(table).toHaveTextContent('4–1–0')
    expect(screen.getByLabelText('Arsenal form')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Arsenal' })).toHaveAttribute('href', '/teams/11')
  })

  it('draws the goal trend', async () => {
    renderPage()

    expect(
      await screen.findByRole('img', {
        name: 'Rolling goals scored and conceded, oldest match first',
      }),
    ).toBeInTheDocument()
  })

  it('says xG was not supplied rather than showing zero', async () => {
    renderPage()

    expect(await screen.findAllByText('No provider supplied xG')).toHaveLength(2) // for and against
  })

  it('shows xG once a provider has supplied it', async () => {
    renderPage({
      ...fixtures.teamDetail,
      record: {
        ...fixtures.teamDetail.record,
        expected_goals_for_per_match: 1.73,
        expected_goals_against_per_match: 0.95,
        matches_with_expected_goals: 10,
      },
    })

    expect(await screen.findByText('1.73')).toBeInTheDocument()
    expect(screen.getAllByText('From 10 matches').length).toBe(2)
  })

  it('reloads with a different window when one is chosen', async () => {
    const fetchMock = renderPage()
    await screen.findByRole('heading', { name: 'Arsenal', level: 1 })

    await userEvent.selectOptions(screen.getByLabelText('Window'), '40')

    expect(fetchMock.mock.calls.some(([input]) => urlOf(input).includes('matches=40'))).toBe(true)
  })

  it('says so when the team has no finished matches', async () => {
    renderPage({
      ...fixtures.teamDetail,
      record: { ...fixtures.teamDetail.record, played: 0 },
      recent_matches: [],
      trend: [],
    })

    expect(await screen.findByText('No finished matches for this team')).toBeInTheDocument()
  })

  it('reports an unknown team', async () => {
    renderPage(new Response(JSON.stringify({ detail: 'team 11 was not found' }), { status: 404 }))

    expect(await screen.findByRole('alert')).toHaveTextContent('team 11 was not found')
  })
})
