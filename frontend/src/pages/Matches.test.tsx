import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { windowFor } from '../matchPeriods.ts'
import { match } from '../test/fixtures.ts'
import { mockApi, page, renderAt, urlOf } from '../test/render.tsx'
import { Matches } from './Matches.tsx'

const competitions = page([
  { id: 7, name: 'Premier League', country: 'England' },
  { id: 8, name: 'UEFA Champions League', country: null },
])

const upcomingUrls = (fetchMock: ReturnType<typeof mockApi>) =>
  fetchMock.mock.calls
    .map(([input]) => urlOf(input))
    .filter((url) => url.includes('/matches/upcoming'))

const league = () => screen.findByRole('combobox', { name: 'League' })

const leagueTeams = page([
  { id: 11, name: 'Arsenal', short_name: 'ARS', country: 'England' },
  { id: 12, name: 'Chelsea', short_name: 'CHE', country: 'England' },
])
const withFilters = {
  '/competitions': competitions,
  '/teams': leagueTeams,
  '/matches/upcoming': page([match]),
}

describe('windowFor', () => {
  // Local dates: 2026-11-04 is a Wednesday.
  const wednesday = new Date(2026, 10, 4, 15, 0)

  it('leaves the dates open for any time', () => {
    expect(windowFor('', wednesday)).toEqual({})
  })

  it('gives one day for today', () => {
    expect(windowFor('today', wednesday)).toEqual({
      from_date: '2026-11-04',
      to_date: '2026-11-04',
    })
  })

  it('gives the coming Saturday and Sunday for this weekend', () => {
    expect(windowFor('weekend', wednesday)).toEqual({
      from_date: '2026-11-07',
      to_date: '2026-11-08',
    })
  })

  it('stays on the current weekend while it is on', () => {
    expect(windowFor('weekend', new Date(2026, 10, 7, 12, 0))).toEqual({
      from_date: '2026-11-07',
      to_date: '2026-11-08',
    })
    expect(windowFor('weekend', new Date(2026, 10, 8, 12, 0))).toEqual({
      from_date: '2026-11-08',
      to_date: '2026-11-08',
    })
  })

  it('gives today and the six days after it for the next 7 days', () => {
    expect(windowFor('week', wednesday)).toEqual({
      from_date: '2026-11-04',
      to_date: '2026-11-10',
    })
  })

  it('crosses a month boundary', () => {
    expect(windowFor('week', new Date(2026, 10, 28, 12, 0))).toEqual({
      from_date: '2026-11-28',
      to_date: '2026-12-04',
    })
  })
})

describe('filters', () => {
  it('appear only once a league is chosen', async () => {
    mockApi(withFilters)
    renderAt(<Matches />, '/matches')
    await league()

    expect(screen.queryByRole('combobox', { name: 'Team' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox', { name: 'When' })).not.toBeInTheDocument()
  })

  it('offer the teams of the chosen league', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7')

    await screen.findByRole('combobox', { name: 'Team' })

    expect(screen.getByRole('option', { name: 'Chelsea' })).toBeInTheDocument()
    const teamUrl = fetchMock.mock.calls
      .map(([input]) => urlOf(input))
      .find((url) => url.includes('/teams'))
    expect(teamUrl).toContain('competition_id=7')
  })

  it('narrow the matches to one team', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7')

    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'Team' }), '12')

    await waitFor(() => {
      expect(upcomingUrls(fetchMock).at(-1)).toContain('team_id=12')
    })
  })

  it('narrow the matches to a period', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7')

    await userEvent.selectOptions(await screen.findByRole('combobox', { name: 'When' }), 'weekend')

    await waitFor(() => {
      const url = upcomingUrls(fetchMock).at(-1) ?? ''
      expect(url).toContain('from_date=')
      expect(url).toContain('to_date=')
    })
  })

  it('reopen as they were left, from the address', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7&team=12&when=today')

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: 'Team' })).toHaveValue('12')
    })
    expect(screen.getByRole('combobox', { name: 'When' })).toHaveValue('today')
    expect(upcomingUrls(fetchMock).at(-1)).toContain('team_id=12')
  })

  it('return to the first page when changed', async () => {
    const fetchMock = mockApi({
      ...withFilters,
      '/matches/upcoming': { ...page([match]), total: 60, limit: 25 },
    })
    renderAt(<Matches />, '/matches?league=7&page=2')
    await screen.findByText(/Page 2 of 3/)

    await userEvent.selectOptions(screen.getByRole('combobox', { name: 'Team' }), '12')

    await waitFor(() => {
      const url = upcomingUrls(fetchMock).at(-1) ?? ''
      expect(url).toContain('team_id=12')
      expect(url).not.toContain('offset=25')
    })
  })

  it('stay behind when the league changes, since its teams do not', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7&team=12&when=week')
    await screen.findByRole('combobox', { name: 'Team' })

    await userEvent.selectOptions(await league(), '8')

    await waitFor(() => {
      const url = upcomingUrls(fetchMock).at(-1) ?? ''
      expect(url).toContain('competition_id=8')
      expect(url).not.toContain('team_id')
      expect(url).not.toContain('from_date')
    })
  })

  it('say when nothing fits them, and how to widen the search', async () => {
    mockApi({ ...withFilters, '/matches/upcoming': page([]) })
    renderAt(<Matches />, '/matches?league=7&when=today')

    expect(await screen.findByText('No matches fit these filters')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear filters' })).toBeInTheDocument()
  })

  it('can all be cleared at once', async () => {
    const fetchMock = mockApi({ ...withFilters, '/matches/upcoming': page([]) })
    renderAt(<Matches />, '/matches?league=7&team=12&when=today')

    await userEvent.click(await screen.findByRole('button', { name: 'Clear filters' }))

    await waitFor(() => {
      const url = upcomingUrls(fetchMock).at(-1) ?? ''
      expect(url).toContain('competition_id=7')
      expect(url).not.toContain('team_id')
      expect(url).not.toContain('from_date')
    })
  })

  it('ignore a period in the address they do not know', async () => {
    const fetchMock = mockApi(withFilters)
    renderAt(<Matches />, '/matches?league=7&when=someday')

    await screen.findByRole('combobox', { name: 'When' })

    expect(screen.getByRole('combobox', { name: 'When' })).toHaveValue('')
    expect(upcomingUrls(fetchMock).at(-1)).not.toContain('from_date')
  })
})

describe('Matches', () => {
  it('asks for a league first and loads no matches until one is chosen', async () => {
    const fetchMock = mockApi({
      '/competitions': competitions,
      '/matches/upcoming': page([match]),
    })
    renderAt(<Matches />, '/matches')

    await league()

    expect(screen.getByText('Choose a league')).toBeInTheDocument()
    expect(upcomingUrls(fetchMock)).toEqual([])
  })

  it('offers every league, with its country when it has one', async () => {
    mockApi({ '/competitions': competitions })
    renderAt(<Matches />, '/matches')

    await league()

    expect(screen.getByRole('option', { name: 'Premier League · England' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'UEFA Champions League' })).toBeInTheDocument()
  })

  it('shows the upcoming matches of the chosen league', async () => {
    const fetchMock = mockApi({
      '/competitions': competitions,
      '/matches/upcoming': page([match]),
    })
    renderAt(<Matches />, '/matches')

    await userEvent.selectOptions(await league(), '7')

    expect(await screen.findByRole('link', { name: 'Arsenal' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Chelsea' })).toBeInTheDocument()
    expect(upcomingUrls(fetchMock).at(-1)).toContain('competition_id=7')
  })

  it('groups matches under the day they are played', async () => {
    mockApi({
      '/competitions': competitions,
      '/matches/upcoming': page([
        match,
        { ...match, id: 2 },
        { ...match, id: 3, kickoff_at: '2026-10-03T14:00:00Z' },
      ]),
    })
    renderAt(<Matches />, '/matches?league=7')

    await screen.findAllByRole('link', { name: 'Arsenal' })

    expect(screen.getAllByRole('heading', { level: 3 })).toHaveLength(2)
  })

  it('opens the prediction page when Predict is pressed', async () => {
    mockApi({ '/competitions': competitions, '/matches/upcoming': page([match]) })
    renderAt(
      <Routes>
        <Route path="/matches" element={<Matches />} />
        <Route path="/matches/:matchId" element={<p>the prediction page</p>} />
      </Routes>,
      '/matches?league=7',
    )

    await userEvent.click(await screen.findByRole('link', { name: 'Predict' }))

    expect(await screen.findByText('the prediction page')).toBeInTheDocument()
  })

  it('reopens on the league in the address, so going back keeps the list', async () => {
    const fetchMock = mockApi({
      '/competitions': competitions,
      '/matches/upcoming': page([match]),
    })
    renderAt(<Matches />, '/matches?league=7')

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: 'League' })).toHaveValue('7')
    })
    expect(upcomingUrls(fetchMock).at(-1)).toContain('competition_id=7')
  })

  it('ignores a league in the address that is not a number', async () => {
    const fetchMock = mockApi({ '/competitions': competitions })
    renderAt(<Matches />, '/matches?league=abc')

    await league()

    expect(screen.getByText('Choose a league')).toBeInTheDocument()
    expect(upcomingUrls(fetchMock)).toEqual([])
  })

  it('says when a league has nothing scheduled', async () => {
    mockApi({ '/competitions': competitions, '/matches/upcoming': page([]) })
    renderAt(<Matches />, '/matches?league=8')

    expect(await screen.findByText('No upcoming matches')).toBeInTheDocument()
    expect(screen.getByText(/UEFA Champions League has nothing scheduled/)).toBeInTheDocument()
  })

  it('pages through a long fixture list', async () => {
    const fetchMock = mockApi({
      '/competitions': competitions,
      '/matches/upcoming': { ...page([match]), total: 60, limit: 25 },
    })
    renderAt(<Matches />, '/matches?league=7')

    expect(await screen.findByText(/Page 1 of 3/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()

    await userEvent.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => {
      expect(upcomingUrls(fetchMock).at(-1)).toContain('offset=25')
    })
    expect(await screen.findByText(/Page 2 of 3/)).toBeInTheDocument()
  })

  it('goes back to the first page when the league changes', async () => {
    const fetchMock = mockApi({
      '/competitions': competitions,
      '/matches/upcoming': { ...page([match]), total: 60, limit: 25 },
    })
    renderAt(<Matches />, '/matches?league=7&page=2')
    await screen.findByText(/Page 2 of 3/)

    await userEvent.selectOptions(await league(), '8')

    await waitFor(() => {
      const url = upcomingUrls(fetchMock).at(-1) ?? ''
      expect(url).toContain('competition_id=8')
      expect(url).not.toContain('offset=25')
    })
  })

  it('names the page in the browser tab', async () => {
    mockApi({ '/competitions': competitions })
    renderAt(<Matches />, '/matches')
    await league()

    expect(document.title).toBe('Matches · Football Prediction Platform')
  })

  it('reports a failure loading matches, with a retry', async () => {
    mockApi({
      '/competitions': competitions,
      '/matches/upcoming': new Response(JSON.stringify({ detail: 'database is down' }), {
        status: 503,
      }),
    })
    renderAt(<Matches />, '/matches?league=7')

    expect(await screen.findByRole('alert')).toHaveTextContent('database is down')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('reports a failure loading the leagues, with a retry', async () => {
    mockApi({
      '/competitions': new Response(JSON.stringify({ detail: 'database is down' }), {
        status: 503,
      }),
    })
    renderAt(<Matches />, '/matches')

    expect(await screen.findByRole('alert')).toHaveTextContent('database is down')
  })

  it('keeps the chosen league in the address', async () => {
    function Address() {
      return <output>{useLocation().search}</output>
    }
    mockApi({ '/competitions': competitions, '/matches/upcoming': page([match]) })
    renderAt(
      <>
        <Matches />
        <Address />
      </>,
      '/matches',
    )

    await userEvent.selectOptions(await league(), '7')

    expect(await screen.findByText('?league=7')).toBeInTheDocument()
  })
})
