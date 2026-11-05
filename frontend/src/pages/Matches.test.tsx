import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
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
