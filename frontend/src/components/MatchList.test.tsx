import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { MatchRead } from '../api/types.ts'
import * as fixtures from '../test/fixtures.ts'
import { MatchList } from './MatchList.tsx'
import { FormPills, Stat, StatGrid } from './Stat.tsx'

const played: MatchRead = {
  ...fixtures.match,
  id: 2,
  status: 'finished',
  home_score: 2,
  away_score: 0,
}

const renderWithRouter = (ui: React.ReactElement) => render(<MemoryRouter>{ui}</MemoryRouter>)

describe('MatchList', () => {
  it('links each match to its prediction and each team to its page', () => {
    renderWithRouter(<MatchList matches={[fixtures.match]} />)

    expect(screen.getByRole('link', { name: 'Prediction' })).toHaveAttribute('href', '/matches/1')
    expect(screen.getByRole('link', { name: 'Arsenal' })).toHaveAttribute('href', '/teams/11')
    expect(screen.getByRole('link', { name: 'Chelsea' })).toHaveAttribute('href', '/teams/12')
  })

  it('shows the competition and kickoff', () => {
    renderWithRouter(<MatchList matches={[fixtures.match]} />)

    expect(screen.getByText(/Premier League/)).toBeInTheDocument()
  })

  it('shows a result only when asked to', () => {
    const { rerender } = renderWithRouter(<MatchList matches={[played]} />)
    expect(screen.queryByText('2–0')).not.toBeInTheDocument()

    rerender(
      <MemoryRouter>
        <MatchList matches={[played]} showResult />
      </MemoryRouter>,
    )

    expect(screen.getByText('2–0')).toBeInTheDocument()
  })

  it('shows a dash for a match that has not been played', () => {
    renderWithRouter(<MatchList matches={[fixtures.match]} showResult />)

    expect(screen.getByText('—')).toBeInTheDocument()
  })

  it('renders nothing but an empty list when there are no matches', () => {
    renderWithRouter(<MatchList matches={[]} />)

    expect(screen.getByRole('list')).toBeEmptyDOMElement()
  })
})

describe('Stat', () => {
  it('pairs a label with its value', () => {
    render(
      <StatGrid>
        <Stat label="Accuracy" value="52.0%" note="Outcome" />
      </StatGrid>,
    )

    expect(screen.getByText('Accuracy')).toBeInTheDocument()
    expect(screen.getByText('52.0%')).toBeInTheDocument()
    expect(screen.getByText('Outcome')).toBeInTheDocument()
  })

  it('works without a note', () => {
    render(
      <StatGrid>
        <Stat label="Predictions" value={40} />
      </StatGrid>,
    )

    expect(screen.getByText('40')).toBeInTheDocument()
  })
})

describe('FormPills', () => {
  it('shows the results in order, most recent last', () => {
    render(<FormPills form={['W', 'W', 'D', 'L', 'W']} label="Arsenal form" />)

    const list = screen.getByRole('list', { name: 'Arsenal form' })
    expect(
      within(list)
        .getAllByRole('listitem')
        .map((item) => item.textContent),
    ).toEqual(['W', 'W', 'D', 'L', 'W'])
  })

  it('says so when a team has not played', () => {
    render(<FormPills form={[]} />)

    expect(screen.getByText('No matches yet')).toBeInTheDocument()
    expect(screen.queryByRole('list')).not.toBeInTheDocument()
  })
})
