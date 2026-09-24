/**
 * The charts are how a prediction is actually read, so these tests check what a
 * person can see and what a screen reader is told — not the SVG geometry.
 */
import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { BarChart, CalibrationChart, ColumnChart, ContributionChart, LineChart } from './Charts.tsx'

describe('BarChart', () => {
  const bars = [
    { label: 'Arsenal', value: 0.51, highlight: true },
    { label: 'Draw', value: 0.26 },
    { label: 'Chelsea', value: 0.23 },
  ]

  it('labels the chart and shows every value as a percentage', () => {
    render(<BarChart bars={bars} ariaLabel="Result probabilities" max={1} />)

    const list = screen.getByRole('list', { name: 'Result probabilities' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(3)
    expect(within(list).getByText('51.0%')).toBeInTheDocument()
    expect(within(list).getByText('26.0%')).toBeInTheDocument()
    expect(within(list).getByText('23.0%')).toBeInTheDocument()
  })

  it('accepts another formatter for values that are not probabilities', () => {
    render(
      <BarChart
        bars={[{ label: 'Goals', value: 1.6 }]}
        ariaLabel="Expected goals"
        format={(value) => value.toFixed(2)}
      />,
    )

    expect(screen.getByText('1.60')).toBeInTheDocument()
  })

  it('draws a zero value without breaking the scale', () => {
    render(<BarChart bars={[{ label: 'None', value: 0 }]} ariaLabel="Empty" />)

    expect(screen.getByText('0.0%')).toBeInTheDocument()
  })
})

describe('ColumnChart', () => {
  it('is readable as an image with a name, and labels each column', () => {
    render(
      <ColumnChart
        ariaLabel="Total goals distribution"
        bars={[
          { label: '0', value: 0.08 },
          { label: '1', value: 0.2 },
          { label: '2', value: 0.26, highlight: true },
        ]}
      />,
    )

    const chart = screen.getByRole('img', { name: 'Total goals distribution' })
    expect(within(chart).getByText('0')).toBeInTheDocument()
    expect(within(chart).getByText('2')).toBeInTheDocument()
    expect(within(chart).getByTitle('2: 26.0%')).toBeInTheDocument()
  })
})

describe('LineChart', () => {
  const series = [
    {
      name: 'Scored',
      colour: '#16a34a',
      points: [
        { x: 0, y: 1 },
        { x: 1, y: 2 },
      ],
    },
    {
      name: 'Conceded',
      colour: '#dc2626',
      points: [
        { x: 0, y: 0.5 },
        { x: 1, y: 1.5 },
      ],
    },
  ]

  it('names the chart and lists its series', () => {
    render(<LineChart series={series} ariaLabel="Rolling goals" />)

    expect(screen.getByRole('img', { name: 'Rolling goals' })).toBeInTheDocument()
    expect(screen.getByText('Scored')).toBeInTheDocument()
    expect(screen.getByText('Conceded')).toBeInTheDocument()
  })

  it('survives a series with a single point', () => {
    render(
      <LineChart
        series={[{ name: 'One', colour: '#000', points: [{ x: 0, y: 1 }] }]}
        ariaLabel="Single"
      />,
    )

    expect(screen.getByRole('img', { name: 'Single' })).toBeInTheDocument()
  })
})

describe('CalibrationChart', () => {
  it('describes each point in words for anyone not reading the picture', () => {
    render(
      <CalibrationChart
        ariaLabel="Reliability diagram"
        bins={[
          { mean_predicted: 0.25, observed: 0.22, count: 40 },
          { mean_predicted: 0.55, observed: 0.6, count: 25 },
        ]}
      />,
    )

    const chart = screen.getByRole('img', { name: 'Reliability diagram' })
    // The description is an SVG <title> inside each point, so it is matched as text.
    expect(
      within(chart).getByText('predicted 25.0%, happened 22.0% (40 cases)'),
    ).toBeInTheDocument()
    expect(
      within(chart).getByText('predicted 55.0%, happened 60.0% (25 cases)'),
    ).toBeInTheDocument()
  })

  it('renders with no bins at all', () => {
    render(<CalibrationChart ariaLabel="Nothing yet" bins={[]} />)

    expect(screen.getByRole('img', { name: 'Nothing yet' })).toBeInTheDocument()
  })
})

describe('ContributionChart', () => {
  const items = [
    { label: 'Rating difference', value: 0.21, detail: 'pushed up the score' },
    { label: 'Away form', value: -0.09, detail: 'pushed down the score' },
  ]

  it('signs every contribution so the direction is unmistakable', () => {
    render(<ContributionChart items={items} ariaLabel="Feature contributions" />)

    const list = screen.getByRole('list', { name: 'Feature contributions' })
    expect(within(list).getByText('+0.210')).toBeInTheDocument()
    expect(within(list).getByText('-0.090')).toBeInTheDocument()
  })

  it('carries the explanation of each bar as its title', () => {
    render(<ContributionChart items={items} ariaLabel="Feature contributions" />)

    expect(screen.getAllByTitle('pushed up the score').length).toBeGreaterThan(0)
  })

  it('handles a contribution of exactly zero as positive, not missing', () => {
    render(<ContributionChart items={[{ label: 'Unused', value: 0 }]} ariaLabel="Contributions" />)

    expect(screen.getByText('+0.000')).toBeInTheDocument()
  })
})
