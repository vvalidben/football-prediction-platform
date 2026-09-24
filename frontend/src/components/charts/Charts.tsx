/**
 * Small SVG charts, drawn here rather than pulled from a charting library: the
 * shapes are simple, the bundle stays small, and every chart carries a label so
 * it can also be read by assistive technology.
 */

export interface Bar {
  label: string
  value: number
  highlight?: boolean
}

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

/** Horizontal bars, for probabilities that should be compared at a glance. */
export function BarChart({
  bars,
  ariaLabel,
  format = percent,
  max,
}: {
  bars: Bar[]
  ariaLabel: string
  format?: (value: number) => string
  max?: number
}) {
  const ceiling = max ?? Math.max(...bars.map((bar) => bar.value), 0.0001)
  return (
    <ul aria-label={ariaLabel} className="space-y-2">
      {bars.map((bar) => (
        <li key={bar.label} className="grid grid-cols-[6rem_1fr_4rem] items-center gap-2 text-sm">
          <span className="truncate text-slate-600">{bar.label}</span>
          <span className="h-3 overflow-hidden rounded-full bg-slate-100">
            <span
              className={`block h-full rounded-full ${bar.highlight ? 'bg-pitch-600' : 'bg-slate-400'}`}
              style={{ width: `${String(Math.max((bar.value / ceiling) * 100, 1))}%` }}
            />
          </span>
          <span className="text-right font-medium tabular-nums text-slate-900">
            {format(bar.value)}
          </span>
        </li>
      ))}
    </ul>
  )
}

/** Columns, for a distribution such as total goals or exact scores. */
export function ColumnChart({
  bars,
  ariaLabel,
  height = 120,
}: {
  bars: Bar[]
  ariaLabel: string
  height?: number
}) {
  const ceiling = Math.max(...bars.map((bar) => bar.value), 0.0001)
  return (
    <div role="img" aria-label={ariaLabel} className="flex items-end gap-1" style={{ height }}>
      {bars.map((bar) => (
        <div key={bar.label} className="flex flex-1 flex-col items-center gap-1">
          <div
            title={`${bar.label}: ${percent(bar.value)}`}
            className={`w-full rounded-t ${bar.highlight ? 'bg-pitch-600' : 'bg-slate-300'}`}
            style={{ height: `${String(Math.max((bar.value / ceiling) * (height - 20), 2))}px` }}
          />
          <span className="text-[10px] text-slate-500">{bar.label}</span>
        </div>
      ))}
    </div>
  )
}

export interface Point {
  x: number
  y: number
}

/** One or more series over time, e.g. rolling goals for and against. */
export function LineChart({
  series,
  ariaLabel,
  height = 160,
  yMax,
}: {
  series: { name: string; colour: string; points: Point[] }[]
  ariaLabel: string
  height?: number
  yMax?: number
}) {
  const width = 480
  const padding = { top: 8, right: 8, bottom: 20, left: 28 }
  const all = series.flatMap((line) => line.points)
  const ceiling = yMax ?? Math.max(...all.map((point) => point.y), 1)
  const xMax = Math.max(...all.map((point) => point.x), 1)
  const scaleX = (x: number) => padding.left + (x / xMax) * (width - padding.left - padding.right)
  const scaleY = (y: number) =>
    height - padding.bottom - (y / ceiling) * (height - padding.top - padding.bottom)

  return (
    <figure className="w-full">
      <svg
        role="img"
        aria-label={ariaLabel}
        viewBox={`0 0 ${String(width)} ${String(height)}`}
        className="h-auto w-full"
      >
        {[0, ceiling / 2, ceiling].map((value) => (
          <g key={value}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={scaleY(value)}
              y2={scaleY(value)}
              className="stroke-slate-200"
              strokeWidth={1}
            />
            <text x={2} y={scaleY(value) + 3} className="fill-slate-400 text-[9px]">
              {value.toFixed(1)}
            </text>
          </g>
        ))}
        {series.map((line) => (
          <polyline
            key={line.name}
            fill="none"
            stroke={line.colour}
            strokeWidth={2}
            points={line.points
              .map((point) => `${String(scaleX(point.x))},${String(scaleY(point.y))}`)
              .join(' ')}
          />
        ))}
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-slate-600">
        {series.map((line) => (
          <span key={line.name} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="inline-block h-0.5 w-4 rounded"
              style={{ backgroundColor: line.colour }}
            />
            {line.name}
          </span>
        ))}
      </figcaption>
    </figure>
  )
}

/**
 * A reliability diagram: predicted probability against how often it happened.
 * Points on the dashed diagonal mean the probabilities can be taken at face value.
 */
export function CalibrationChart({
  bins,
  ariaLabel,
}: {
  bins: { mean_predicted: number; observed: number; count: number }[]
  ariaLabel: string
}) {
  const size = 220
  const padding = 28
  const scale = (value: number) => padding + value * (size - padding * 2)
  const largest = Math.max(...bins.map((bin) => bin.count), 1)

  return (
    <svg
      role="img"
      aria-label={ariaLabel}
      viewBox={`0 0 ${String(size)} ${String(size)}`}
      className="h-auto w-full max-w-[320px]"
    >
      <line
        x1={scale(0)}
        y1={size - scale(0)}
        x2={scale(1)}
        y2={size - scale(1)}
        className="stroke-slate-300"
        strokeDasharray="4 3"
        strokeWidth={1}
      />
      <line
        x1={padding}
        y1={padding}
        x2={padding}
        y2={size - padding}
        className="stroke-slate-300"
      />
      <line
        x1={padding}
        y1={size - padding}
        x2={size - padding}
        y2={size - padding}
        className="stroke-slate-300"
      />
      {bins.map((bin) => (
        <circle
          key={`${String(bin.mean_predicted)}-${String(bin.count)}`}
          cx={scale(bin.mean_predicted)}
          cy={size - scale(bin.observed)}
          r={3 + (bin.count / largest) * 4}
          className="fill-pitch-600/70"
        >
          <title>{`predicted ${percent(bin.mean_predicted)}, happened ${percent(bin.observed)} (${String(bin.count)} cases)`}</title>
        </circle>
      ))}
      <text x={size / 2 - 20} y={size - 6} className="fill-slate-500 text-[9px]">
        predicted
      </text>
      <text x={4} y={padding - 8} className="fill-slate-500 text-[9px]">
        observed
      </text>
    </svg>
  )
}

export interface Contribution {
  label: string
  value: number
  detail?: string
}

/**
 * Contributions either side of a centre line: what pushed the model's output up
 * (right) and what pushed it down (left). The sign carries the meaning, so the
 * bars grow from the middle rather than from the left edge.
 */
export function ContributionChart({
  items,
  ariaLabel,
  format = (value: number) => value.toFixed(3),
}: {
  items: Contribution[]
  ariaLabel: string
  format?: (value: number) => string
}) {
  const widest = Math.max(...items.map((item) => Math.abs(item.value)), 1e-6)
  return (
    <ul aria-label={ariaLabel} className="space-y-1.5">
      {items.map((item) => {
        const share = (Math.abs(item.value) / widest) * 50
        const positive = item.value >= 0
        return (
          <li key={item.label} className="grid grid-cols-[1fr_9rem_4.5rem] items-center gap-2">
            <span className="truncate text-xs text-slate-600" title={item.detail ?? item.label}>
              {item.label}
            </span>
            <span className="relative h-3 rounded bg-slate-50" title={item.detail ?? item.label}>
              <span aria-hidden="true" className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
              <span
                className={`absolute inset-y-0 rounded ${positive ? 'bg-pitch-600' : 'bg-red-500'}`}
                style={
                  positive
                    ? { left: '50%', width: `${String(share)}%` }
                    : { right: '50%', width: `${String(share)}%` }
                }
              />
            </span>
            <span
              className={`text-right text-xs font-medium tabular-nums ${positive ? 'text-pitch-700' : 'text-red-700'}`}
            >
              {positive ? '+' : ''}
              {format(item.value)}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
