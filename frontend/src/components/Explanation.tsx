/**
 * "Why this prediction": the features that moved the model, and which way.
 *
 * The wording here is deliberate. Everything is phrased as something the *model*
 * did — contributed to, pushed up, pushed down — never as something that made a
 * result happen. The backend's own caveat is shown with the figures, not hidden
 * behind a tooltip.
 */
import { useState } from 'react'
import type { Explanation as ExplanationData, ExplanationTarget } from '../api/types.ts'
import { decimal, percent } from '../format.ts'
import type { AsyncState } from '../hooks/useApi.ts'
import { ContributionChart } from './charts/Charts.tsx'
import { Empty, ErrorState, Loading } from './states.tsx'

function outputOf(target: ExplanationTarget, value: number | null): string {
  if (value === null) return '—'
  return target.output_unit === 'probability' ? percent(value) : decimal(value)
}

function valueOf(item: { value: number | null; feature: string }): string {
  if (item.value === null) return 'no value for this match'
  return `value ${decimal(item.value)}`
}

export function ExplanationPanel({ explanation }: { explanation: ExplanationData }) {
  const [targetKey, setTargetKey] = useState(explanation.targets.at(0)?.key ?? '')
  const target: ExplanationTarget | undefined =
    explanation.targets.find((item) => item.key === targetKey) ?? explanation.targets.at(0)

  if (target === undefined) return <Empty title="This prediction has no explanation to show" />

  const items = target.contributions.map((item) => ({
    label: item.label,
    value: item.contribution,
    detail: `${item.label} (${item.feature}, ${valueOf(item)}): ${
      item.direction === 'positive' ? 'pushed up' : 'pushed down'
    } the model's ${target.label.toLowerCase()} score by ${decimal(Math.abs(item.contribution), 3)}`,
  }))

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-slate-700">
          These features contributed to the model&rsquo;s prediction.
        </p>
        <div role="group" aria-label="Outcome to explain" className="flex flex-wrap gap-1">
          {explanation.targets.map((option) => (
            <button
              key={option.key}
              type="button"
              aria-pressed={option.key === target.key}
              onClick={() => {
                setTargetKey(option.key)
              }}
              className={`rounded-md px-2.5 py-1 text-xs font-medium ${
                option.key === target.key
                  ? 'bg-pitch-50 text-pitch-700 ring-1 ring-inset ring-pitch-500/30'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Starting from {outputOf(target, target.baseline_output)} for{' '}
        {explanation.baseline_description}, the model arrived at{' '}
        <span className="font-medium text-slate-700">
          {outputOf(target, target.predicted_output)}
        </span>{' '}
        for {target.label.toLowerCase()}. Bars are on the model&rsquo;s {target.unit} scale, where
        the contributions add up.
      </p>

      {items.length === 0 ? (
        <Empty title="No feature moved this outcome" />
      ) : (
        <ContributionChart
          ariaLabel={`Feature contributions to ${target.label.toLowerCase()}`}
          items={items}
        />
      )}

      {target.other_features > 0 && (
        <p className="text-xs text-slate-500">
          The other {target.other_features} features together contributed{' '}
          <span className="tabular-nums">
            {target.other_contribution >= 0 ? '+' : ''}
            {decimal(target.other_contribution, 3)}
          </span>
          .
        </p>
      )}

      <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
        <p>
          <span className="font-medium text-slate-700">Method:</span> {explanation.method} —{' '}
          {explanation.method_detail}
        </p>
        <p className="mt-1.5">{explanation.caveat}</p>
      </div>
    </div>
  )
}

/**
 * A model with no valid method is not an error: the backend says so on purpose,
 * and the page reports that plainly instead of showing a red failure.
 */
export function ExplanationCard({
  state,
  onRetry,
}: {
  state: AsyncState<ExplanationData | null>
  onRetry: () => void
}) {
  if (state.kind === 'loading') return <Loading label="Working out the contributions…" />
  if (state.kind === 'error') {
    return state.status === 409 ? (
      <Empty title="This model cannot be explained" hint={state.message} />
    ) : (
      <ErrorState message={state.message} status={state.status} onRetry={onRetry} />
    )
  }
  if (state.data === null) return <Empty title="No prediction to explain yet" />
  return <ExplanationPanel explanation={state.data} />
}
