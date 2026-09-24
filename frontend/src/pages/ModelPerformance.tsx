import { useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getImportance, getModelPerformance, getModels } from '../api/endpoints.ts'
import type { ModelVersion } from '../api/types.ts'
import { Async } from '../components/Async.tsx'
import { BarChart, CalibrationChart, LineChart } from '../components/charts/Charts.tsx'
import { Stat, StatGrid } from '../components/Stat.tsx'
import { StatusBadge } from '../components/StatusBadge.tsx'
import { Card, Empty, ErrorState, Loading } from '../components/states.tsx'
import { dateOnly, decimal, percent } from '../format.ts'
import { useDocumentTitle } from '../hooks/useDocumentTitle.ts'
import { useApi } from '../hooks/useApi.ts'
import { PageHeader } from './PageHeader.tsx'

/** The training metrics as recorded at registration, flattened for display. */
function trainingMetrics(model: ModelVersion): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  for (const [group, values] of Object.entries(model.metrics)) {
    if (typeof values !== 'object' || values === null) {
      rows.push({ label: group, value: String(values) })
      continue
    }
    for (const [name, value] of Object.entries(values as Record<string, unknown>)) {
      if (typeof value === 'number')
        rows.push({ label: `${group} · ${name}`, value: value.toFixed(4) })
    }
  }
  return rows
}

export function ModelPerformance() {
  useDocumentTitle('Model performance')
  const [params, setParams] = useSearchParams()
  const models = useApi(
    useCallback((signal: AbortSignal) => getModels({}, signal), []),
    [],
  )

  const selectedParam = params.get('model')
  const rows = models.state.kind === 'success' ? models.state.data : []
  const selected: ModelVersion | undefined =
    rows.find((row) => String(row.id) === selectedParam) ??
    rows.find((row) => row.is_active) ??
    rows.at(0)
  const selectedId = selected === undefined ? null : selected.id

  const performance = useApi(
    useCallback(
      (signal: AbortSignal) =>
        selectedId === null ? Promise.resolve(null) : getModelPerformance(selectedId, signal),
      [selectedId],
    ),
    [selectedId],
  )
  const importance = useApi(
    useCallback(
      (signal: AbortSignal) =>
        selectedId === null ? Promise.resolve(null) : getImportance(selectedId, 10, signal),
      [selectedId],
    ),
    [selectedId],
  )

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model performance"
        subtitle="Measured on stored predictions whose match has since been played. Training metrics are shown separately — they are not evidence of live performance."
        action={
          rows.length > 0 && (
            <label className="flex items-center gap-2 text-xs text-slate-600">
              Model
              <select
                value={selectedId === null ? '' : String(selectedId)}
                onChange={(event) => {
                  setParams({ model: event.target.value })
                }}
                className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs"
              >
                {rows.map((row) => (
                  <option key={row.id} value={row.id}>
                    {row.name} {row.version}
                    {row.is_active ? ' (active)' : ''}
                  </option>
                ))}
              </select>
            </label>
          )
        }
      />

      <Async state={models.state} onRetry={models.reload} loadingLabel="Loading models…">
        {(all) =>
          all.length === 0 ? (
            <Empty
              title="No model versions registered"
              hint="Train a model and register it before performance can be reported."
            />
          ) : (
            <Async
              state={performance.state}
              onRetry={performance.reload}
              loadingLabel="Loading performance…"
            >
              {(data) =>
                data === null ? (
                  <Empty title="No model selected" />
                ) : (
                  <div className="space-y-6">
                    <Card
                      title={`${data.model.name} ${data.model.version}`}
                      action={
                        <StatusBadge
                          tone={data.model.is_active ? 'ok' : 'pending'}
                          label={data.model.is_active ? 'Active' : 'Not serving'}
                        />
                      }
                    >
                      {data.evaluated === 0 ? (
                        <Empty
                          title="No evaluated predictions yet"
                          hint="Accuracy, calibration and history appear once predictions made by this version have been scored against results."
                        />
                      ) : (
                        <StatGrid>
                          <Stat
                            label="Evaluated"
                            value={data.evaluated}
                            note={`${dateOnly(data.first_match)} – ${dateOnly(data.last_match)}`}
                          />
                          <Stat
                            label="Accuracy"
                            value={data.accuracy === null ? '—' : percent(data.accuracy)}
                            note="1X2 outcome"
                          />
                          <Stat
                            label="Log loss"
                            value={decimal(data.log_loss, 4)}
                            note="Lower is better"
                          />
                          <Stat
                            label="Brier"
                            value={decimal(data.brier, 4)}
                            note="Lower is better"
                          />
                          <Stat
                            label="RPS"
                            value={decimal(data.ranked_probability_score, 4)}
                            note="Ranked probability score"
                          />
                          <Stat
                            label="Exact score"
                            value={
                              data.exact_score_accuracy === null
                                ? '—'
                                : percent(data.exact_score_accuracy)
                            }
                            note="Most probable score was right"
                          />
                        </StatGrid>
                      )}
                    </Card>

                    <div className="grid gap-6 lg:grid-cols-2">
                      <Card title="Calibration">
                        {data.calibration.length === 0 ? (
                          <Empty title="Nothing to calibrate yet" />
                        ) : (
                          <div className="space-y-2">
                            <CalibrationChart
                              ariaLabel="Reliability diagram: predicted probability against observed frequency"
                              bins={data.calibration}
                            />
                            <p className="text-xs text-slate-500">
                              Each point is a probability band across all three outcomes; the dashed
                              line is perfect calibration. Point size is the number of cases.
                            </p>
                          </div>
                        )}
                      </Card>

                      <Card title="History">
                        {data.monthly.length === 0 ? (
                          <Empty title="No history yet" />
                        ) : (
                          <div className="space-y-2">
                            <LineChart
                              ariaLabel="Monthly accuracy and log loss"
                              yMax={Math.max(1, ...data.monthly.map((entry) => entry.log_loss))}
                              series={[
                                {
                                  name: 'Accuracy',
                                  colour: '#16a34a',
                                  points: data.monthly.map((entry, index) => ({
                                    x: index,
                                    y: entry.accuracy,
                                  })),
                                },
                                {
                                  name: 'Log loss',
                                  colour: '#2563eb',
                                  points: data.monthly.map((entry, index) => ({
                                    x: index,
                                    y: entry.log_loss,
                                  })),
                                },
                              ]}
                            />
                            <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                              {data.monthly.map((entry) => (
                                <li key={entry.month} className="tabular-nums">
                                  {entry.month}: {entry.matches} matches,{' '}
                                  {percent(entry.accuracy, 0)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </Card>
                    </div>

                    <Card
                      title="What this model relies on"
                      action={
                        <span className="text-xs text-slate-500">
                          Model behaviour, not a ranking of what matters in football
                        </span>
                      }
                    >
                      {importance.state.kind === 'loading' && (
                        <Loading label="Loading feature importance…" />
                      )}
                      {importance.state.kind === 'error' &&
                        (importance.state.status === 409 ? (
                          <Empty
                            title="This model has no importance measure"
                            hint={importance.state.message}
                          />
                        ) : (
                          <ErrorState
                            message={importance.state.message}
                            status={importance.state.status}
                            onRetry={importance.reload}
                          />
                        ))}
                      {importance.state.kind === 'success' && importance.state.data && (
                        <div className="space-y-3">
                          <BarChart
                            ariaLabel="Feature importance"
                            max={1}
                            bars={importance.state.data.entries.map((entry, index) => ({
                              label: entry.label,
                              value: entry.importance,
                              highlight: index === 0,
                            }))}
                          />
                          {importance.state.data.other_features > 0 && (
                            <p className="text-xs text-slate-500">
                              The other {importance.state.data.other_features} features account for{' '}
                              {percent(importance.state.data.other_importance)} between them.
                            </p>
                          )}
                          <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
                            <p>
                              <span className="font-medium text-slate-700">Method:</span>{' '}
                              {importance.state.data.method} — {importance.state.data.method_detail}
                            </p>
                            <p className="mt-1.5">{importance.state.data.caveat}</p>
                          </div>
                        </div>
                      )}
                    </Card>

                    {selected && (
                      <Card title="Recorded at registration">
                        <p className="mb-3 text-xs text-slate-500">
                          {selected.algorithm} · features {selected.feature_set_version ?? '—'} ·
                          dataset {selected.dataset_version} · trained{' '}
                          {dateOnly(selected.training_period.start)} to{' '}
                          {dateOnly(selected.training_period.end)}
                        </p>
                        {trainingMetrics(selected).length === 0 ? (
                          <Empty title="No training metrics were recorded" />
                        ) : (
                          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
                            {trainingMetrics(selected).map((metric) => (
                              <div key={metric.label} className="flex justify-between gap-2">
                                <dt className="truncate text-slate-500">{metric.label}</dt>
                                <dd className="shrink-0 font-medium tabular-nums">
                                  {metric.value}
                                </dd>
                              </div>
                            ))}
                          </dl>
                        )}
                      </Card>
                    )}
                  </div>
                )
              }
            </Async>
          )
        }
      </Async>
    </div>
  )
}
