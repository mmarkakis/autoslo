import { useEffect, useState } from 'react'
import './RunSummaryTable.css'
import TemplateStatsModal from './TemplateStatsModal.jsx'

const PCT = (v) => `${(v * 100).toFixed(1)}%`
const FMT = (v, decimals = 2) =>
    v == null ? '—' : Number(v).toFixed(decimals)

const METRIC_LABELS = {
    binary: 'Binary',
    absolute_s: 'Absolute (s)',
    relative: 'Relative',
}

const METRIC_COLORS = {
    binary: '#ECC94B',
    absolute_s: '#63b3ed',
    relative: '#b794f4',
}

export default function RunSummaryTable({
    experimentName,
    selectedRun,
    onSelectRun,
}) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [statsRunId, setStatsRunId] = useState(null)
    const [statsRunMetric, setStatsRunMetric] = useState(null)

    useEffect(() => {
        if (!experimentName) return
        setLoading(true)
        setError(null)
        fetch(`/api/simulator/experiments/${encodeURIComponent(experimentName)}`)
            .then((r) => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`)
                return r.json()
            })
            .then((d) => {
                setData(d)
                setLoading(false)
            })
            .catch((e) => {
                setError(e.message)
                setLoading(false)
            })
    }, [experimentName])

    if (loading) return <div className="rst-status">Loading…</div>
    if (error) return <div className="rst-status error">{error}</div>
    if (!data) return null

    const { runs } = data

    return (
        <>
            <div className="rst-wrapper">
                <div className="rst-meta">
                    <span>{runs.length} runs</span>
                </div>
                <div className="rst-scroll">
                    <table className="rst-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Run ID</th>
                                <th>Seed</th>

                                <th>Num Queries</th>
                                <th>Total Cost</th>   

                                <th>SLO Metric Optimized</th>
                                <th>Threshold</th>

                                <th>SLO Viol. Rate</th>
                                <th>Total SLO Viol. Amount (s)</th>
                                <th>Mean Relative SLO Viol.</th>
                                
                                <th>Per-Template Stats</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs.map((run, idx) => {
                                const metric = run.slo_metric
                                const threshold = run.slo_threshold

                                // For the actively-optimized metric column: green if ≤ threshold, red if >
                                const ACTIVE_VAL = {
                                    binary: run.violation_rate,
                                    absolute_s: run.violation_amount_s,
                                    relative: run.violation_relative_mean,
                                }
                                const activeVal = metric ? ACTIVE_VAL[metric] : run.violation_rate
                                const activeBad = threshold != null && activeVal != null && activeVal > threshold
                                const activeColor = activeBad ? 'rst-bad' : 'rst-good'
                                const metricCellClass = (m) => m === metric ? activeColor : 'rst-neutral'
                                return (
                                <tr
                                    key={run.run_id}
                                    className={`rst-row${selectedRun === run.run_id ? ' selected' : ''}`}
                                    onClick={() => onSelectRun(run.run_id, idx)}
                                >
                                    <td className="rst-index">{idx}</td>
                                    <td className="rst-run-id">{run.run_id}</td>
                                    <td>{run.seed ?? '—'}</td>
                                    <td>{run.num_queries ?? '—'}</td>
                                    <td>{run.total_cost != null ? `$${FMT(run.total_cost)}` : '—'}</td>

                                    <td>
                                        {metric ? (
                                            <span
                                                className="rst-metric-badge"
                                                style={{ color: METRIC_COLORS[metric] ?? '#a0aec0', background: '#2d3748' }}
                                            >
                                                {METRIC_LABELS[metric] ?? metric}
                                            </span>
                                        ) : '—'}
                                    </td>
                                    <td>{threshold != null ? FMT(threshold, 4) : '—'}</td>

                                    <td className={metricCellClass('binary')}>
                                        {run.violation_rate != null ? PCT(run.violation_rate) : '—'}
                                    </td>
                                    <td className={metricCellClass('absolute_s')}>{run.violation_amount_s != null ? FMT(run.violation_amount_s, 3) : '—'}</td>
                                    <td className={metricCellClass('relative')}>{run.violation_relative_mean != null ? PCT(run.violation_relative_mean) : '—'}</td>
                                    
                                    
                                    <td
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            setStatsRunId(run.run_id)
                                            setStatsRunMetric(run.slo_metric ?? null)
                                        }}
                                        className="rst-stats-cell"
                                        title="View per-template stats"
                                    >
                                        📊
                                    </td>
                                </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {statsRunId && (
                <TemplateStatsModal
                    experimentName={experimentName}
                    runId={statsRunId}
                    sloMetric={statsRunMetric}
                    onClose={() => { setStatsRunId(null); setStatsRunMetric(null) }}
                />
            )}
        </>
    )
}
