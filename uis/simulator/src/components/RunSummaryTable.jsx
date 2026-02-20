import { useEffect, useState } from 'react'
import './RunSummaryTable.css'
import TemplateStatsModal from './TemplateStatsModal.jsx'

const PCT = (v) => `${(v * 100).toFixed(1)}%`
const FMT = (v, decimals = 2) =>
    v == null ? '—' : Number(v).toFixed(decimals)

export default function RunSummaryTable({
    experimentName,
    selectedRun,
    onSelectRun,
}) {
    const [data, setData] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [statsRunId, setStatsRunId] = useState(null)

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
                                <th>Queries</th>
                                <th>Violations</th>
                                <th>Violation Rate</th>
                                <th>Total Cost</th>
                                <th>SLO (s)</th>
                                <th>Stats</th>
                            </tr>
                        </thead>
                        <tbody>
                            {runs.map((run, idx) => (
                                <tr
                                    key={run.run_id}
                                    className={`rst-row${selectedRun === run.run_id ? ' selected' : ''}`}
                                    onClick={() => onSelectRun(run.run_id, idx)}
                                >
                                    <td className="rst-index">{idx}</td>
                                    <td className="rst-run-id">{run.run_id}</td>
                                    <td>{run.seed ?? '—'}</td>
                                    <td>{run.total_queries ?? '—'}</td>
                                    <td>{run.violating_queries ?? '—'}</td>
                                    <td
                                        className={
                                            run.violation_rate > 0.05 ? 'rst-bad' : 'rst-good'
                                        }
                                    >
                                        {run.violation_rate != null ? PCT(run.violation_rate) : '—'}
                                    </td>
                                    <td>{run.total_cost != null ? `$${FMT(run.total_cost)}` : '—'}</td>
                                    <td>{run.slo_s ?? '—'}</td>
                                    <td
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            setStatsRunId(run.run_id)
                                        }}
                                        className="rst-stats-cell"
                                        title="View per-template stats"
                                    >
                                        📊
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {statsRunId && (
                <TemplateStatsModal
                    experimentName={experimentName}
                    runId={statsRunId}
                    onClose={() => setStatsRunId(null)}
                />
            )}
        </>
    )
}
