import { useEffect, useState, useCallback } from 'react'
import './TemplateStatsModal.css'

const FMT = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d))
const PCT = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

const COLS = [
  { key: 'template_id',    label: 'Template',     sortNum: true  },
  { key: 'occurrences',    label: 'Count',        sortNum: true  },
  { key: 'p50_latency_s',  label: 'p50',          sortNum: true, group: 'latency'  },
  { key: 'p90_latency_s',  label: 'p90',          sortNum: true, group: 'latency'  },
  { key: 'p95_latency_s',  label: 'p95',          sortNum: true, group: 'latency'  },
  { key: 'violation_rate', label: 'SLO Viol. Rate', sortNum: true },
  { key: 'total_violation_amount_s', label: 'Total Viol. Amount (s)', sortNum: true },
  { key: 'mean_relative_violation', label: 'Mean Rel. Viol.', sortNum: true },
]

const METRIC_COL = {
  binary:     'violation_rate',
  absolute_s: 'total_violation_amount_s',
  relative:   'mean_relative_violation',
}

function violationClass(value, colKey, activeCol) {
  if (value == null || colKey !== activeCol) return ''
  // For rate: good ≤5%, warn ≤20%, bad >20%
  // For total amount / mean relative: use same numeric thresholds scaled by ~same feel
  if (colKey === 'violation_rate') {
    if (value <= 0.05) return 'tsm-good'
    if (value <= 0.20) return 'tsm-warn'
    return 'tsm-bad'
  }
  // absolute_s and relative: good=0, warn=small, bad=larger
  if (value === 0) return 'tsm-good'
  if (value <= 0.05) return 'tsm-warn'
  return 'tsm-bad'
}

export default function TemplateStatsModal({
  experimentName,
  runId,
  sloMetric,
  onClose,
}) {
  const activeCol = METRIC_COL[sloMetric] ?? null
  const [rows, setRows] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sortKey, setSortKey] = useState('template_id')
  const [sortAsc, setSortAsc] = useState(true)

  useEffect(() => {
    if (!experimentName || !runId) return
    setLoading(true)
    setError(null)
    fetch(
      `/api/simulator/runs/${encodeURIComponent(experimentName)}/${encodeURIComponent(runId)}/template_stats`
    )
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => { setRows(d); setLoading(false) })
      .catch((e) => { setError(e.message); setLoading(false) })
  }, [experimentName, runId])

  const handleSort = useCallback((key) => {
    setSortKey((prev) => {
      if (prev === key) { setSortAsc((a) => !a); return key }
      setSortAsc(true)
      return key
    })
  }, [])

  const sorted = rows
    ? [...rows].sort((a, b) => {
        const va = a[sortKey]
        const vb = b[sortKey]
        const cmp = va < vb ? -1 : va > vb ? 1 : 0
        return sortAsc ? cmp : -cmp
      })
    : []

  // Summary aggregates
  const summary = rows && rows.length > 0 ? (() => {
    const totalOcc = rows.reduce((s, r) => s + r.occurrences, 0)
    const totalViolations = rows.reduce((s, r) => s + r.violation_rate * r.occurrences, 0)
    const totalViolAmount = rows.reduce((s, r) => s + r.total_violation_amount_s, 0)
    const avgRelViol = rows.reduce((s, r) => s + r.mean_relative_violation * r.occurrences, 0) / totalOcc
    const p50Lat = rows.reduce((s, r) => s + r.p50_latency_s * r.occurrences, 0) / totalOcc
    const p90Lat = rows.reduce((s, r) => s + r.p90_latency_s * r.occurrences, 0) / totalOcc
    const p95Lat = rows.reduce((s, r) => s + r.p95_latency_s * r.occurrences, 0) / totalOcc
    return { 
      totalOcc, 
      violRate: totalViolations / totalOcc, 
      totalViolAmount,
      avgRelViol,
      p50Lat,
      p90Lat,
      p95Lat
    }
  })() : null

  // Close on backdrop click
  function handleBackdropClick(e) {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div className="tsm-backdrop" onClick={handleBackdropClick}>
      <div className="tsm-modal" role="dialog" aria-modal="true">
        <div className="tsm-header">
          <span className="tsm-title">Per-Template Stats — Run {runId}</span>
          <button className="tsm-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="tsm-body">
          {loading && <div className="tsm-status">Loading…</div>}
          {error && <div className="tsm-status tsm-error">{error}</div>}
          {!loading && !error && rows && (
            <div className="tsm-scroll">
              <table className="tsm-table">
                <thead>
                  {/* Row 1: top-level groups */}
                  <tr className="tsm-hdr-top">
                    <th rowSpan={2} className={sortKey === 'template_id' ? 'tsm-sorted' : ''} onClick={() => handleSort('template_id')}>
                      Template{sortKey === 'template_id' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th rowSpan={2} className={sortKey === 'occurrences' ? 'tsm-sorted' : ''} onClick={() => handleSort('occurrences')}>
                      Count{sortKey === 'occurrences' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th rowSpan={2} className={sortKey === 'slo_s' ? 'tsm-sorted' : ''} onClick={() => handleSort('slo_s')}>
                      SLO (s){sortKey === 'slo_s' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th colSpan={3} className="tsm-latency-group">Latency (s)</th>
                    <th colSpan={3} className="tsm-violation-group">SLO Violation</th>
                  </tr>
                  {/* Row 2: sub-columns */}
                  <tr className="tsm-hdr-sub">
                    <th className={sortKey === 'p50_latency_s' ? 'tsm-sorted' : ''} onClick={() => handleSort('p50_latency_s')}>
                      p50{sortKey === 'p50_latency_s' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th className={sortKey === 'p90_latency_s' ? 'tsm-sorted' : ''} onClick={() => handleSort('p90_latency_s')}>
                      p90{sortKey === 'p90_latency_s' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th className={sortKey === 'p95_latency_s' ? 'tsm-sorted' : ''} onClick={() => handleSort('p95_latency_s')}>
                      p95{sortKey === 'p95_latency_s' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th className={sortKey === 'violation_rate' ? 'tsm-sorted' : ''} onClick={() => handleSort('violation_rate')}>
                      Rate{sortKey === 'violation_rate' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th className={sortKey === 'total_violation_amount_s' ? 'tsm-sorted' : ''} onClick={() => handleSort('total_violation_amount_s')}>
                      Total Amount (s){sortKey === 'total_violation_amount_s' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                    <th className={sortKey === 'mean_relative_violation' ? 'tsm-sorted' : ''} onClick={() => handleSort('mean_relative_violation')}>
                      Mean Relative{sortKey === 'mean_relative_violation' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((row) => (
                    <tr key={row.template_id}>
                      <td>{row.template_id}</td>
                      <td>{row.occurrences}</td>
                      <td>{FMT(row.slo_s)}</td>
                      <td>{FMT(row.p50_latency_s)}</td>
                      <td>{FMT(row.p90_latency_s)}</td>
                      <td>{FMT(row.p95_latency_s)}</td>
                      <td className={violationClass(row.violation_rate, 'violation_rate', activeCol)}>
                        {PCT(row.violation_rate)}
                      </td>
                      <td className={violationClass(row.total_violation_amount_s, 'total_violation_amount_s', activeCol)}>
                        {FMT(row.total_violation_amount_s)}
                      </td>
                      <td className={violationClass(row.mean_relative_violation, 'mean_relative_violation', activeCol)}>
                        {PCT(row.mean_relative_violation)}
                      </td>
                    </tr>
                  ))}
                </tbody>
                {summary && (
                  <tfoot>
                    <tr className="tsm-summary-row">
                      <td><em>All templates</em></td>
                      <td>{summary.totalOcc}</td>
                      <td></td>
                      <td></td>
                      <td></td>
                      <td></td>
                      <td className={violationClass(summary.violRate, 'violation_rate', activeCol)}>
                        {PCT(summary.violRate)}
                      </td>
                      <td className={violationClass(summary.totalViolAmount, 'total_violation_amount_s', activeCol)}>
                        {FMT(summary.totalViolAmount)}
                      </td>
                      <td className={violationClass(summary.avgRelViol, 'mean_relative_violation', activeCol)}>
                        {PCT(summary.avgRelViol)}
                      </td>
                    </tr>
                  </tfoot>
                )}
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
