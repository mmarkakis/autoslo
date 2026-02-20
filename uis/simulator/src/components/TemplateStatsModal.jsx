import { useEffect, useState, useCallback } from 'react'
import './TemplateStatsModal.css'

const FMT = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d))
const PCT = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

const COLS = [
  { key: 'template_id',    label: 'Template',     sortNum: true  },
  { key: 'slo_s',          label: 'SLO (s)',      sortNum: true  },
  { key: 'occurrences',    label: 'Count',        sortNum: true  },
  { key: 'compliance_rate',label: 'Compliance',   sortNum: true  },
  { key: 'avg_latency_s',  label: 'Avg lat (s)',  sortNum: true  },
  { key: 'p50_latency_s',  label: 'p50 (s)',      sortNum: true  },
  { key: 'p95_latency_s',  label: 'p95 (s)',      sortNum: true  },
  { key: 'avg_excess_s',   label: 'Avg excess (s)',sortNum: true },
]

function complianceClass(rate) {
  if (rate == null) return ''
  if (rate >= 0.95) return 'tsm-good'
  if (rate >= 0.80) return 'tsm-warn'
  return 'tsm-bad'
}

export default function TemplateStatsModal({
  experimentName,
  runId,
  onClose,
}) {
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
    const totalComp = rows.reduce((s, r) => s + r.compliant, 0)
    const avgLat = rows.reduce((s, r) => s + r.avg_latency_s * r.occurrences, 0) / totalOcc
    const avgExcess = rows.reduce((s, r) => s + r.avg_excess_s * r.occurrences, 0) / totalOcc
    return { totalOcc, compRate: totalComp / totalOcc, avgLat, avgExcess }
  })() : null

  // Close on backdrop click
  function handleBackdropClick(e) {
    if (e.target === e.currentTarget) onClose()
  }

  return (
    <div className="tsm-backdrop" onClick={handleBackdropClick}>
      <div className="tsm-modal" role="dialog" aria-modal="true">
        <div className="tsm-header">
          <span className="tsm-title">Per-template stats — run {runId}</span>
          <button className="tsm-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="tsm-body">
          {loading && <div className="tsm-status">Loading…</div>}
          {error && <div className="tsm-status tsm-error">{error}</div>}
          {!loading && !error && rows && (
            <div className="tsm-scroll">
              <table className="tsm-table">
                <thead>
                  <tr>
                    {COLS.map(({ key, label }) => (
                      <th
                        key={key}
                        className={sortKey === key ? 'tsm-sorted' : ''}
                        onClick={() => handleSort(key)}
                      >
                        {label}{sortKey === key ? (sortAsc ? ' ▲' : ' ▼') : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((row) => (
                    <tr key={row.template_id}>
                      <td>{row.template_id}</td>
                      <td>{FMT(row.slo_s)}</td>
                      <td>{row.occurrences}</td>
                      <td className={complianceClass(row.compliance_rate)}>
                        {PCT(row.compliance_rate)}
                      </td>
                      <td>{FMT(row.avg_latency_s)}</td>
                      <td>{FMT(row.p50_latency_s)}</td>
                      <td>{FMT(row.p95_latency_s)}</td>
                      <td>{FMT(row.avg_excess_s)}</td>
                    </tr>
                  ))}
                </tbody>
                {summary && (
                  <tfoot>
                    <tr className="tsm-summary-row">
                      <td colSpan={2}><em>All templates</em></td>
                      <td>{summary.totalOcc}</td>
                      <td className={complianceClass(summary.compRate)}>
                        {PCT(summary.compRate)}
                      </td>
                      <td>{FMT(summary.avgLat)}</td>
                      <td>—</td>
                      <td>—</td>
                      <td>{FMT(summary.avgExcess)}</td>
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
