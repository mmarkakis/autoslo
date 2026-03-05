import { useEffect, useState } from 'react'
import GanttChart from './GanttChart.jsx'

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

const ACTIVE_METRIC_VALUE = {
  binary: (t) => t.violation_rate,
  absolute_s: (t) => t.violation_amount_s,
  relative: (t) => t.violation_relative_mean,
}

export default function GanttViewer({ experimentName, runId }) {
  const [timeline, setTimeline] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!experimentName || !runId) return
    setLoading(true)
    setError(null)
    setTimeline(null)

    const url = `/api/simulator/runs/${encodeURIComponent(experimentName)}/${encodeURIComponent(runId)}/timeline`
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => {
        setTimeline(d)
        setLoading(false)
      })
      .catch((e) => {
        setError(e.message)
        setLoading(false)
      })
  }, [experimentName, runId])

  if (loading) {
    return (
      <div style={{ padding: '1rem', color: '#718096', fontSize: '0.875rem' }}>
        Loading timeline…
      </div>
    )
  }
  if (error) {
    return (
      <div style={{ padding: '1rem', color: '#fc8181', fontSize: '0.875rem' }}>
        {error}
      </div>
    )
  }
  if (!timeline) return null

  const metric = timeline.slo_metric
  const metricLabel = metric ? (METRIC_LABELS[metric] ?? metric) : null
  const metricColor = metric ? (METRIC_COLORS[metric] ?? '#a0aec0') : '#a0aec0'

  // Determine color for the actively-optimized metric (green/red vs threshold)
  const activeVal = metric && ACTIVE_METRIC_VALUE[metric] ? ACTIVE_METRIC_VALUE[metric](timeline) : null
  const activeBad = timeline.slo_threshold != null && activeVal != null && activeVal > timeline.slo_threshold
  const activeColor = activeBad ? '#fc8181' : '#68d391'
  const metricCellColor = (m) => m === metric ? activeColor : '#e2e8f0'

  const ROW = { display: 'block', fontSize: '0.8rem', color: '#718096', lineHeight: 1.9 }
  const SEP = <span style={{ color: '#4a5568' }}> &nbsp;·&nbsp; </span>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ marginBottom: '0.5rem' }}>
        {/* Line 1: queries + cost */}
        <span style={ROW}>
          Num Queries: <span style={{ color: '#cbd5e0' }}>{timeline.total_queries}</span>
          {SEP}
          Total Cost: <span style={{ color: '#cbd5e0' }}>${Number(timeline.total_cost).toFixed(2)}</span>
        </span>

        {/* Line 2: default SLO + template overrides */}
        <span style={ROW}>
          Default SLO: <span style={{ color: '#cbd5e0' }}>{timeline.default_slo_s}s</span>
          {timeline.slo_dict && Object.keys(timeline.slo_dict).length > 0 && (
            <>{SEP}<span style={{ color: '#a0aec0' }}>{Object.keys(timeline.slo_dict).length} template overrides</span></>
          )}
        </span>

        {/* Line 3: metric badge + threshold */}
        <span style={{ ...ROW, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span>SLO Metric Optimized:</span>
          {metricLabel ? (
            <span
              style={{
                display: 'inline-block',
                padding: '0.1rem 0.45rem',
                borderRadius: '4px',
                background: '#2d3748',
                color: metricColor,
                fontWeight: 600,
                fontSize: '0.72rem',
              }}
            >
              {metricLabel}
            </span>
          ) : <span style={{ color: '#4a5568' }}>—</span>}
          {timeline.slo_threshold != null && (
            <><span style={{ color: '#4a5568' }}>·</span><span>Threshold:</span> <span style={{ color: '#cbd5e0' }}>{timeline.slo_threshold}</span></>
          )}
        </span>

        {/* Line 4: all three violation metrics; active one is green/red, others white */}
        <span style={ROW}>
          SLO Violation Rate{' '}
          <span style={{ color: metricCellColor('binary') }}>
            {(timeline.violation_rate * 100).toFixed(1)}%
          </span>
          {SEP}
          Avg SLO Violation Amount{' '}
          <span style={{ color: metricCellColor('absolute_s') }}>
            {Number(timeline.violation_amount_s).toFixed(3)}s
          </span>
          {SEP}
          Avg Relative SLO Violation{' '}
          <span style={{ color: metricCellColor('relative') }}>
            {(timeline.violation_relative_mean * 100).toFixed(2)}%
          </span>
        </span>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <GanttChart
          intervals={timeline.intervals}
          sloS={timeline.default_slo_s}
          sloMetric={metric}
        />
      </div>
    </div>
  )
}
