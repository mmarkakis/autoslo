import { useEffect, useState } from 'react'
import GanttChart from './GanttChart.jsx'

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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ fontSize: '0.8rem', color: '#718096', marginBottom: '0.5rem' }}>
        {timeline.total_queries} queries &nbsp;·&nbsp;
        Default SLO: {timeline.default_slo_s}s
        {timeline.slo_dict && Object.keys(timeline.slo_dict).length > 0 && (
          <span style={{ color: '#a0aec0' }}>
            {' '}· {Object.keys(timeline.slo_dict).length} template overrides
          </span>
        )}{' '}
        &nbsp;·&nbsp; violations{' '}
        <span style={{ color: timeline.violation_rate > 0.05 ? '#fc8181' : '#68d391' }}>
          {(timeline.violation_rate * 100).toFixed(1)}%
        </span>{' '}
        &nbsp;·&nbsp; cost ${Number(timeline.total_cost).toFixed(2)}
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <GanttChart intervals={timeline.intervals} sloS={timeline.default_slo_s} />
      </div>
    </div>
  )
}
