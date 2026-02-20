import { useEffect, useState } from 'react'
import './ExperimentSidebar.css'

export default function ExperimentSidebar({ selected, onSelect }) {
  const [experiments, setExperiments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    fetch('/api/simulator/experiments')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data) => {
        setExperiments(data)
        setLoading(false)
      })
      .catch((e) => {
        setError(e.message)
        setLoading(false)
      })
  }, [])

  return (
    <aside className="sidebar">
      <div className="sidebar-header">Experiments</div>
      {loading && <div className="sidebar-status">Loading…</div>}
      {error && <div className="sidebar-status error">{error}</div>}
      {!loading && !error && experiments.length === 0 && (
        <div className="sidebar-status muted">No experiments found.</div>
      )}
      <ul className="sidebar-list">
        {experiments.map((name) => (
          <li
            key={name}
            className={`sidebar-item${selected === name ? ' selected' : ''}`}
            onClick={() => onSelect(name)}
          >
            {name}
          </li>
        ))}
      </ul>
    </aside>
  )
}
