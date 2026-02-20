import { useState } from 'react'
import ExperimentSidebar from './components/ExperimentSidebar.jsx'
import RunSummaryTable from './components/RunSummaryTable.jsx'
import GanttViewer from './components/GanttViewer.jsx'
import './App.css'

export default function App() {
  const [selectedExperiment, setSelectedExperiment] = useState(null)
  const [openGanttTabs, setOpenGanttTabs] = useState([]) // [{ runId, runIndex }]
  const [activeTabKey, setActiveTabKey] = useState('summary') // 'summary' | runId

  function handleSelectExperiment(name) {
    setSelectedExperiment(name)
    setOpenGanttTabs([])
    setActiveTabKey('summary')
  }

  function handleSelectRun(runId, runIndex) {
    setOpenGanttTabs((prev) => {
      if (prev.some((t) => t.runId === runId)) return prev
      return [...prev, { runId, runIndex }]
    })
    setActiveTabKey(runId)
  }

  function closeTab(runId) {
    setOpenGanttTabs((prev) => {
      const idx = prev.findIndex((t) => t.runId === runId)
      const next = prev.filter((t) => t.runId !== runId)
      if (activeTabKey === runId) {
        const newActive = next[idx - 1]?.runId ?? next[0]?.runId ?? 'summary'
        setActiveTabKey(newActive)
      }
      return next
    })
  }

  const activeGanttRunId = activeTabKey !== 'summary' ? activeTabKey : null

  return (
    <div className="app-layout">
      <header className="app-header">
        <h1>Simulator Viewer</h1>
        {selectedExperiment && (
          <span className="breadcrumb">{selectedExperiment}</span>
        )}
      </header>

      <div className="app-body">
        <ExperimentSidebar
          selected={selectedExperiment}
          onSelect={handleSelectExperiment}
        />

        <main className="main-panel">
          {!selectedExperiment ? (
            <div className="empty-state">
              Select an experiment from the sidebar.
            </div>
          ) : (
            <>
              <div className="tab-bar">
                <button
                  className={activeTabKey === 'summary' ? 'tab active' : 'tab'}
                  onClick={() => setActiveTabKey('summary')}
                >
                  Run Summary
                </button>
                {openGanttTabs.map(({ runId, runIndex }) => (
                  <span
                    key={runId}
                    className={`tab-wrap${activeTabKey === runId ? ' active' : ''}`}
                  >
                    <button
                      className="tab"
                      onClick={() => setActiveTabKey(runId)}
                    >
                      Run {runIndex}
                    </button>
                    <button
                      className="tab-close"
                      onClick={(e) => { e.stopPropagation(); closeTab(runId) }}
                      title="Close"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>

              <div className="tab-content">
                {activeTabKey === 'summary' && (
                  <RunSummaryTable
                    experimentName={selectedExperiment}
                    selectedRun={activeGanttRunId}
                    onSelectRun={handleSelectRun}
                  />
                )}
                {openGanttTabs.map(({ runId }) => (
                  <div
                    key={runId}
                    style={{
                      display: activeTabKey === runId ? 'flex' : 'none',
                      flexDirection: 'column',
                      height: '100%',
                    }}
                  >
                    <GanttViewer
                      experimentName={selectedExperiment}
                      runId={runId}
                    />
                  </div>
                ))}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}
