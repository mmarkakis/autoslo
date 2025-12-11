import { useEffect, useState } from 'react'
import DesignerTab from './components/DesignerTab'
import PerformanceTab from './components/PerformanceTab'


export default function App() {
  const [activeTab, setActiveTab] = useState('designer')
  const [workloads, setWorkloads] = useState([])
  const [loadingWorkloads, setLoadingWorkloads] = useState(false)

  const openRedoc = () => {
    window.open('/api/redoc', '_blank', 'noopener,noreferrer')
  }

  const refreshWorkloadList = async () => {
    setLoadingWorkloads(true)
    try {
      const res = await fetch('/api/composite')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const list = await res.json()
      const names = Array.isArray(list) ? list : []
      setWorkloads(names)
      return names
    } catch (err) {
      console.error('Failed to load composite workloads:', err)
      setWorkloads([])
      return []
    } finally {
      setLoadingWorkloads(false)
    }
  }

  useEffect(() => {
    refreshWorkloadList()
  }, [])

  return (
    <div className="app">
      <header className="hdr">
        <h1>AutoSLO Studio</h1>
        <button className="secondary" onClick={openRedoc} title="Open API docs (ReDoc)">
          API docs
        </button>
      </header>

      <nav className="tabs">
        <button
          type="button"
          className={`tab ${activeTab === 'designer' ? 'active' : ''}`}
          onClick={() => setActiveTab('designer')}
        >
          Workload designer
        </button>
        <button
          type="button"
          className={`tab ${activeTab === 'performance' ? 'active' : ''}`}
          onClick={() => setActiveTab('performance')}
        >
          Workload performance
        </button>
      </nav>

      {activeTab === 'designer' ? (
        <DesignerTab
          workloads={workloads}
          loading={loadingWorkloads}
          onRefreshWorkloads={refreshWorkloadList}
        />
      ) : (
        <PerformanceTab
          workloads={workloads}
          loading={loadingWorkloads}
          onRefreshWorkloads={refreshWorkloadList}
        />
      )}
    </div>
  )
}
