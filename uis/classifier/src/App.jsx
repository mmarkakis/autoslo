import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import { Line } from 'react-chartjs-2'
import 'chartjs-adapter-date-fns'
import './App.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
  zoomPlugin
)

// Classification colors
const CLASSIFICATION_COLORS = {
  windowed: { bg: 'rgba(54, 162, 235, 0.7)', border: 'rgb(54, 162, 235)' },
  normal: { bg: 'rgba(75, 192, 92, 0.7)', border: 'rgb(75, 192, 92)' },
  'ad-hoc': { bg: 'rgba(255, 159, 64, 0.7)', border: 'rgb(255, 159, 64)' },
  unclassified: { bg: 'rgba(201, 203, 207, 0.7)', border: 'rgb(201, 203, 207)' },
}

// Distinct colors for different templates
const TEMPLATE_COLORS = [
  { bg: 'rgba(255, 99, 132, 0.2)', border: 'rgb(255, 99, 132)' },
  { bg: 'rgba(54, 162, 235, 0.2)', border: 'rgb(54, 162, 235)' },
  { bg: 'rgba(255, 206, 86, 0.2)', border: 'rgb(255, 206, 86)' },
  { bg: 'rgba(75, 192, 192, 0.2)', border: 'rgb(75, 192, 192)' },
  { bg: 'rgba(153, 102, 255, 0.2)', border: 'rgb(153, 102, 255)' },
  { bg: 'rgba(255, 159, 64, 0.2)', border: 'rgb(255, 159, 64)' },
  { bg: 'rgba(199, 199, 199, 0.2)', border: 'rgb(199, 199, 199)' },
  { bg: 'rgba(83, 102, 255, 0.2)', border: 'rgb(83, 102, 255)' },
  { bg: 'rgba(255, 99, 255, 0.2)', border: 'rgb(255, 99, 255)' },
  { bg: 'rgba(99, 255, 132, 0.2)', border: 'rgb(99, 255, 132)' },
]

// Bucket size in seconds (default, will be updated from server)
const DEFAULT_BUCKET_SIZE_S = 60

// Helper to format bucket size for display
function formatBucketSize(seconds) {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}min`
  return `${Math.round(seconds / 3600)}hr`
}

// Convert server bucket data to chart format
function bucketsToChartData(buckets, useLogScale = false) {
  return buckets.map(b => ({ 
    x: b.bucket_start_s * 1000, 
    y: useLogScale && b.count === 0 ? 1e-9 : b.count 
  }))
}

// File tree component
function FileTree({ files, onSelect, selectedPath, level = 0 }) {
  const [expanded, setExpanded] = useState({})

  const toggleExpand = (path) => {
    setExpanded(prev => ({ ...prev, [path]: !prev[path] }))
  }

  return (
    <ul className={`file-tree ${level === 0 ? 'root' : ''}`}>
      {files.map((file) => (
        <li key={file.path} className="file-tree-item">
          {file.is_dir ? (
            <>
              <div
                className="file-tree-folder"
                onClick={() => toggleExpand(file.path)}
              >
                <span className="file-icon">{expanded[file.path] ? '📂' : '📁'}</span>
                <span className="file-name">{file.name}</span>
              </div>
              {expanded[file.path] && file.children && (
                <FileTree
                  files={file.children}
                  onSelect={onSelect}
                  selectedPath={selectedPath}
                  level={level + 1}
                />
              )}
            </>
          ) : (
            <div
              className={`file-tree-file ${selectedPath === file.path ? 'selected' : ''}`}
              onClick={() => onSelect(file.path)}
            >
              <span className="file-icon">📄</span>
              <span className="file-name">{file.name}</span>
            </div>
          )}
        </li>
      ))}
    </ul>
  )
}

// Helper to bin classification results client-side (only for chart display)
function binClassificationBuckets(aggregateBuckets, templateBuckets, classifications, classType, useLogScale = false) {
  // Use the same bucket boundaries as the main plot (aggregateBuckets)
  // Sum only the templates that belong to this classification
  
  return aggregateBuckets.map(bucket => {
    const bucketStartS = bucket.bucket_start_s
    let count = 0
    
    // Sum counts from all templates in this classification for this bucket
    for (const [tidStr, data] of Object.entries(templateBuckets)) {
      const tid = parseInt(tidStr, 10)
      if (classifications[tid] !== classType) continue
      
      // Find the matching bucket in this template's data
      const templateBucket = data.buckets.find(b => b.bucket_start_s === bucketStartS)
      if (templateBucket) {
        count += templateBucket.count
      }
    }
    
    return {
      x: bucketStartS * 1000,
      y: useLogScale && count === 0 ? 1e-9 : count
    }
  })
}

function App() {
  const [files, setFiles] = useState([])
  const [loadingFiles, setLoadingFiles] = useState(true)
  const [selectedFilePath, setSelectedFilePath] = useState(null)
  // Aggregated bucket data from server
  const [aggregateBuckets, setAggregateBuckets] = useState([])
  const [templateIds, setTemplateIds] = useState([])
  const [numQueries, setNumQueries] = useState(0)
  const [bucketSizeS, setBucketSizeS] = useState(DEFAULT_BUCKET_SIZE_S)
  // Template-specific bucket data (fetched on demand)
  const [templateBucketsCache, setTemplateBucketsCache] = useState({})
  // Classification results
  const [classifications, setClassifications] = useState({})
  const [classificationBuckets, setClassificationBuckets] = useState({})
  const [summary, setSummary] = useState(null)
  const [isClassified, setIsClassified] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedTemplates, setSelectedTemplates] = useState([])
  // Log scale toggles
  const [mainLogScale, setMainLogScale] = useState(false)
  const [classLogScale, setClassLogScale] = useState({})
  // Modal for classification details
  const [expandedClass, setExpandedClass] = useState(null)
  // Selected templates per classification chart
  const [classSelectedTemplates, setClassSelectedTemplates] = useState({})
  // Bucket size editing
  const [editingBucketSize, setEditingBucketSize] = useState(false)
  const [tempBucketSize, setTempBucketSize] = useState('')

  // Available templates (not yet selected)
  const availableTemplates = useMemo(() => {
    return templateIds.filter(id => !selectedTemplates.includes(id))
  }, [templateIds, selectedTemplates])

  // Convert aggregate buckets to chart format (depends on log scale)
  const chartBuckets = useMemo(() => {
    return bucketsToChartData(aggregateBuckets, mainLogScale)
  }, [aggregateBuckets, mainLogScale])

  // Load file tree on mount
  useEffect(() => {
    const fetchFiles = async () => {
      try {
        const response = await fetch('/api/classifier/files')
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const data = await response.json()
        setFiles(data)
      } catch (err) {
        setError(`Failed to load file list: ${err.message}`)
      } finally {
        setLoadingFiles(false)
      }
    }
    fetchFiles()
  }, [])

  const handleFileSelect = useCallback(async (path, customBucketSize = null) => {
    setSelectedFilePath(path)
    setError(null)
    setIsLoading(true)
    setIsClassified(false)
    setClassifications({})
    setSummary(null)
    setSelectedTemplates([])
    setTemplateBucketsCache({})
    setClassificationBuckets({})
    setClassSelectedTemplates({})
    setExpandedClass(null)

    try {
      let url = `/api/classifier/load?path=${encodeURIComponent(path)}`
      if (customBucketSize) {
        url += `&bucket_size_s=${customBucketSize}`
      }
      const response = await fetch(url)
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${response.status}`)
      }

      const data = await response.json()
      setAggregateBuckets(data.aggregate_buckets || [])
      setTemplateIds(data.template_ids || [])
      setNumQueries(data.num_queries || 0)
      setBucketSizeS(data.bucket_size_s || DEFAULT_BUCKET_SIZE_S)
    } catch (err) {
      setError(`Failed to load file: ${err.message}`)
      setAggregateBuckets([])
      setTemplateIds([])
      setBucketSizeS(DEFAULT_BUCKET_SIZE_S)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleClassify = useCallback(async () => {
    if (!selectedFilePath) return

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `/api/classifier/classify-file?path=${encodeURIComponent(selectedFilePath)}`,
        { method: 'POST' }
      )

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${response.status}`)
      }

      const result = await response.json()

      // Build classification map
      const classMap = {}
      for (const c of result.classifications) {
        classMap[c.template_id] = c.classification
      }

      setClassifications(classMap)
      setClassificationBuckets(result.template_buckets || {})
      setSummary(result.summary)
      setIsClassified(true)
    } catch (err) {
      setError(`Classification failed: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [selectedFilePath])

  const handleReset = useCallback(() => {
    setSelectedFilePath(null)
    setAggregateBuckets([])
    setTemplateIds([])
    setNumQueries(0)
    setBucketSizeS(DEFAULT_BUCKET_SIZE_S)
    setTemplateBucketsCache({})
    setClassifications({})
    setClassificationBuckets({})
    setSummary(null)
    setIsClassified(false)
    setSelectedTemplates([])
    setClassSelectedTemplates({})
    setExpandedClass(null)
    setEditingBucketSize(false)
    setError(null)
  }, [])

  // Handle bucket size change
  const handleBucketSizeEdit = useCallback(() => {
    setTempBucketSize(String(bucketSizeS))
    setEditingBucketSize(true)
  }, [bucketSizeS])

  const handleBucketSizeSubmit = useCallback(() => {
    const newSize = parseInt(tempBucketSize, 10)
    if (!isNaN(newSize) && newSize >= 10 && newSize <= 86400 && selectedFilePath) {
      handleFileSelect(selectedFilePath, newSize)
    }
    setEditingBucketSize(false)
  }, [tempBucketSize, selectedFilePath, handleFileSelect])

  const handleBucketSizeCancel = useCallback(() => {
    setEditingBucketSize(false)
  }, [])

  const handleAddTemplate = useCallback(async (templateId) => {
    if (templateId === null || templateId === undefined || selectedTemplates.includes(templateId) || !selectedFilePath) return

    // Add to selected templates immediately for UI responsiveness
    setSelectedTemplates(prev => [...prev, templateId])

    // If we already have the buckets cached, no need to fetch
    if (templateBucketsCache[templateId]) return

    // Fetch template-specific buckets from server
    try {
      const response = await fetch(
        `/api/classifier/template-buckets/${templateId}?path=${encodeURIComponent(selectedFilePath)}`
      )
      if (!response.ok) {
        console.error(`Failed to fetch template ${templateId} buckets`)
        return
      }
      const data = await response.json()
      setTemplateBucketsCache(prev => ({
        ...prev,
        [templateId]: data.buckets
      }))
    } catch (err) {
      console.error(`Error fetching template ${templateId}:`, err)
    }
  }, [selectedTemplates, selectedFilePath, templateBucketsCache])

  const handleRemoveTemplate = useCallback((templateId) => {
    setSelectedTemplates(prev => prev.filter(id => id !== templateId))
  }, [])

  // Handle template selection for classification charts
  const handleAddClassTemplate = useCallback(async (classType, templateId) => {
    if (templateId === null || templateId === undefined || !selectedFilePath) return
    
    const currentList = classSelectedTemplates[classType] || []
    if (currentList.includes(templateId)) return
    
    setClassSelectedTemplates(prev => ({
      ...prev,
      [classType]: [...(prev[classType] || []), templateId]
    }))

    // Fetch template buckets if not cached
    if (!templateBucketsCache[templateId]) {
      try {
        const response = await fetch(
          `/api/classifier/template-buckets/${templateId}?path=${encodeURIComponent(selectedFilePath)}`
        )
        if (response.ok) {
          const data = await response.json()
          setTemplateBucketsCache(prev => ({
            ...prev,
            [templateId]: data.buckets
          }))
        }
      } catch (err) {
        console.error(`Error fetching template ${templateId}:`, err)
      }
    }
  }, [classSelectedTemplates, selectedFilePath, templateBucketsCache])

  const handleRemoveClassTemplate = useCallback((classType, templateId) => {
    setClassSelectedTemplates(prev => ({
      ...prev,
      [classType]: (prev[classType] || []).filter(id => id !== templateId)
    }))
  }, [])

  // Main chart data (aggregate + selected templates)
  const mainChartData = useMemo(() => {
    if (aggregateBuckets.length === 0) return { datasets: [] }

    const datasets = []

    // Aggregate line (zeros become null if log scale)
    datasets.push({
      label: 'All Arrivals',
      data: chartBuckets,
      borderColor: 'rgb(75, 75, 75)',
      backgroundColor: 'rgba(75, 75, 75, 0.1)',
      borderWidth: 2,
      fill: true,
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 4,
      spanGaps: false,
    })

    // Individual template lines (from cache)
    selectedTemplates.forEach((templateId, idx) => {
      const templateBuckets = templateBucketsCache[templateId]
      if (!templateBuckets) return // Still loading
      const templateData = bucketsToChartData(templateBuckets, mainLogScale)
      const colors = TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length]
      datasets.push({
        label: `Template ${templateId}`,
        data: templateData,
        borderColor: colors.border,
        backgroundColor: colors.bg,
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        spanGaps: false,
      })
    })

    // Compute max and min y values from all datasets
    let maxY = 0
    let minY = Infinity
    for (const ds of datasets) {
      for (const pt of ds.data) {
        if (pt.y !== null && pt.y > maxY) maxY = pt.y
        // For min, ignore placeholder values (1e-9) used for zeros in log scale
        if (pt.y !== null && pt.y >= 1e-8 && pt.y < minY) minY = pt.y
      }
    }
    if (minY === Infinity) minY = 0

    return { datasets, maxY, minY }
  }, [aggregateBuckets, chartBuckets, selectedTemplates, templateBucketsCache, mainLogScale])

  // Classification charts data - one per classification type
  const classificationCharts = useMemo(() => {
    if (!isClassified || Object.keys(classificationBuckets).length === 0 || aggregateBuckets.length === 0 || !summary) return []

    const charts = []
    // Only create charts for classes that appear in summary (same as the cards)
    const classTypes = Object.keys(summary.by_class)

    for (const classType of classTypes) {
      const isLogEnabled = classLogScale[classType] || false
      // Aggregate buckets for this classification type using the same bucket boundaries as the main plot
      const classBuckets = binClassificationBuckets(
        aggregateBuckets, classificationBuckets, classifications, classType, isLogEnabled
      )
      if (classBuckets.length === 0) continue

      // Get templates in this class
      const templatesInClass = []
      let queryCount = 0
      for (const [tidStr, data] of Object.entries(classificationBuckets)) {
        const tid = parseInt(tidStr, 10)
        if (classifications[tid] === classType) {
          templatesInClass.push(tid)
          queryCount += data.total_queries || 0
        }
      }

      const datasets = []

      // Aggregate for this class
      datasets.push({
        label: `All ${classType}`,
        data: classBuckets,
        borderColor: CLASSIFICATION_COLORS[classType].border,
        backgroundColor: CLASSIFICATION_COLORS[classType].bg,
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
        spanGaps: false,
      })

      // Add selected templates for this class
      const selectedForClass = classSelectedTemplates[classType] || []
      selectedForClass.forEach((templateId, idx) => {
        const templateBuckets = templateBucketsCache[templateId]
        if (!templateBuckets) return // Still loading
        const templateData = bucketsToChartData(templateBuckets, isLogEnabled)
        const colors = TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length]
        datasets.push({
          label: `Template ${templateId}`,
          data: templateData,
          borderColor: colors.border,
          backgroundColor: colors.bg,
          borderWidth: 2,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          spanGaps: false,
        })
      })

      // Compute max and min y values from all datasets for this chart
      let maxY = 0
      let minY = Infinity
      for (const ds of datasets) {
        for (const pt of ds.data) {
          if (pt.y !== null && pt.y > maxY) maxY = pt.y
          // For min, ignore placeholder values (1e-9) used for zeros in log scale
          if (pt.y !== null && pt.y >= 1e-8 && pt.y < minY) minY = pt.y
        }
      }
      if (minY === Infinity) minY = 0

      // Available templates = templates in class minus already selected
      const availableForClass = templatesInClass.filter(id => !selectedForClass.includes(id))

      charts.push({
        classification: classType,
        data: { datasets },
        maxY,
        minY,
        templateCount: templatesInClass.length,
        queryCount,
        templatesInClass,
        selectedTemplates: selectedForClass,
        availableTemplates: availableForClass,
      })
    }

    return charts
  }, [isClassified, aggregateBuckets, classificationBuckets, classifications, classLogScale, classSelectedTemplates, templateBucketsCache, summary])

  // Reference to main chart for reset
  const mainChartRef = useRef(null)
  const classChartRefs = useRef({})

  const resetMainZoom = useCallback(() => {
    if (mainChartRef.current) {
      mainChartRef.current.resetZoom()
    }
  }, [])

  const resetClassZoom = useCallback((classType) => {
    if (classChartRefs.current[classType]) {
      classChartRefs.current[classType].resetZoom()
    }
  }, [])

  const toggleMainLogScale = useCallback(() => {
    // Preserve x-axis zoom bounds when toggling scale type
    const chart = mainChartRef.current
    if (chart) {
      const xScale = chart.scales?.x
      if (xScale && xScale.min !== undefined && xScale.max !== undefined) {
        const bounds = { min: xScale.min, max: xScale.max }
        setMainLogScale(prev => !prev)
        // Restore zoom after chart re-renders
        setTimeout(() => {
          if (mainChartRef.current) {
            mainChartRef.current.zoomScale('x', bounds, 'none')
          }
        }, 0)
        return
      }
    }
    setMainLogScale(prev => !prev)
  }, [])

  const toggleClassLogScale = useCallback((classType) => {
    // Preserve x-axis zoom bounds when toggling scale type
    const chart = classChartRefs.current[classType]
    if (chart) {
      const xScale = chart.scales?.x
      if (xScale && xScale.min !== undefined && xScale.max !== undefined) {
        const bounds = { min: xScale.min, max: xScale.max }
        setClassLogScale(prev => ({
          ...prev,
          [classType]: !prev[classType]
        }))
        // Restore zoom after chart re-renders
        setTimeout(() => {
          if (classChartRefs.current[classType]) {
            classChartRefs.current[classType].zoomScale('x', bounds, 'none')
          }
        }, 0)
        return
      }
    }
    setClassLogScale(prev => ({
      ...prev,
      [classType]: !prev[classType]
    }))
  }, [])

  const mainChartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Query Arrivals (drag to zoom, scroll to pan)',
      },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (items.length === 0) return ''
            const date = new Date(items[0].parsed.x)
            return date.toLocaleString()
          },
          label: (context) => {
            const value = context.parsed.y < 1e-8 ? 0 : context.parsed.y
            return `${context.dataset.label}: ${value} queries`
          },
        },
      },
      zoom: {
        pan: {
          enabled: false,
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          drag: {
            enabled: true,
            backgroundColor: 'rgba(54, 162, 235, 0.3)',
            borderColor: 'rgb(54, 162, 235)',
            borderWidth: 1,
          },
          mode: 'x',
        },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM d',
          },
        },
        ticks: {
          autoSkip: true,
          maxTicksLimit: 15,
          callback: function(value, index, ticks) {
            const date = new Date(value)
            // Show date on first tick or when day changes
            if (index === 0) {
              return [date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })]
            }
            // Check if this is a new day compared to previous tick
            if (index > 0 && ticks[index - 1]) {
              const prevDate = new Date(ticks[index - 1].value)
              if (date.getDate() !== prevDate.getDate()) {
                return [date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })]
              }
            }
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
          },
        },
        title: {
          display: true,
          text: 'Time',
        },
      },
      y: {
        type: mainLogScale ? 'logarithmic' : 'linear',
        ...(mainLogScale ? {} : { beginAtZero: false }),
        min: mainLogScale 
          ? (mainChartData.minY > 0 ? mainChartData.minY * 0.5 : undefined)
          : (mainChartData.minY > 0 ? mainChartData.minY * 0.8 : 0),
        max: mainLogScale 
          ? (mainChartData.maxY > 0 ? mainChartData.maxY * 2 : undefined)
          : (mainChartData.maxY > 0 ? Math.ceil(mainChartData.maxY * 1.2) : undefined),
        title: {
          display: true,
          text: `Arrivals per Bucket${mainLogScale ? ' (log)' : ''}`,
        },
      },
    },
  }), [mainLogScale, mainChartData.maxY, mainChartData.minY])

  const classificationChartOptions = useCallback((classType, maxY = 0, minY = 0) => {
    const isLog = classLogScale[classType] || false
    return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: `${classType.charAt(0).toUpperCase() + classType.slice(1)} Templates (drag to zoom)`,
        color: CLASSIFICATION_COLORS[classType]?.border || '#666',
      },
      tooltip: {
        callbacks: {
          title: (items) => {
            if (items.length === 0) return ''
            const date = new Date(items[0].parsed.x)
            return date.toLocaleString()
          },
          label: (context) => {
            const value = context.parsed.y < 1e-8 ? 0 : context.parsed.y
            return `${value} queries`
          },
        },
      },
      zoom: {
        pan: {
          enabled: false,
        },
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          drag: {
            enabled: true,
            backgroundColor: 'rgba(54, 162, 235, 0.3)',
            borderColor: 'rgb(54, 162, 235)',
            borderWidth: 1,
          },
          mode: 'x',
        },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM d',
          },
        },
        ticks: {
          autoSkip: true,
          maxTicksLimit: 10,
          callback: function(value, index, ticks) {
            const date = new Date(value)
            if (index === 0) {
              return [date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })]
            }
            if (index > 0 && ticks[index - 1]) {
              const prevDate = new Date(ticks[index - 1].value)
              if (date.getDate() !== prevDate.getDate()) {
                return [date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })]
              }
            }
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
          },
        },
        title: {
          display: false,
        },
      },
      y: {
        type: isLog ? 'logarithmic' : 'linear',
        ...(isLog ? {} : { beginAtZero: false }),
        min: isLog 
          ? (minY > 0 ? minY * 0.5 : undefined)
          : (minY > 0 ? minY * 0.8 : 0),
        max: isLog 
          ? (maxY > 0 ? maxY * 2 : undefined)
          : (maxY > 0 ? Math.ceil(maxY * 1.2) : undefined),
        title: {
          display: true,
          text: `Arrivals/bucket${isLog ? ' (log)' : ''}`,
        },
      },
    },
  }
  }, [classLogScale])

  return (
    <div className="app">
      <header className="header">
        <h1>Arrival Classifier</h1>
        <p className="subtitle">Analyze and classify query arrival patterns</p>
      </header>

      <main className="main">
        <div className="layout">
          {/* File Browser Sidebar */}
          <aside className="sidebar">
            <div className="sidebar-header">
              <h2>📁 Data Files</h2>
            </div>
            <div className="sidebar-content">
              {loadingFiles ? (
                <div className="sidebar-loading">Loading files...</div>
              ) : files.length === 0 ? (
                <div className="sidebar-empty">No parquet files found in data directory</div>
              ) : (
                <FileTree
                  files={files}
                  onSelect={handleFileSelect}
                  selectedPath={selectedFilePath}
                />
              )}
            </div>
          </aside>

          {/* Main Content */}
          <div className="main-content">
            {aggregateBuckets.length === 0 ? (
              <div className="placeholder">
                <div className="placeholder-icon">📊</div>
                <h2>Select a file to visualize</h2>
                <p>
                  Choose a parquet file from the sidebar to view and classify
                  its arrival time series.
                </p>
              </div>
            ) : (
              <div className="content-scroll">
                <div className="content">
                  <div className="toolbar">
                    <div className="toolbar-info">
                      <span className="badge badge-file" title={selectedFilePath}>
                        📄 {selectedFilePath?.split('/').pop()}
                      </span>
                      <span className="badge">{numQueries.toLocaleString()} queries</span>
                      <span className="badge">{templateIds.length} templates</span>
                      {editingBucketSize ? (
                        <span className="badge bucket-edit">
                          ⏱️
                          <input
                            type="number"
                            value={tempBucketSize}
                            onChange={(e) => setTempBucketSize(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleBucketSizeSubmit()
                              if (e.key === 'Escape') handleBucketSizeCancel()
                            }}
                            min={10}
                            max={86400}
                            className="bucket-input"
                            autoFocus
                          />
                          <span className="bucket-unit">s</span>
                          <button onClick={handleBucketSizeSubmit} className="bucket-btn">✓</button>
                          <button onClick={handleBucketSizeCancel} className="bucket-btn">✕</button>
                        </span>
                      ) : (
                        <span 
                          className="badge badge-clickable" 
                          title="Click to change bucket size"
                          onClick={handleBucketSizeEdit}
                        >
                          ⏱️ {formatBucketSize(bucketSizeS)} buckets
                        </span>
                      )}
                      {isClassified && <span className="badge badge-success">Classified</span>}
                    </div>
                    <div className="toolbar-actions">
                      {!isClassified && (
                        <button
                          onClick={handleClassify}
                          className="btn btn-primary"
                          disabled={isLoading}
                        >
                          {isLoading ? 'Classifying...' : '🔍 Classify'}
                        </button>
                      )}
                      <button onClick={handleReset} className="btn btn-secondary">
                        Reset
                      </button>
                    </div>
                  </div>

                  {/* Template selector */}
                  <div className="template-selector">
                    <div className="template-selector-left">
                      <label htmlFor="template-select">Add template:</label>
                      <select
                        id="template-select"
                        onChange={(e) => {
                          const val = parseInt(e.target.value, 10)
                          if (!isNaN(val)) {
                            handleAddTemplate(val)
                            e.target.value = ''
                          }
                        }}
                        defaultValue=""
                      >
                        <option value="" disabled>Select template...</option>
                        {availableTemplates.map(id => (
                          <option key={id} value={id}>Template {id}</option>
                        ))}
                      </select>
                    </div>
                    {selectedTemplates.length > 0 && (
                      <div className="selected-templates">
                        {selectedTemplates.map((id, idx) => (
                          <span
                            key={id}
                            className="template-tag"
                            style={{
                              borderColor: TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length].border,
                              backgroundColor: TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length].bg,
                            }}
                          >
                            Template {id}
                            <button
                              className="template-tag-remove"
                              onClick={() => handleRemoveTemplate(id)}
                              title="Remove"
                            >
                              ×
                            </button>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Main Chart */}
                  <div className="chart-container">
                    <div className="chart-toolbar">
                      <button
                        onClick={toggleMainLogScale}
                        className={`btn btn-sm ${mainLogScale ? 'btn-sm-active' : ''}`}
                        title="Toggle logarithmic scale"
                      >
                        📊 {mainLogScale ? 'Linear' : 'Log'} Scale
                      </button>
                      <button onClick={resetMainZoom} className="btn btn-sm" title="Reset zoom">
                        🔍 Reset Zoom
                      </button>
                    </div>
                    <Line ref={mainChartRef} data={mainChartData} options={mainChartOptions} />
                  </div>
                </div>

                {/* Classification Results */}
                {isClassified && (
                  <div className="classification-section">
                    <h3>Classification Results</h3>

                    {summary && (
                      <div className="summary">
                        <div className="summary-grid">
                          {Object.entries(summary.by_class).map(([cls, data]) => (
                            <div
                              key={cls}
                              className="summary-card summary-card-clickable"
                              style={{
                                borderLeftColor: CLASSIFICATION_COLORS[cls]?.border || '#ccc',
                              }}
                              onClick={() => setExpandedClass(cls)}
                              title="Click for details"
                            >
                              <div className="summary-card-header">{cls} <span className="expand-icon">↗</span></div>
                              <div className="summary-card-stats">
                                <div>
                                  <strong>{data.num_templates}</strong> templates
                                  <span className="pct">({data.pct_templates.toFixed(1)}%)</span>
                                </div>
                                <div>
                                  <strong>{data.num_queries}</strong> queries
                                  <span className="pct">({data.pct_queries.toFixed(1)}%)</span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Classification Charts */}
                    <div className="classification-charts">
                      {classificationCharts.map(chart => (
                        <div key={chart.classification} className="classification-chart-wrapper">
                          <div className="classification-chart-header">
                            <div className="chart-toolbar-buttons">
                              <select
                                className="template-select-sm"
                                onChange={(e) => {
                                  const val = parseInt(e.target.value, 10)
                                  if (!isNaN(val)) {
                                    handleAddClassTemplate(chart.classification, val)
                                    e.target.value = ''
                                  }
                                }}
                                defaultValue=""
                              >
                                <option value="" disabled>+ Template</option>
                                {chart.availableTemplates.map(id => (
                                  <option key={id} value={id}>T{id}</option>
                                ))}
                              </select>
                              <button
                                onClick={() => toggleClassLogScale(chart.classification)}
                                className={`btn btn-sm ${classLogScale[chart.classification] ? 'btn-sm-active' : ''}`}
                                title="Toggle logarithmic scale"
                              >
                                {classLogScale[chart.classification] ? 'Lin' : 'Log'}
                              </button>
                              <button
                                onClick={() => resetClassZoom(chart.classification)}
                                className="btn btn-sm"
                                title="Reset zoom"
                              >
                                🔍
                              </button>
                            </div>
                          </div>
                          {/* Selected templates for this class */}
                          {chart.selectedTemplates.length > 0 && (
                            <div className="class-selected-templates">
                              {chart.selectedTemplates.map((id, idx) => (
                                <span
                                  key={id}
                                  className="template-tag-sm"
                                  style={{
                                    borderColor: TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length].border,
                                    backgroundColor: TEMPLATE_COLORS[idx % TEMPLATE_COLORS.length].bg,
                                  }}
                                >
                                  T{id}
                                  <button
                                    className="template-tag-remove-sm"
                                    onClick={() => handleRemoveClassTemplate(chart.classification, id)}
                                  >
                                    ×
                                  </button>
                                </span>
                              ))}
                            </div>
                          )}
                          <div className="classification-chart">
                            <Line
                              ref={(el) => { classChartRefs.current[chart.classification] = el }}
                              data={chart.data}
                              options={classificationChartOptions(chart.classification, chart.maxY, chart.minY)}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)} className="error-close">×</button>
          </div>
        )}

        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner"></div>
          </div>
        )}

        {/* Classification Details Modal */}
        {expandedClass && summary?.by_class?.[expandedClass] && (
          <div className="modal-overlay" onClick={() => setExpandedClass(null)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header" style={{ borderLeftColor: CLASSIFICATION_COLORS[expandedClass]?.border }}>
                <h2>{expandedClass.charAt(0).toUpperCase() + expandedClass.slice(1)} Templates</h2>
                <button className="modal-close" onClick={() => setExpandedClass(null)}>×</button>
              </div>
              <div className="modal-body">
                <div className="modal-summary">
                  <strong>{summary.by_class[expandedClass].num_templates}</strong> templates • 
                  <strong> {summary.by_class[expandedClass].num_queries.toLocaleString()}</strong> queries
                  ({summary.by_class[expandedClass].pct_queries.toFixed(1)}% of total)
                </div>
                <div className="modal-templates-list">
                  <table className="templates-table">
                    <thead>
                      <tr>
                        <th>Template</th>
                        <th>Queries</th>
                        {expandedClass === 'windowed' && (
                          <>
                            <th>Period</th>
                            <th>Active</th>
                            <th>Idle %</th>
                          </>
                        )}
                        {(expandedClass === 'normal' || expandedClass === 'ad-hoc') && (
                          <>
                            <th>Days</th>
                            <th>Daily Periodicity (Weekdays)</th>
                            <th>Daily Periodicity (Weekend)</th>
                            <th>Weekly Periodicity</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {summary.by_class[expandedClass].templates?.map(t => (
                        <tr key={t.template_id}>
                          <td>Template {t.template_id}</td>
                          <td>{t.num_queries?.toLocaleString()}</td>
                          {expandedClass === 'windowed' && (
                            <>
                              <td>{t.period_s ? formatBucketSize(t.period_s) : '-'}</td>
                              <td>{t.active_length_s ? formatBucketSize(t.active_length_s) : '-'}</td>
                              <td>{t.idle_ratio ? `${(t.idle_ratio * 100).toFixed(0)}%` : '-'}</td>
                            </>
                          )}
                          {(expandedClass === 'normal' || expandedClass === 'ad-hoc') && (
                            <>
                              <td>{t.num_unique_days || '-'}</td>
                              <td style={{ textAlign: 'center' }}>{t.has_weekday_seasonality ? '✓' : '-'}</td>
                              <td style={{ textAlign: 'center' }}>{t.has_weekend_seasonality ? '✓' : '-'}</td>
                              <td style={{ textAlign: 'center' }}>{t.has_weekly_seasonality ? '✓' : '-'}</td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
