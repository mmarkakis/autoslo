import { Fragment, useEffect, useMemo, useState } from 'react'


// Weekday names and label helper (Monday = 1)
const WEEKDAYS_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const dayLabel = (n) => `${WEEKDAYS_SHORT[(n - 1) % 7]} (${n})`

// Control the visible order
const makeDay = (n) => ({ id: `d${n}`, label: dayLabel(n) })

// Helper to make stable ids for "chunk types" and unique ids for instances
const chunkTypeId = (h, t) => `h${h}_t${t}`
const makeInstId = () =>
  (globalThis.crypto?.randomUUID?.() ?? `i_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`)

/**
 * Catalog of chunk TYPES formed by the grid (shape x color).
 * Each item is a type; dragging from palette spawns a new INSTANCE of that type.
 */
const buildCatalog = (hShapeMap, tColorMap, hLevels, tLevels) => {
  const items = []
  for (const t of tLevels) {
    for (const h of hLevels) {
      items.push({
        typeId: chunkTypeId(h, t),
        h,
        t,
        shapeChar: hShapeMap[h],
        colorHex: tColorMap[t],
      })
    }
  }
  return items
}

export default function App() {
  // days shown (start with 7)
  const [days, setDays] = useState(() => Array.from({ length: 7 }, (_, i) => makeDay(i + 1)))

  // Each day stores an array of INSTANCES: { instId, typeId }, keyed by day.id
  const [dayAssignments, setDayAssignments] = useState(
    () => Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`d${i + 1}`, []]))
  )
  const [dragOverDay, setDragOverDay] = useState(null)

  // highlight state for trash zone
  const [dragOverTrash, setDragOverTrash] = useState(false)

  const onDragOverTrash = (e) => {
    e.preventDefault()
    setDragOverTrash(true)
  }
  const onDragLeaveTrash = () => setDragOverTrash(false)
  const onDropToTrash = (e) => {
    const payload = getDragPayload(e)
    setDragOverTrash(false)
    if (!payload) return
    // Only remove existing instances dragged from a day
    if (payload.from === 'day') {
      const { day: fromDay, instId } = payload
      if (!fromDay || !instId) return
      setDayAssignments(prev => {
        const next = structuredClone(prev)
        next[fromDay] = (next[fromDay] ?? []).filter(x => x.instId !== instId)
        return next
      })
    }
  }

  // Maps fetched from backend
  const [hShapeMap, setHShapeMap] = useState(null)
  const [tColorMap, setTColorMap] = useState(null)

  // Fetch legend/meta from backend
  useEffect(() => {
    let cancelled = false
    fetch('/api/chunk/graphics')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(meta => {
        if (cancelled) return
        // Expect { H_SHAPE_MAP: {...}, T_COLOR_MAP: {...} }
        setHShapeMap(meta?.H_SHAPE_MAP ?? null)
        setTColorMap(meta?.T_COLOR_MAP ?? null)
      })
      .catch(err => {
        console.error('Failed to load chunk meta:', err)
      })
    return () => { cancelled = true }
  }, [])

  // Derive visible order from backend maps
  const H_LEVELS = useMemo(() => {
    if (!hShapeMap) return []
    return Object.keys(hShapeMap).map(Number).sort((a, b) => a - b) // left → right
  }, [hShapeMap])
  const T_LEVELS = useMemo(() => {
    if (!tColorMap) return []
    return Object.keys(tColorMap).map(Number).sort((a, b) => b - a) // top → bottom
  }, [tColorMap])

  // Catalog built only after maps are available
  const CATALOG = useMemo(() => {
    if (!hShapeMap || !tColorMap) return []
    return buildCatalog(hShapeMap, tColorMap, H_LEVELS, T_LEVELS)
  }, [hShapeMap, tColorMap, H_LEVELS, T_LEVELS])

  // Fast lookup from typeId → type info
  const typeById = useMemo(
    () => Object.fromEntries(CATALOG.map(c => [c.typeId, c])),
    [CATALOG]
  )

  // ---- Drag payload helpers -------------------------------------------------
  const setDragPayload = (e, payload) => {
    e.dataTransfer.setData('application/json', JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'move'
  }
  const getDragPayload = (e) => {
    try {
      const raw = e.dataTransfer.getData('application/json')
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  }

  // From palette: create a NEW instance of a type
  const onDragStartFromPalette = (e, typeId) => {
    setDragPayload(e, { from: 'palette', typeId })
  }

  // From day: move an existing instance between days
  const onDragStartInstance = (e, dayId, instId) => {
    setDragPayload(e, { from: 'day', day: dayId, instId })
  }

  const onDragOverDay = (e, day) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverDay(day)
  }
  const onDragLeaveDay = (day) => {
    setDragOverDay(prev => prev === day ? null : prev)
  }

  const onDropToDay = (e, targetDay) => {
    const payload = getDragPayload(e)
    setDragOverDay(null)
    if (!payload) return

    setDayAssignments(prev => {
      const next = structuredClone(prev)

      if (payload.from === 'palette') {
        const { typeId } = payload
        next[targetDay] = [...next[targetDay], { instId: makeInstId(), typeId }]
        return next
      }

      if (payload.from === 'day') {
        const { day: fromDay, instId } = payload
        if (!fromDay || !instId) return prev
        const idx = next[fromDay].findIndex(x => x.instId === instId)
        if (idx === -1) return prev
        const [inst] = next[fromDay].splice(idx, 1)
        next[targetDay].push(inst)
        return next
      }

      return prev
    })
  }

  const onRemoveFromDay = (day, instId) => {
    setDayAssignments(prev => ({
      ...prev,
      [day]: prev[day].filter(x => x.instId !== instId)
    }))
  }

  // JSON you’ll POST to Python (preserves order within day)
  const buildWorkloadJSON = () => {
    const payload = {}
    for (const d of days) {
      payload[d.label] = (dayAssignments[d.id] ?? []).map(({ typeId }) => {
        const { h, t } = typeById[typeId]
        return { H: h, T: t }           // compact, Python-friendly
      })
    }
    return payload
  }

  const handleRun = () => {
    const payload = buildWorkloadJSON()
    console.log('Would POST payload:', payload)
    alert('Check the console for the workload JSON (icons only, duplicates allowed).')
  }

  const handleReset = () => {
    setDayAssignments(Object.fromEntries(days.map(d => [d.id, []])))
  }

  const handleAddDay = () => {
    setDays(prev => {
      const nextIdx = prev.length + 1
      const newDay = makeDay(nextIdx)
      setDayAssignments(prevDA => ({ ...prevDA, [newDay.id]: [] }))
      return [...prev, newDay]
    })
  }

  const handleAddWeek = () => {
    setDays(prev => {
      const startIdx = prev.length + 1
      const newDays = Array.from({ length: 7 }, (_, i) => makeDay(startIdx + i))
      setDayAssignments(prevDA => ({
        ...prevDA,
        ...Object.fromEntries(newDays.map(d => [d.id, []])),
      }))
      return [...prev, ...newDays]
    })
  }

  // Remove the last 7 days (keep at least one), and clean their assignments
  const handleRemoveWeek = () => {
    setDays(prev => {
      if (prev.length <= 1) return prev
      const removeCount = Math.min(7, prev.length - 1)
      const removed = prev.slice(-removeCount)
      setDayAssignments(prevDA => {
        const next = { ...prevDA }
        for (const d of removed) delete next[d.id]
        return next
      })
      setDragOverDay(d => (removed.some(r => r.id === d) ? null : d))
      return prev.slice(0, prev.length - removeCount)
    })
  }

  const handleRemoveDay = () => {
    setDays(prev => {
      if (prev.length <= 1) return prev
      const last = prev[prev.length - 1]
      setDayAssignments(prevDA => {
        const { [last.id]: _omit, ...rest } = prevDA
        return rest
      })
      setDragOverDay(d => (d === last.id ? null : d))
      return prev.slice(0, -1)
    })
  }

  // New: open ReDoc docs in a new tab
  const openRedoc = () => {
    window.open('/api/redoc', '_blank', 'noopener,noreferrer')
  }

  const [workloads, setWorkloads] = useState([])
  const [selectedWorkload, setSelectedWorkload] = useState('')
  const [dataPath, setDataPath] = useState('')
  const [newWorkloadName, setNewWorkloadName] = useState('')
  const [loadingWorkloads, setLoadingWorkloads] = useState(false)

  // SLO pane state
  const [sloWorkloadName, setSloWorkloadName] = useState('')
  const [sloSeconds, setSloSeconds] = useState('')
  const [sloPercentile, setSloPercentile] = useState('95')
  // NEW: calculate state
  const [sloLoading, setSloLoading] = useState(false)
  const [sloSeries, setSloSeries] = useState([])
  const [sloError, setSloError] = useState('')

  // load list of composite workloads from the Python API
  useEffect(() => {
    refreshWorkloadList()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // NEW: reusable fetch for workload list
  const refreshWorkloadList = async () => {
    try {
      setLoadingWorkloads(true)
      const res = await fetch('/api/composite')
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const list = await res.json()
      const names = Array.isArray(list) ? list : []
      setWorkloads(names)
      setSelectedWorkload(prev => (prev && names.includes(prev)) ? prev : (names[0] ?? ''))
      // sync SLO workload selection
      setSloWorkloadName(prev => (prev && names.includes(prev)) ? prev : (names[0] ?? ''))
    } catch (err) {
      console.error('Failed to load composite workloads:', err)
    } finally {
      setLoadingWorkloads(false)
    }
  }

  // fetch contents of a named workload (tries two common endpoints for compatibility)
  const loadCompositeWorkload = async (name) => {
    if (!name) return null
    const tryFetch = async (url) => {
      const res = await fetch(url)
      if (!res.ok) return null
      // server may return JSON or plain text; prefer text
      const text = await res.text()
      return text
    }

    const encoded = encodeURIComponent(name)
    const candidates = [
      `/api/composite/${encoded}`
    ]

    for (const url of candidates) {
      const body = await tryFetch(url)
      if (body !== null) {
        return body
      }
    }

    throw new Error('Workload not found on server')
  }

  // Load selected workload from library and populate the workload pane
  const handleLoadLibraryWorkload = async () => {
    try {
      if (!selectedWorkload) return
      const body = await loadCompositeWorkload(selectedWorkload)

      // Prefer JSON; the server route returns JSON
      let def = body
      if (typeof body === 'string') {
        try { def = JSON.parse(body) } catch { /* ignore */ }
      }
      if (!def || typeof def !== 'object') {
        alert('Unsupported workload format from server.')
        return
      }

      // Normalize into ordered labels[] and per-day chunk arrays
      const toNum = (v) => (typeof v === 'number' ? v : (v != null && !Number.isNaN(Number(v)) ? Number(v) : NaN))
      const mondayIndex = Number.isInteger(def?.monday_index)
        ? def.monday_index % 7
        : (typeof def?.monday_index === 'number' ? Math.trunc(def.monday_index) % 7 : 0)

      let perDay = []
      let orderCount = 0

      if (Array.isArray(def.days)) {
        // YAML format: days: [ { chunks: [ {H,T,...}, ... ] }, ... ]
        perDay = def.days.map(d => {
          const arr = Array.isArray(d?.chunks) ? d.chunks : (Array.isArray(d) ? d : [])
          return Array.isArray(arr) ? arr : []
        })
        orderCount = perDay.length
      } else {
        // Accept top-level days/workload/schedule map, or a direct day->list mapping
        const pickDayMap = (obj) => {
          for (const key of ['days', 'workload', 'schedule']) {
            const v = obj[key]
            if (v && typeof v === 'object' && !Array.isArray(v)) return v
          }
          if (obj && typeof obj === 'object' && Object.values(obj).every(Array.isArray)) return obj
          return null
        }
        const dayMap = pickDayMap(def)
        if (!dayMap) {
          alert('Workload definition missing a day-to-items mapping.')
          return
        }
        let keys = Object.keys(dayMap)
        const dayNum = (s) => {
          const m = /^day\s*(\d+)$/i.exec(String(s).trim())
          return m ? parseInt(m[1], 10) : null
        }
        if (keys.every(l => dayNum(l) != null)) {
          keys = keys.sort((a, b) => dayNum(a) - dayNum(b))
        }
        perDay = keys.map(l => (Array.isArray(dayMap[l]) ? dayMap[l] : []))
        orderCount = perDay.length
      }

      // Insert empty leading days equal to monday_index, then fill
      const leadEmpty = Math.max(0, Number.isFinite(mondayIndex) ? mondayIndex : 0)
      const totalDays = leadEmpty + orderCount

      const newDays = Array.from({ length: totalDays }, (_, i) => ({
        id: `d${i + 1}`,
        label: dayLabel(i + 1),
      }))
      const newAssignments = Object.fromEntries(newDays.map(d => [d.id, []]))

      for (let i = 0; i < orderCount; i++) {
        const dayId = `d${leadEmpty + i + 1}`
        const items = Array.isArray(perDay[i]) ? perDay[i] : []
        const mapped = []
        for (const it of items) {
          if (!it || typeof it !== 'object') continue
          const h = toNum(it.H ?? it.h)
          const t = toNum(it.T ?? it.t)
          if (Number.isNaN(h) || Number.isNaN(t)) continue
          const typeId = chunkTypeId(h, t)
          if (!typeById[typeId]) continue
          mapped.push({ instId: makeInstId(), typeId })
        }
        newAssignments[dayId] = mapped
      }

      setDays(newDays)
      setDayAssignments(newAssignments)
      setDragOverDay(null)
      setDragOverTrash(false)
    } catch (err) {
      console.error('Failed to load workload:', err)
      alert('Failed to load workload. See console for details.')
    }
  }

  // Build a composite definition from current UI state for creation
  const buildCompositeDefinitionForCreate = () => {
    const name = newWorkloadName.trim()
    if (!name) return null

    // Map each day to [{H,T}, ...]
    const dayChunks = days.map(d =>
      (dayAssignments[d.id] ?? []).map(({ typeId }) => {
        const { h, t } = typeById[typeId]
        return { H: h, T: t }
      })
    )

    // Compute leading empty days for monday_index
    let leadEmpty = dayChunks.findIndex(arr => Array.isArray(arr) && arr.length > 0)
    if (leadEmpty === -1) leadEmpty = 0

    // Trim trailing empties for a clean payload
    let end = dayChunks.length
    while (end > leadEmpty && (dayChunks[end - 1]?.length ?? 0) === 0) end--

    // Each day must be an object with a "chunks" array
    const payloadDays = dayChunks.slice(leadEmpty, end).map(chs => ({ chunks: chs }))

    return {
      name,
      monday_index: leadEmpty % 7,
      days: payloadDays,
    }
  }

  // Create composite by calling the backend. Tries POST first, then GET fallback.
  const handleCreateComposite = async () => {
    try {
      const def = buildCompositeDefinitionForCreate()
      if (!def) {
        alert('Please enter a workload name.')
        return
      }

      // Try POST JSON
      let res = await fetch('/api/composite/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(def),
      })

      // Fallback to GET with query param if POST not available
      if (!res.ok) {
        const qs = new URLSearchParams({ workload_definition: JSON.stringify(def) }).toString()
        res = await fetch(`/api/composite/create?${qs}`, { method: 'GET' })
      }
      if (!res.ok) {
        const msg = await res.text().catch(() => '')
        throw new Error(`Create failed: HTTP ${res.status} ${msg}`)
      }

      // Refresh workloads list and select the new one
      const listRes = await fetch('/api/composite')
      if (listRes.ok) {
        const list = await listRes.json()
        setWorkloads(Array.isArray(list) ? list : [])
      }
      setSelectedWorkload(def.name)
      alert('Workload created.')
    } catch (err) {
      console.error('Create workload error:', err)
      alert('Failed to create workload. See console for details.')
    }
  }

  const sloSecondsNum = Number(sloSeconds)
  const sloPercentileNum = Number(sloPercentile)
  const validSloSeconds = sloSeconds.trim() !== '' && Number.isFinite(sloSecondsNum) && sloSecondsNum > 0
  const validSloPercentile = [90, 95, 99].includes(sloPercentileNum)

  // NEW: calculate SLO adherence series
  const handleCalculateSLO = async () => {
    if (!sloWorkloadName || !validSloSeconds || !validSloPercentile) return
    try {
      setSloLoading(true)
      setSloError('')
      const qs = new URLSearchParams({
        workload_name: sloWorkloadName,
        tail_slo_s: String(sloSecondsNum),
        percentile: String(sloPercentileNum),
      }).toString()
      const res = await fetch(`/api/strat/cheapest_adherent_cluster?${qs}`, { method: 'POST' })
      if (!res.ok) {
        const msg = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status} ${msg}`)
      }
      const data = await res.json()
      setSloSeries(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('SLO calculate error:', err)
      setSloError('Failed to calculate. See console.')
      setSloSeries([])
    } finally {
      setSloLoading(false)
    }
  }

  // NEW: auto-invoke calculate on workload/SLO/percentile changes (debounced)
  useEffect(() => {
    if (sloLoading) return
    if (!sloWorkloadName || !validSloSeconds || !validSloPercentile) return
    const t = setTimeout(() => { handleCalculateSLO() }, 200)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sloWorkloadName, sloSeconds, sloPercentile, validSloSeconds, validSloPercentile, sloLoading])

  return (
    <div className="app">
      <header className="hdr">
        <h1>AutoSLO Studio</h1>
        {/* New: API docs button */}
        <button className="secondary" onClick={openRedoc} title="Open API docs (ReDoc)">
          API docs
        </button>
      </header>

      <div className="row">
        <div className="left-col">
          {/* Palette: dynamic grid */}
          <section className="card">
            <h2>Chunk Palette</h2>
            <PaletteGrid
              catalog={CATALOG}
              hLevels={H_LEVELS}
              tLevels={T_LEVELS}
              onDragStartFromPalette={onDragStartFromPalette}
            />

            {/* Remove tools: drop-zone + remove-all */}
            <div className="palette-actions">
              <div
                className={`trash-zone ${dragOverTrash ? 'over' : ''}`}
                onDragOver={onDragOverTrash}
                onDragLeave={onDragLeaveTrash}
                onDrop={onDropToTrash}
                title="Drag a placed icon here to remove it"
              >
                <span className="action-emoji" aria-hidden>🗑️</span>
              </div>
              <button className="secondary" onClick={handleReset}>
                <span className="action-emoji" aria-hidden>🧹</span>
                <span className="action-text">Clear All</span>
              </button>
            </div>
          </section>

          {/* Workload library (right under the palette, no gap) */}
          <section className="card library-card">
            <h2>Workload library</h2>
            <select
              value={selectedWorkload}
              onChange={(e) => setSelectedWorkload(e.target.value)}
              aria-label="Select a composite workload"
            >
              {workloads.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
              {workloads.length === 0 && <option value="" disabled>(none found)</option>}
            </select>

            {/* Load + Refresh buttons */}
            <div className="btns">
              <button onClick={handleLoadLibraryWorkload} disabled={!selectedWorkload}>Load</button>
              <button className="secondary" onClick={refreshWorkloadList} disabled={loadingWorkloads}>
                🔄 Refresh
              </button>
            </div>

            <div className="library-note">
              {workloads.length
                ? `Loaded ${workloads.length}`
                : `No workloads found.`}
            </div>
          </section>
        </div>

        {/* Week grid (right column) */}
        <section className="card workload-card">
          <div className="workload-header">
            <h2>Workload</h2>
            <div className="days-toolbar">
              {/* Day controls: − day + */}
              <div className="toolbar-group">
                <button className="btn-icon minus" onClick={handleRemoveDay} aria-label="Remove day">−</button>
                <span className="toolbar-label">day</span>
                <button className="btn-icon plus" onClick={handleAddDay} aria-label="Add day">+</button>
              </div>
              {/* Week controls: − week + */}
              <div className="toolbar-group">
                <button className="btn-icon minus" onClick={handleRemoveWeek} aria-label="Remove week">−</button>
                <span className="toolbar-label">week</span>
                <button className="btn-icon plus" onClick={handleAddWeek} aria-label="Add week">+</button>
              </div>
            </div>
          </div>
          <div className="days-strip">
            {days.map((d, idx) => {
              const isWeekend = (idx % 7 === 5) || (idx % 7 === 6)
              const items = (dayAssignments[d.id] ?? []).map(({ instId, typeId }) => ({
                instId,
                ...typeById[typeId],
              }))
              return (
                <DayColumn
                  key={d.id}
                  day={d.label}
                  items={items}
                  dragOver={dragOverDay === d.id}
                  onDragOver={(e) => onDragOverDay(e, d.id)}
                  onDragLeave={() => onDragLeaveDay(d.id)}
                  onDrop={(e) => onDropToDay(e, d.id)}
                  onRemove={onRemoveFromDay}
                  onDragStartInstance={(e, instId) => onDragStartInstance(e, d.id, instId)}
                  isWeekend={isWeekend}
                />
              )
            })}
          </div>

          {/* New: create workload controls at bottom */}
          <div className="btns create-row">
            <input
              type="text"
              className="name-input"
              placeholder="Workload name"
              value={newWorkloadName}
              onChange={(e) => setNewWorkloadName(e.target.value)}
            />
            <button onClick={handleCreateComposite} disabled={!newWorkloadName.trim()}>
              Create
            </button>
          </div>
        </section>
      </div>

      {/* SLO adherence full-width pane */}
      <section className="card slo-card">
        <h2>Minimum Cluster Size to Meet SLO</h2>
        <div className="slo-grid">
          <div className="form-col">
            <label className="form-label">Workload</label>
            <select
              className="input-field"
              value={sloWorkloadName}
              onChange={(e) => setSloWorkloadName(e.target.value)}
              aria-label="Select workload for SLO"
            >
              {workloads.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
              {workloads.length === 0 && <option value="" disabled>(none found)</option>}
            </select>

            <label className="form-label">SLO (seconds)</label>
            <div className="slo-slider-row">
              <input
                type="range"
                min="0.1"
                max="600"
                step="0.1"
                value={Number.isFinite(sloSecondsNum) && sloSecondsNum > 0 ? sloSecondsNum : 0.1}
                onChange={(e) => setSloSeconds(e.target.value)}
                aria-label="SLO seconds slider"
              />
              <input
                type="number"
                className={`input-field tight ${validSloSeconds ? '' : 'invalid'}`}
                min="0.1"
                step="0.1"
                value={sloSeconds}
                onChange={(e) => setSloSeconds(e.target.value)}
                aria-label="SLO seconds value"
                style={{ width: 90 }}
              />
            </div>
            {!validSloSeconds && sloSeconds !== '' && (
              <small className="error-text">Enter a positive number.</small>
            )}

            <label className="form-label">Percentile</label>
            <div className="segmented" role="group" aria-label="Percentile">
              {[90, 95, 99].map(p => (
                <button
                  key={p}
                  type="button"
                  className={`seg-btn ${Number(sloPercentile) === p ? 'selected' : ''}`}
                  aria-pressed={Number(sloPercentile) === p}
                  onClick={() => setSloPercentile(String(p))}
                >
                  {p}
                </button>
              ))}
            </div>
            {!validSloPercentile && (
              <small className="error-text">Pick 90, 95 or 99.</small>
            )}

            {/* Removed Calculate button; auto-calc is active */}
            {sloError && <small className="error-text">{sloError}</small>}
          </div>

          {/* Right side: simple SVG line plot */}
          <div>
            {sloSeries?.length > 0
              ? <SLOLineChart data={sloSeries} />
              : <div style={{ opacity: 0.6, fontSize: 12 }}>{sloLoading ? 'Calculating…' : 'No results yet.'}</div>}
          </div>
        </div>
      </section>
    </div>
  )
}

/** Palette: renders a grid. Each cell is draggable and spawns a new instance. */
function PaletteGrid({ catalog, hLevels, tLevels, onDragStartFromPalette }) {
  // Fast lookup: typeId -> catalog entry
  const byTypeId = useMemo(
    () => Object.fromEntries(catalog.map(c => [c.typeId, c])),
    [catalog]
  )

  // Inline columns count to match dynamic number of H levels
  const cols = hLevels?.length ?? 0

  return (
    <div
      className="palette-grid"
      style={{ gridTemplateColumns: `auto repeat(${cols}, 44px)` }}
    >
      {/* top-left corner cell (empty) */}
      <div className="palette-corner palette-head" aria-hidden />

      {/* column headers: H values */}
      {hLevels.map(h => (
        <div key={`hlabel-${h}`} className="palette-head palette-collabel">
          <span className="label-title">H</span>
          <span className="label-value">{h}</span>
        </div>
      ))}

      {/* rows: T label + cells */}
      {tLevels.map(t => (
        <Fragment key={`row-${t}`}>
          <div className="palette-head palette-rowlabel">
            <span className="label-title">T</span>
            <span className="label-value">{t}</span>
          </div>
          {hLevels.map(h => {
            const typeId = chunkTypeId(h, t)
            const cell = byTypeId[typeId]
            return (
              <div
                key={typeId}
                className="palette-cell"
                draggable={!!cell}
                onDragStart={(e) => cell && onDragStartFromPalette(e, typeId)}
                title={cell ? `H=${h} shape=${cell.shapeChar}, T=${t}` : ''}
              >
                {cell && <ChunkIcon color={cell.colorHex} shapeChar={cell.shapeChar} />}
              </div>
            )
          })}
        </Fragment>
      ))}
    </div>
  )
}

/** Day column with icons only; items can be dragged to other days or removed. */
function DayColumn({
  day, items, dragOver, onDragOver, onDragLeave, onDrop, onRemove, onDragStartInstance, isWeekend
}) {
  return (
    <div
      className={`day ${dragOver ? 'drag-over' : ''} ${isWeekend ? 'weekend' : ''}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <h4>{day}</h4>
      <div className="slot icons-flow">
        {items.map(item => (
          <div
            key={item.instId}
            className="day-icon-wrap"
            draggable
            onDragStart={(e) => onDragStartInstance(e, item.instId)}
          >
            <ChunkIcon
              color={item.colorHex}
              shapeChar={item.shapeChar}
              title={`H=${item.h} shape=${item.shapeChar}, T=${item.t}`}
            />
          </div>
        ))}
        {items.length === 0 && (
          <div style={{ opacity: 0.6, fontSize: 12 }}>Drop icons here</div>
        )}
      </div>
    </div>
  )
}

/** Icon-only chunk renderer (SVG). Adds support for '*' star shape. */
function ChunkIcon({ color, shapeChar, ...svgProps }) {
  const size = 36
  const pad = 4
  const s2 = size / 2
  const sMin = size - pad * 2

  const stroke = '#1e293b'
  const strokeWidth = 2

  const HEX_SCALE = 1.08
  const maxR = s2 - (pad + 2) - strokeWidth / 2
  const baseR = (sMin / 2) - strokeWidth - 1
  const rHex = Math.min(baseR * HEX_SCALE, maxR)

  // helper: regular hexagon points
  const hexPoints = (cx, cy, r, rotationRad = Math.PI / 6) => {
    const pts = []
    for (let i = 0; i < 6; i++) {
      const a = rotationRad + (Math.PI / 3) * i
      pts.push(`${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`)
    }
    return pts.join(' ')
  }

  // helper: 5-point star
  const starPoints = (cx, cy, rOuter, rInner, rotationRad = -Math.PI / 2) => {
    const pts = []
    for (let i = 0; i < 10; i++) {
      const r = i % 2 === 0 ? rOuter : rInner
      const a = rotationRad + (Math.PI / 5) * i
      pts.push(`${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`)
    }
    return pts.join(' ')
  }

  const rOuter = Math.min(baseR, maxR)
  const rInner = rOuter * 0.5

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-hidden
      style={{ pointerEvents: 'none' }}
      {...svgProps}
    >
      {shapeChar === 'o' && (
        <circle cx={s2} cy={s2} r={(sMin / 2) - strokeWidth} fill={color} stroke={stroke} strokeWidth={strokeWidth} />
      )}
      {shapeChar === 's' && (
        <rect x={pad + 2} y={pad + 2} width={sMin - 4} height={sMin - 4} fill={color} stroke={stroke} strokeWidth={strokeWidth} rx="4" />
      )}
      {shapeChar === '^' && (
        <polygon
          points={`${s2},${pad + 2} ${size - pad - 2},${size - pad - 2} ${pad + 2},${size - pad - 2}`}
          fill={color}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
        />
      )}
      {shapeChar === 'H' && (
        <polygon
          points={hexPoints(s2, s2, rHex)}
          fill={color}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
        />
      )}
      {shapeChar === '*' && (
        <polygon
          points={starPoints(s2, s2, rOuter, rInner)}
          fill={color}
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeLinejoin="round"
        />
      )}
    </svg>
  )
}

// NEW: tiny inline SVG line chart
function SLOLineChart({ data }) {
  const w = 640, h = 220, pad = 28

  // x scale
  const xs = (i) => pad + (data.length > 1 ? (i * (w - 2 * pad)) / (data.length - 1) : 0)

  // keep only positive values for log scale
  const pos = data.filter(v => v != null && isFinite(v) && v > 0)

  // fixed log2 domain [4, 32]
  const axisMin = 4
  const axisMax = 32
  const log2 = (v) => Math.log2(v)
  const tOf = (v) => (log2(v) - log2(axisMin)) / Math.max(log2(axisMax) - log2(axisMin), 1e-9)
  const ys = (v) => {
    const vv = Math.min(Math.max(v, axisMin), axisMax) // clamp
    return pad + (h - 2 * pad) * (1 - tOf(vv))
  }

  // ticks at 4, 8, 16, 32
  const ticks = [4, 8, 16, 32]
  const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
  const fmtTick = (v) => (v >= 1 ? nf.format(v) : String(v))

  // path (skip nonpositive points)
  let d = ''
  let penDown = false
  data.forEach((v, i) => {
    if (!(v != null && isFinite(v) && v > 0)) { penDown = false; return }
    const x = xs(i), y = ys(v)
    d += penDown ? ` L ${x} ${y}` : ` M ${x} ${y}`
    penDown = true
  })

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-label="SLO line chart">
      {/* axes */}
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />

      {/* y grid + tick labels */}
      {ticks.map((tv) => {
        const y = ys(tv)
        return (
          <g key={`ty-${tv}`}>
            <line x1={pad} y1={y} x2={w - pad} y2={y} stroke="#e2e8f0" strokeWidth="1" />
            <text x={pad - 6} y={y + 3} fontSize="10" fill="#475569" textAnchor="end">{fmtTick(tv)}</text>
          </g>
        )
      })}

      {/* line path */}
      <path d={d} fill="none" stroke="#2563eb" strokeWidth="2" />

      {/* points */}
      {data.map((v, i) => (v != null && isFinite(v) && v > 0) ? (
        <circle key={i} cx={xs(i)} cy={ys(v)} r="3" fill="#2563eb" stroke="#1e293b" strokeWidth="1" />
      ) : null)}

      {/* x axis labels: day indices */}
      {data.map((_, i) => (
        <text key={`t${i}`} x={xs(i)} y={h - pad + 14} fontSize="10" fill="#475569" textAnchor="middle">{i + 1}</text>
      ))}

      {/* axis titles */}
      <text x={w - pad} y={h - 6} fontSize="11" fill="#475569" textAnchor="end">day</text>
      <text x={pad} y={12} fontSize="11" fill="#475569">RPU (log2)</text>
    </svg>
  )
}
