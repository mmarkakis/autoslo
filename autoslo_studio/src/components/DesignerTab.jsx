import { Fragment, useEffect, useMemo, useState } from 'react'

const WEEKDAYS_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const dayLabel = (n) => `${WEEKDAYS_SHORT[(n - 1) % 7]} (${n})`
const makeDay = (n) => ({ id: `d${n}`, label: dayLabel(n) })
const chunkTypeId = (h, t) => `h${h}_t${t}`
const makeInstId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `i_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

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

const loadCompositeWorkload = async (name) => {
  if (!name) return null
  const res = await fetch(`/api/composite/${encodeURIComponent(name)}`)
  if (!res.ok) throw new Error('Workload not found on server')
  const text = await res.text()
  return text
}

export default function DesignerTab({ workloads, loading, onRefreshWorkloads }) {
  const [days, setDays] = useState(() => Array.from({ length: 7 }, (_, i) => makeDay(i + 1)))
  const [dayAssignments, setDayAssignments] = useState(
    () => Object.fromEntries(Array.from({ length: 7 }, (_, i) => [`d${i + 1}`, []]))
  )
  const [dragOverDay, setDragOverDay] = useState(null)
  const [dragOverTrash, setDragOverTrash] = useState(false)
  const [hShapeMap, setHShapeMap] = useState(null)
  const [tColorMap, setTColorMap] = useState(null)
  const [selectedWorkload, setSelectedWorkload] = useState('')
  const [newWorkloadName, setNewWorkloadName] = useState('')

  useEffect(() => {
    let cancelled = false
    fetch('/api/chunk/graphics')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then(meta => {
        if (cancelled) return
        setHShapeMap(meta?.H_SHAPE_MAP ?? null)
        setTColorMap(meta?.T_COLOR_MAP ?? null)
      })
      .catch(err => {
        console.error('Failed to load chunk meta:', err)
      })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    setSelectedWorkload(prev => (prev && workloads.includes(prev)) ? prev : (workloads[0] ?? ''))
  }, [workloads])

  const H_LEVELS = useMemo(() => {
    if (!hShapeMap) return []
    return Object.keys(hShapeMap).map(Number).sort((a, b) => a - b)
  }, [hShapeMap])

  const T_LEVELS = useMemo(() => {
    if (!tColorMap) return []
    return Object.keys(tColorMap).map(Number).sort((a, b) => b - a)
  }, [tColorMap])

  const CATALOG = useMemo(() => {
    if (!hShapeMap || !tColorMap) return []
    return buildCatalog(hShapeMap, tColorMap, H_LEVELS, T_LEVELS)
  }, [hShapeMap, tColorMap, H_LEVELS, T_LEVELS])

  const typeById = useMemo(
    () => Object.fromEntries(CATALOG.map(c => [c.typeId, c])),
    [CATALOG]
  )

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

  const onDragStartFromPalette = (e, typeId) => {
    setDragPayload(e, { from: 'palette', typeId })
  }

  const onDragStartInstance = (e, dayId, instId) => {
    setDragPayload(e, { from: 'day', day: dayId, instId })
  }

  const onDragOverDay = (e, day) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDragOverDay(day)
  }

  const onDragLeaveDay = (day) => {
    setDragOverDay(prev => (prev === day ? null : prev))
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

  const onDragOverTrash = (e) => {
    e.preventDefault()
    setDragOverTrash(true)
  }
  const onDragLeaveTrash = () => setDragOverTrash(false)
  const onDropToTrash = (e) => {
    const payload = getDragPayload(e)
    setDragOverTrash(false)
    if (!payload || payload.from !== 'day') return
    const { day: fromDay, instId } = payload
    if (!fromDay || !instId) return
    setDayAssignments(prev => {
      const next = structuredClone(prev)
      next[fromDay] = (next[fromDay] ?? []).filter(x => x.instId !== instId)
      return next
    })
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

  const handleLoadLibraryWorkload = async () => {
    try {
      if (!selectedWorkload) return
      const body = await loadCompositeWorkload(selectedWorkload)
      let def = body
      if (typeof body === 'string') {
        try { def = JSON.parse(body) } catch { /* ignore */ }
      }
      if (!def || typeof def !== 'object') {
        alert('Unsupported workload format from server.')
        return
      }

      const toNum = (v) => (typeof v === 'number'
        ? v
        : (v != null && !Number.isNaN(Number(v)) ? Number(v) : NaN))

      const mondayIndex = Number.isInteger(def?.monday_index)
        ? def.monday_index % 7
        : (typeof def?.monday_index === 'number' ? Math.trunc(def.monday_index) % 7 : 0)

      let perDay = []
      let orderCount = 0

      if (Array.isArray(def.days)) {
        perDay = def.days.map(d => {
          const arr = Array.isArray(d?.chunks) ? d.chunks : (Array.isArray(d) ? d : [])
          return Array.isArray(arr) ? arr : []
        })
        orderCount = perDay.length
      } else {
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

  const buildCompositeDefinitionForCreate = () => {
    const name = newWorkloadName.trim()
    if (!name) return null

    const dayChunks = days.map(d =>
      (dayAssignments[d.id] ?? []).map(({ typeId }) => {
        const { h, t } = typeById[typeId]
        return { H: h, T: t }
      })
    )

    let leadEmpty = dayChunks.findIndex(arr => Array.isArray(arr) && arr.length > 0)
    if (leadEmpty === -1) leadEmpty = 0

    let end = dayChunks.length
    while (end > leadEmpty && (dayChunks[end - 1]?.length ?? 0) === 0) end--

    const payloadDays = dayChunks.slice(leadEmpty, end).map(chs => ({ chunks: chs }))

    return {
      name,
      monday_index: leadEmpty % 7,
      days: payloadDays,
    }
  }

  const handleCreateComposite = async () => {
    try {
      const def = buildCompositeDefinitionForCreate()
      if (!def) {
        alert('Please enter a workload name.')
        return
      }

      let res = await fetch('/api/composite/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(def),
      })

      if (!res.ok) {
        const qs = new URLSearchParams({ workload_definition: JSON.stringify(def) }).toString()
        res = await fetch(`/api/composite/create?${qs}`, { method: 'GET' })
      }
      if (!res.ok) {
        const msg = await res.text().catch(() => '')
        throw new Error(`Create failed: HTTP ${res.status} ${msg}`)
      }

      await onRefreshWorkloads?.()
      setSelectedWorkload(def.name)
      alert('Workload created.')
    } catch (err) {
      console.error('Create workload error:', err)
      alert('Failed to create workload. See console for details.')
    }
  }

  return (
    <div className="row">
      <div className="left-col">
        

        <section className="card library-card">
          <h2>Load Workload</h2>
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
          <div className="btns"> 
            <button
              className="secondary"
              onClick={() => onRefreshWorkloads?.()}
              disabled={loading}
            >
              🔄 Refresh List
            </button>
            <button onClick={handleLoadLibraryWorkload} disabled={!selectedWorkload}>Load</button>
          </div>
          <div className="library-note">
            {workloads.length ? `Loaded ${workloads.length}` : 'No workloads found.'}
          </div>
        </section>

        <section className="card">
          <h2>Chunk Palette</h2>
          <PaletteGrid
            catalog={CATALOG}
            hLevels={H_LEVELS}
            tLevels={T_LEVELS}
            onDragStartFromPalette={onDragStartFromPalette}
          />
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

        <section className="card">
          <h2>Save Workload</h2>
          <input
            type="text"
            className="input-field"
            placeholder="Workload name"
            value={newWorkloadName}
            onChange={(e) => setNewWorkloadName(e.target.value)}
          />
          <div className="btns" style={{ marginTop: 10 }}>
            <button onClick={handleCreateComposite} disabled={!newWorkloadName.trim()}>
              Save
            </button>
          </div>
        </section>
      </div>

      <section className="card workload-card">
        <div className="workload-header">
          <h2>Workload</h2>
          <div className="days-toolbar">
            <div className="toolbar-group">
              <button className="btn-icon minus" onClick={handleRemoveDay} aria-label="Remove day">−</button>
              <span className="toolbar-label">day</span>
              <button className="btn-icon plus" onClick={handleAddDay} aria-label="Add day">+</button>
            </div>
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
                onDragStartInstance={(e, instId) => onDragStartInstance(e, d.id, instId)}
                isWeekend={isWeekend}
              />
            )
          })}
        </div>
      </section>
    </div>
  )
}

function PaletteGrid({ catalog, hLevels, tLevels, onDragStartFromPalette }) {
  const byTypeId = useMemo(
    () => Object.fromEntries(catalog.map(c => [c.typeId, c])),
    [catalog]
  )
  const cols = hLevels?.length ?? 0

  return (
    <div
      className="palette-grid"
      style={{ gridTemplateColumns: `auto repeat(${cols}, 44px)` }}
    >
      <div className="palette-corner palette-head" aria-hidden />
      {hLevels.map(h => (
        <div key={`hlabel-${h}`} className="palette-head palette-collabel">
          <span className="label-title">H</span>
          <span className="label-value">{h}</span>
        </div>
      ))}
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

function DayColumn({ day, items, dragOver, onDragOver, onDragLeave, onDrop, onDragStartInstance, isWeekend }) {
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

  const hexPoints = (cx, cy, r, rotationRad = Math.PI / 6) => {
    const pts = []
    for (let i = 0; i < 6; i++) {
      const a = rotationRad + (Math.PI / 3) * i
      pts.push(`${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`)
    }
    return pts.join(' ')
  }

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
