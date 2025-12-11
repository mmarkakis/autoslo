import { useEffect, useState } from 'react'

const normalizeBlueprintRouterMap = (raw) => {
    const result = {}
    if (!raw || typeof raw !== 'object') return result

    const coerceRouterList = (val) => {
        if (!val && val !== 0) return []
        if (Array.isArray(val)) return val
        if (typeof val === 'string') return [val]
        if (val && typeof val === 'object') {
            if (Array.isArray(val.routers)) return val.routers
            if (Array.isArray(val.query_routers)) return val.query_routers
            if (Array.isArray(val.queryRouters)) return val.queryRouters
            return Object.values(val).flatMap(coerceRouterList)
        }
        return []
    }

    const assign = (bp, sourceVal) => {
        if (!bp) return
        const routers = Array.from(
            new Set(coerceRouterList(sourceVal).map((r) => String(r).trim()).filter(Boolean))
        )
        result[bp] = routers
    }

    if (Array.isArray(raw.blueprints)) {
        const sharedRouters = coerceRouterList(raw.routers ?? raw.query_routers)
        raw.blueprints.forEach((bp) => assign(bp, sharedRouters))
        return result
    }

    const blueprintSource = (
        raw.blueprints &&
        typeof raw.blueprints === 'object' &&
        !Array.isArray(raw.blueprints)
    )
        ? raw.blueprints
        : raw

    Object.entries(blueprintSource).forEach(([bp, val]) => {
        if (bp === 'routers' || bp === 'blueprints') return
        assign(bp, val)
    })

    return result
}

const normalizeTailSeries = (raw, percentilesRequested) => {
    if (Array.isArray(raw)) {
        return raw.map((day) => {
            const entry = {}
            percentilesRequested.forEach((p) => {
                const val =
                    day?.[p] ??
                    day?.[String(p)] ??
                    day?.[`p${p}`]
                if (Number.isFinite(val)) entry[p] = val
            })
            return entry
        })
    }
    if (raw && typeof raw === 'object') {
        const keys = Object.keys(raw)
        const length = Math.max(
            0,
            ...keys.map((k) =>
                Array.isArray(raw[k]) ? raw[k].length : 0
            )
        )
        return Array.from({ length }, (_, idx) => {
            const entry = {}
            keys.forEach((k) => {
                const percentile = Number.parseInt(k, 10)
                const arr = raw[k]
                const val = Array.isArray(arr) ? arr[idx] : undefined
                if (Number.isFinite(val)) entry[percentile] = val
            })
            return entry
        })
    }
    return []
}
const PERCENTILE_COLORS = { p90: '#10b981', p95: '#2563eb', p99: '#ef4444' }
const WEEKEND_FILL = 'lightgray'

export default function PerformanceTab({ workloads, loading, onRefreshWorkloads }) {
    const [pendingWorkload, setPendingWorkload] = useState('')
    const [selectedWorkload, setSelectedWorkload] = useState('')
    const [sloSeconds, setSloSeconds] = useState('60')
    const [sloPercentile, setSloPercentile] = useState('95')
    const [sloLoading, setSloLoading] = useState(false)
    const [sloSeries, setSloSeries] = useState([])
    const [sloError, setSloError] = useState('')

    const [tailWorkloadName, setTailWorkloadName] = useState('')
    const [tailBlueprints, setTailBlueprints] = useState([])
    const [tailBlueprintName, setTailBlueprintName] = useState('')
    const [tailRouters, setTailRouters] = useState([])
    const [tailRouterName, setTailRouterName] = useState('')
    const [tailPercentiles, setTailPercentiles] = useState({ p90: true, p95: false, p99: false })
    const [tailLoading, setTailLoading] = useState(false)
    const [tailData, setTailData] = useState([])
    const [tailError, setTailError] = useState('')
    const [blueprintRouterMap, setBlueprintRouterMap] = useState({})
    const [definitionImgError, setDefinitionImgError] = useState(false)
    const [definitionImgVersion, setDefinitionImgVersion] = useState(0)

    useEffect(() => {
        setPendingWorkload(prev => (prev && workloads.includes(prev)) ? prev : (workloads[0] ?? ''))
        setSelectedWorkload(prev => (prev && workloads.includes(prev)) ? prev : '')
    }, [workloads])

    useEffect(() => {
        if (!selectedWorkload) {
            setBlueprintRouterMap({})
            setTailBlueprints([])
            setTailBlueprintName('')
            setTailRouters([])
            setTailRouterName('')
            return
        }
        let cancelled = false
        fetch(`/api/composite/${encodeURIComponent(selectedWorkload)}/blueprints_and_routers`)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`)
                return res.json()
            })
            .then(data => {
                if (cancelled) return
                const map = normalizeBlueprintRouterMap(data)
                const blueprints = Object.keys(map).sort()
                const firstBlueprint = blueprints[0] ?? ''
                const routers = firstBlueprint ? [...(map[firstBlueprint] ?? [])].sort() : []
                setBlueprintRouterMap(map)
                setTailBlueprints(blueprints)
                setTailBlueprintName(prev => (prev && blueprints.includes(prev)) ? prev : firstBlueprint)
                setTailRouters(routers)
                setTailRouterName(prev => (prev && routers.includes(prev)) ? prev : (routers[0] ?? ''))
            })
            .catch(err => {
                if (cancelled) return
                console.error('Failed to load blueprints and routers:', err)
                setBlueprintRouterMap({})
                setTailBlueprints([])
                setTailBlueprintName('')
                setTailRouters([])
                setTailRouterName('')
            })
        return () => { cancelled = true }
    }, [selectedWorkload])

    useEffect(() => {
        const routers = Array.isArray(blueprintRouterMap[tailBlueprintName])
            ? [...blueprintRouterMap[tailBlueprintName]].sort()
            : []
        setTailRouters(routers)
        setTailRouterName(prev => (prev && routers.includes(prev)) ? prev : (routers[0] ?? ''))
    }, [tailBlueprintName, blueprintRouterMap])

    const sloSecondsNum = Number(sloSeconds)
    const sloPercentileNum = Number(sloPercentile)
    const validSloSeconds = sloSeconds.trim() !== '' && Number.isFinite(sloSecondsNum) && sloSecondsNum > 0
    const validSloPercentile = [90, 95, 99].includes(sloPercentileNum)

    useEffect(() => {
        if (sloLoading) return
        if (!selectedWorkload || !validSloSeconds || !validSloPercentile) return

        let cancelled = false
        const timer = setTimeout(async () => {
            try {
                setSloLoading(true)
                setSloError('')
                const qs = new URLSearchParams({
                    workload_name: selectedWorkload,
                    tail_slo_s: String(sloSecondsNum),
                    percentile: String(sloPercentileNum),
                }).toString()
                const res = await fetch(`/api/strat/cheapest_adherent_cluster?${qs}`, { method: 'POST' })
                if (!res.ok) {
                    const msg = await res.text().catch(() => '')
                    throw new Error(`HTTP ${res.status} ${msg}`)
                }
                const data = await res.json()
                if (!cancelled) setSloSeries(Array.isArray(data) ? data : [])
            } catch (err) {
                if (!cancelled) {
                    console.error('SLO calculate error:', err)
                    setSloError('Failed to calculate. See console.')
                    setSloSeries([])
                }
            } finally {
                if (!cancelled) setSloLoading(false)
            }
        }, 200)

        return () => {
            cancelled = true
            clearTimeout(timer)
        }
    }, [selectedWorkload, sloSecondsNum, sloPercentileNum, validSloSeconds, validSloPercentile])

    useEffect(() => {
        if (!selectedWorkload || !tailBlueprintName || !tailRouterName) {
            setTailData([])
            return
        }
        const activePercentiles = Object.entries(tailPercentiles)
            .filter(([_, active]) => active)
            .map(([label]) => Number.parseInt(label.slice(1), 10))
        if (activePercentiles.length === 0) {
            setTailData([])
            return
        }

        let cancelled = false
        const timer = setTimeout(async () => {
            try {
                setTailLoading(true)
                setTailError('')
                const res = await fetch('/api/composite/tail_perf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        workload_name: selectedWorkload,
                        blueprint_name: tailBlueprintName,
                        query_router_name: tailRouterName,
                        percentiles: activePercentiles,
                    }),
                })
                if (!res.ok) {
                    const msg = await res.text().catch(() => '')
                    throw new Error(`HTTP ${res.status} ${msg}`)
                }
                const payload = await res.json()
                if (!cancelled) setTailData(normalizeTailSeries(payload, activePercentiles))
            } catch (err) {
                if (!cancelled) {
                    console.error('Tail latency calculate error:', err)
                    setTailError('Failed to calculate. See console.')
                    setTailData([])
                }
            } finally {
                if (!cancelled) setTailLoading(false)
            }
        }, 200)

        return () => {
            cancelled = true
            clearTimeout(timer)
        }
    }, [selectedWorkload, tailBlueprintName, tailRouterName, tailPercentiles])

    useEffect(() => {
        setDefinitionImgError(false)
        setDefinitionImgVersion(prev => prev + 1)
    }, [selectedWorkload])

    const handleLoadWorkload = () => {
        if (!pendingWorkload) return
        setSelectedWorkload(pendingWorkload)
    }

    return (
        <>
            <div className="row perf-top-row">
                <div className="left-col">
                    <section className="card library-card">
                        <h2>Load Workload</h2>
                        <select
                            value={pendingWorkload}
                            onChange={(e) => setPendingWorkload(e.target.value)}
                            aria-label="Select workload for performance panels"
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
                            <button onClick={handleLoadWorkload} disabled={!pendingWorkload}>
                                Load
                            </button>
                        </div>
                        <div className="library-note">
                            {workloads.length ? `Loaded ${workloads.length}` : 'No workloads found.'}
                        </div>
                    </section>
                </div>
                <section className="card perf-image-card">
                    {selectedWorkload ? (
                        definitionImgError ? (
                            <div style={{ fontSize: 12, opacity: 0.6 }}>
                                Definition image not found.
                            </div>
                        ) : (
                            <img
                                key={`${selectedWorkload}-${definitionImgVersion}`}
                                src={`/api/composite/${encodeURIComponent(selectedWorkload)}/definition_image?ts=${definitionImgVersion}`}
                                alt={`${selectedWorkload} definition`}
                                onError={() => setDefinitionImgError(true)}
                            />
                        )
                    ) : (
                        <div style={{ fontSize: 12, opacity: 0.6 }}>
                            Select a workload to view its definition.
                        </div>
                    )}
                </section>
            </div>

            <section className="card slo-card">
                <h2>Minimum Cluster Size to Meet SLO</h2>
                <div className="slo-grid">
                    <div className="form-col">
                        

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

                        {sloError && <small className="error-text">{sloError}</small>}
                    </div>

                    <div>
                        {sloSeries?.length > 0
                            ? <SLOLineChart data={sloSeries} />
                            : <div style={{ opacity: 0.6, fontSize: 12 }}>{sloLoading ? 'Calculating…' : 'No results yet.'}</div>}
                    </div>
                </div>
            </section>

            <section className="card slo-card">
                <h2>Tail Latency on Given Blueprint</h2>
                <div className="slo-grid">
                    <div className="form-col">
                      

                        <label className="form-label">Blueprint</label>
                        <select
                            className="input-field"
                            value={tailBlueprintName}
                            onChange={(e) => setTailBlueprintName(e.target.value)}
                            aria-label="Select blueprint for tail latency"
                            disabled={tailBlueprints.length === 0}
                        >
                            {tailBlueprints.map(name => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                            {tailBlueprints.length === 0 && <option value="" disabled>(no blueprints)</option>}
                        </select>

                        <label className="form-label">Query Router</label>
                        <select
                            className="input-field"
                            value={tailRouterName}
                            onChange={(e) => setTailRouterName(e.target.value)}
                            aria-label="Select query router for tail latency"
                            disabled={tailRouters.length === 0}
                        >
                            {tailRouters.map(name => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                            {tailRouters.length === 0 && <option value="" disabled>(no routers)</option>}
                        </select>

                        <label className="form-label">Percentiles</label>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                            {['p90', 'p95', 'p99'].map(p => (
                                <label
                                    key={p}
                                    style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
                                >
                                    <input
                                        type="checkbox"
                                        checked={tailPercentiles[p]}
                                        onChange={(e) => setTailPercentiles(prev => ({ ...prev, [p]: e.target.checked }))}
                                    />
                                    <span
                                        aria-hidden
                                        style={{
                                            display: 'inline-block',
                                            width: 18,
                                            borderTop: `2px solid ${PERCENTILE_COLORS[p]}`,
                                            marginBottom: -1,
                                        }}
                                    />
                                    <span style={{ fontSize: 13, fontWeight: 600 }}>{p}</span>
                                </label>
                            ))}
                        </div>

                        {tailError && <small className="error-text">{tailError}</small>}
                    </div>

                    <div>
                        {tailData.length > 0
                            ? <TailLatencyChart data={tailData} active={tailPercentiles} />
                            : <div style={{ opacity: 0.6, fontSize: 12 }}>{tailLoading ? 'Calculating…' : 'No results yet.'}</div>}
                    </div>
                </div>
            </section>
        </>
    )
}

function SLOLineChart({ data }) {
    const w = 640, h = 220, pad = 28
    const xs = (i) => pad + (data.length > 1 ? (i * (w - 2 * pad)) / (data.length - 1) : 0)
    const axisMin = 4
    const axisMax = 32
    const log2 = (v) => Math.log2(v)
    const tOf = (v) => (log2(v) - log2(axisMin)) / Math.max(log2(axisMax) - log2(axisMin), 1e-9)
    const ys = (v) => {
        const vv = Math.min(Math.max(v, axisMin), axisMax)
        return pad + (h - 2 * pad) * (1 - tOf(vv))
    }
    const ticks = [4, 8, 16, 32]
    const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })
    const fmtTick = (v) => (v >= 1 ? nf.format(v) : String(v))

    let d = ''
    let penDown = false
    data.forEach((v, i) => {
        if (!(v != null && isFinite(v) && v > 0)) { penDown = false; return }
        const x = xs(i), y = ys(v)
        d += penDown ? ` L ${x} ${y}` : ` M ${x} ${y}`
        penDown = true
    })

    const step = data.length > 1 ? (w - 2 * pad) / (data.length - 1) : (w - 2 * pad)
    const weekendRects = data.map((_, i) => {
        if (i % 7 !== 5 && i % 7 !== 6) return null
        const center = pad + step * i
        const half = step / 2
        const x0 = Math.max(pad, center - half)
        const x1 = Math.min(w - pad, center + half)
        return (
            <rect
                key={`wk-${i}`}
                x={x0}
                y={pad}
                width={Math.max(0, x1 - x0)}
                height={h - 2 * pad}
                fill={WEEKEND_FILL}
                fillOpacity="0.5"
            />
        )
    })
    const dayInitials = ['M', 'T', 'W', 'T', 'F', 'S', 'S']

    return (
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-label="SLO line chart">
            {weekendRects}
            <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />
            <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />
            {ticks.map((tv) => {
                const y = ys(tv)
                return (
                    <g key={`ty-${tv}`}>
                        <line x1={pad} y1={y} x2={w - pad} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                        <text x={pad - 6} y={y + 3} fontSize="10" fill="#475569" textAnchor="end">{fmtTick(tv)}</text>
                    </g>
                )
            })}
            <path d={d} fill="none" stroke="#2563eb" strokeWidth="2" />
            {data.map((v, i) => (v != null && isFinite(v) && v > 0) ? (
                <circle key={i} cx={xs(i)} cy={ys(v)} r="3" fill="#2563eb" stroke="#1e293b" strokeWidth="1" />
            ) : null)}
            {data.map((_, i) => (
                <text
                    key={`t${i}`}
                    x={xs(i)}
                    y={h - pad + 14}
                    fontSize="10"
                    fill="#475569"
                    textAnchor="middle"
                >
                    {dayInitials[i % 7]}
                </text>
            ))}
            <text x={pad} y={12} fontSize="11" fill="#475569">RPU (log2)</text>
        </svg>
    )
}

function TailLatencyChart({ data, active }) {
    const selectedPercentiles = Object.entries(active)
        .filter(([_, on]) => on)
        .map(([label]) => Number.parseInt(label.slice(1), 10))
        .sort((a, b) => a - b)

    const clampValue = (v) => Math.min(Math.max(v, 1), 10_000)
    const allValues = []
    data.forEach((day) => {
        selectedPercentiles.forEach((p) => {
            const v = day?.[p]
            if (Number.isFinite(v) && v > 0) allValues.push(clampValue(v))
        })
    })
    if (allValues.length === 0) {
        return <div style={{ opacity: 0.6, fontSize: 12 }}>No data to display.</div>
    }

    const w = 640, h = 220, pad = 28
    const minExp = 0
    const maxExp = 4
    const ys = (v) => {
        const vClamped = clampValue(v)
        const logVal = Math.log10(vClamped)
        return pad + (h - 2 * pad) * (1 - (logVal - minExp) / (maxExp - minExp))
    }
    const ticks = Array.from({ length: maxExp - minExp + 1 }, (_, i) => 10 ** (minExp + i))
    const maxLen = data.length
    const xs = (i) => pad + (maxLen > 1 ? (i * (w - 2 * pad)) / (maxLen - 1) : 0)

    const step = maxLen > 1 ? (w - 2 * pad) / (maxLen - 1) : (w - 2 * pad)
    const weekendRects = data.map((_, i) => {
        if (i % 7 !== 5 && i % 7 !== 6) return null
        const center = pad + step * i
        const half = step / 2
        const x0 = Math.max(pad, center - half)
        const x1 = Math.min(w - pad, center + half)
        return (
            <rect
                key={`wk-${i}`}
                x={x0}
                y={pad}
                width={Math.max(0, x1 - x0)}
                height={h - 2 * pad}
                fill={WEEKEND_FILL}
                fillOpacity="0.5"
            />
        )
    })
    const dayInitials = ['M', 'T', 'W', 'T', 'F', 'S', 'S']

    return (
        <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-label="Tail latency chart">
            {weekendRects}
            <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />
            <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#94a3b8" strokeWidth="1" />
            {ticks.map((tv) => {
                const y = ys(tv)
                const exp = Math.log10(tv)
                return (
                    <g key={`ty-${tv}`}>
                        <line x1={pad} y1={y} x2={w - pad} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                        <text x={pad - 6} y={y + 4} fontSize="10" fill="#475569" textAnchor="end">
                            10<tspan baselineShift="super" fontSize="8">{exp}</tspan>
                        </text>
                    </g>
                )
            })}
            {selectedPercentiles.map((p) => {
                let d = ''
                let penDown = false
                data.forEach((day, i) => {
                    const value = day?.[p]
                    if (!(Number.isFinite(value) && value > 0)) { penDown = false; return }
                    const x = xs(i), y = ys(value)
                    d += penDown ? ` L ${x} ${y}` : ` M ${x} ${y}`
                    penDown = true
                })
                const color = PERCENTILE_COLORS[`p${p}`] ?? '#2563eb'
                return (
                    <g key={p}>
                        <path d={d} fill="none" stroke={color} strokeWidth="2" />
                        {data.map((day, i) => {
                            const value = day?.[p]
                            if (!(Number.isFinite(value) && value > 0)) return null
                            return (
                                <circle
                                    key={`${p}-${i}`}
                                    cx={xs(i)}
                                    cy={ys(value)}
                                    r="3"
                                    fill={color}
                                    stroke="#1e293b"
                                    strokeWidth="1"
                                />
                            )
                        })}
                    </g>
                )
            })}
            {Array.from({ length: maxLen }, (_, i) => (
                <text
                    key={`t${i}`}
                    x={xs(i)}
                    y={h - pad + 14}
                    fontSize="10"
                    fill="#475569"
                    textAnchor="middle"
                >
                    {dayInitials[i % 7]}
                </text>
            ))}
            <text x={pad} y={12} fontSize="11" fill="#475569">Seconds (log10)</text>
        </svg>
    )
}
