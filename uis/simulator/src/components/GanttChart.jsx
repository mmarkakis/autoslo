/**
 * GanttChart.jsx
 *
 * Renders a Gantt chart of query execution intervals using Plotly shapes.
 * Intervals are grouped by cluster, then packed greedily into lanes so that
 * non-overlapping intervals share a horizontal lane.
 *
 * Props:
 *   intervals  – array of TimelineInterval from the /timeline API
 *   sloS       – SLO in seconds (used for debug; SLO violation already in data)
 */
import { useMemo } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'

const Plot = createPlotlyComponent(Plotly)

const COLOR_MET = '#5dbb63'      // completed, slo met
const COLOR_VIOLATED = '#c0392b' // completed, slo violated
const COLOR_RUNNING = '#7f8c8d'  // still running / unknown

// Fractions along each bar where hover points are placed
const HOVER_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9]

function intervalColor(iv) {
  if (iv.state !== 'COMPLETED') return COLOR_RUNNING
  return iv.violates_slo ? COLOR_VIOLATED : COLOR_MET
}

/**
 * Greedy lane-pack a list of intervals (already sorted by start_s).
 * Returns an array of lane arrays: lanes[laneIdx] = [iv, ...].
 */
function packIntoLanes(sortedIntervals) {
  const lanes = []
  const laneLastEnd = []
  for (const iv of sortedIntervals) {
    let placed = false
    for (let i = 0; i < laneLastEnd.length; i++) {
      if (iv.start_s >= laneLastEnd[i]) {
        lanes[i].push(iv)
        laneLastEnd[i] = iv.end_s
        placed = true
        break
      }
    }
    if (!placed) {
      lanes.push([iv])
      laneLastEnd.push(iv.end_s)
    }
  }
  return lanes
}

/**
 * Build plot data for all intervals across all clusters.
 * Returns:
 *   shapes        – Plotly shape objects (rects + separator lines)
 *   hoverX/Y      – hover point coordinates (HOVER_FRACS.length points per bar)
 *   hoverText     – tooltip HTML strings
 *   hoverColor    – marker colors (invisible, just for lookup)
 *   clusterTicks  – { label, y } for y-axis labels
 *   totalY        – total height of the plot
 */
function buildPlotData(intervals) {
  if (!intervals || intervals.length === 0) {
    return { shapes: [], hoverX: [], hoverY: [], hoverText: [], hoverColor: [], clusterTicks: [], totalY: 10 }
  }

  // Group by cluster
  const byCluster = {}
  for (const iv of intervals) {
    const c = iv.cluster_name ?? 'unknown'
    if (!byCluster[c]) byCluster[c] = []
    byCluster[c].push(iv)
  }

  const LANE_HEIGHT = 1.0
  const CLUSTER_GAP = 1.5

  const shapes = []
  const hoverX = []
  const hoverY = []
  const hoverText = []
  const hoverColor = []
  const clusterTicks = []
  let yOffset = 0
  const clusterEntries = Object.entries(byCluster)

  for (let ci = 0; ci < clusterEntries.length; ci++) {
    const [clusterName, ivs] = clusterEntries[ci]

    const sorted = [...ivs].sort((a, b) => a.start_s - b.start_s)
    const lanes = packIntoLanes(sorted)

    const clusterYStart = yOffset
    for (let laneIdx = 0; laneIdx < lanes.length; laneIdx++) {
      const laneY = yOffset + laneIdx * LANE_HEIGHT
      for (const iv of lanes[laneIdx]) {
        const color = intervalColor(iv)
        const y0 = laneY
        const y1 = laneY + LANE_HEIGHT * 0.85

        shapes.push({
          type: 'rect',
          x0: iv.start_s,
          x1: iv.end_s,
          y0,
          y1,
          fillcolor: color,
          line: { width: 0 },
          opacity: 0.85,
        })

        const dur = (iv.end_s - iv.start_s).toFixed(2)
        const violation = iv.violates_slo ? '⚠ SLO violated' : '✓ SLO met'
        const sloLine = iv.slo_s != null ? `SLO: ${Number(iv.slo_s).toFixed(2)}s<br>` : ''
        const tooltip =
          `<b>Query ${iv.query_id}</b><br>` +
          (iv.tpcds_temp_and_q_idx != null ? `${iv.tpcds_temp_and_q_idx}<br>` : '') +
          `Cluster: ${iv.cluster_name}<br>` +
          sloLine +
          `${iv.start_s.toFixed(2)}s → ${iv.end_s.toFixed(2)}s (${dur}s)<br>` +
          `${iv.state} · ${violation}`

        const yCtr = (y0 + y1) / 2

        // Emit multiple hover points spread across the bar width
        for (const frac of HOVER_FRACS) {
          hoverX.push(iv.start_s + frac * (iv.end_s - iv.start_s))
          hoverY.push(yCtr)
          hoverText.push(tooltip)
          hoverColor.push(color)
        }
      }
    }

    const clusterYEnd = yOffset + lanes.length * LANE_HEIGHT
    clusterTicks.push({
      label: clusterName,
      y: (clusterYStart + clusterYEnd) / 2,
    })

    yOffset += lanes.length * LANE_HEIGHT + CLUSTER_GAP

    if (ci < clusterEntries.length - 1) {
      shapes.push({
        type: 'line',
        xref: 'paper',
        yref: 'y',
        x0: 0, x1: 1,
        y0: yOffset - CLUSTER_GAP / 2,
        y1: yOffset - CLUSTER_GAP / 2,
        line: { color: '#2d3748', width: 1 },
      })
    }
  }

  return { shapes, hoverX, hoverY, hoverText, hoverColor, clusterTicks, totalY: yOffset }
}

export default function GanttChart({ intervals, sloS }) {
  const { shapes, hoverX, hoverY, hoverText, hoverColor, clusterTicks, totalY } =
    useMemo(() => buildPlotData(intervals), [intervals])

  const hoverTrace = {
    type: 'scatter',
    mode: 'markers',
    x: hoverX,
    y: hoverY,
    text: hoverText,
    hovertemplate: '%{text}<extra></extra>',
    marker: {
      color: hoverColor,
      size: 8,
      opacity: 0,
    },
    showlegend: false,
  }

  // Dummy legend traces
  const legendTraces = [
    { name: 'SLO met', color: COLOR_MET },
    { name: 'SLO violated', color: COLOR_VIOLATED },
    { name: 'Running', color: COLOR_RUNNING },
  ].map(({ name, color }) => ({
    type: 'scatter',
    mode: 'markers',
    x: [null],
    y: [null],
    name,
    marker: { color, size: 12, symbol: 'square' },
    showlegend: true,
  }))

  const layout = {
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { color: '#e0e0e0', size: 11 },
    margin: { t: 30, r: 20, b: 60, l: 160 },
    shapes,
    xaxis: {
      title: 'Time (s)',
      gridcolor: '#2d3748',
      zeroline: false,
      color: '#a0aec0',
    },
    yaxis: {
      range: [-0.5, (totalY ?? 10) + 0.5],
      tickvals: clusterTicks?.map((t) => t.y) ?? [],
      ticktext: clusterTicks?.map((t) => t.label) ?? [],
      showgrid: false,
      zeroline: false,
      color: '#a0aec0',
      automargin: true,
    },
    legend: {
      orientation: 'h',
      x: 0,
      y: -0.15,
      bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#a0aec0', size: 11 },
    },
    hovermode: 'closest',
    autosize: true,
  }

  return (
    <Plot
      data={[hoverTrace, ...legendTraces]}
      layout={layout}
      useResizeHandler={true}
      style={{ width: '100%', height: '100%' }}
      config={{ displayModeBar: true, responsive: true }}
    />
  )
}

