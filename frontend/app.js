/**
 * DriveIQ — Frontend Core Application Logic
 * Pure Vanilla JavaScript (No React, 0 External Framework Dependencies)
 */

const API = '' // Same-origin direct backend API
const POLL_MS = 2500
const HEALTH_POLL_MS = 15000
const BACKEND_FAIL_THRESHOLD = 3

// ─── UTILITY HELPERS ──────────────────────────────────────────────────────────

function formatClock(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

function formatTime(sec) {
  return formatClock(sec)
}

function toNum(val, fallback = 0) {
  const n = Number(val)
  return Number.isFinite(n) ? n : fallback
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b)
  if (!sorted.length) return 0
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  const next = sorted[Math.min(base + 1, sorted.length - 1)]
  return sorted[base] + (next - sorted[base]) * rest
}

function deriveSeverityThresholds(windows = []) {
  const scores = windows
    .map((w) => toNum(w?.score, Number.NaN))
    .filter((v) => Number.isFinite(v))

  if (scores.length < 3) return { yellowMin: 50, greenMin: 75, mode: 'fixed' }
  const yellowMin = quantile(scores, 1 / 3)
  const greenMin = quantile(scores, 2 / 3)
  if (!Number.isFinite(yellowMin) || !Number.isFinite(greenMin) || Math.abs(greenMin - yellowMin) < 1e-6) {
    return { yellowMin: 50, greenMin: 75, mode: 'fixed' }
  }
  return { yellowMin, greenMin, mode: 'dynamic' }
}

function classifySeverity(score, thresholds) {
  const yellowMin = Number(thresholds?.yellowMin ?? 50)
  const greenMin = Number(thresholds?.greenMin ?? 75)
  if (score >= greenMin) return 'green'
  if (score >= yellowMin) return 'yellow'
  return 'red'
}

function severityClass(score) {
  if (score === 'green' || score === 'yellow' || score === 'red') {
    return `severity-${score}`
  }
  const s = Number(score)
  if (s >= 75) return 'severity-green'
  if (s >= 50) return 'severity-yellow'
  return 'severity-red'
}

function severityLabel(score) {
  if (score >= 75) return 'green'
  if (score >= 50) return 'yellow'
  return 'red'
}

function mostFrequent(items, key, fallback = '') {
  const counts = new Map()
  const firstIdx = new Map()
  items.forEach((item, idx) => {
    const val = String(item?.[key] ?? fallback)
    counts.set(val, (counts.get(val) || 0) + 1)
    if (!firstIdx.has(val)) firstIdx.set(val, idx)
  })
  let best = fallback
  let bestCount = -1
  let bestIdx = Number.MAX_SAFE_INTEGER
  counts.forEach((count, val) => {
    const idx = firstIdx.get(val) ?? Number.MAX_SAFE_INTEGER
    if (count > bestCount || (count === bestCount && idx < bestIdx)) {
      best = val
      bestCount = count
      bestIdx = idx
    }
  })
  return best
}

function buildSegment(windows, thresholds) {
  const first = windows[0]
  const last = windows[windows.length - 1]
  const scoreSum = windows.reduce((acc, w) => acc + toNum(w?.score), 0)
  const avgScore = scoreSum / Math.max(1, windows.length)
  const dominantIssue = mostFrequent(windows, 'top_issue', 'smooth_driving')
  const severity = classifySeverity(avgScore, thresholds)
  return {
    start_sec: toNum(first?.timestamp_sec),
    end_sec: toNum(last?.timestamp_sec),
    avg_score: Number(avgScore.toFixed(2)),
    dominant_issue: dominantIssue,
    severity,
    coach_note: first?.coach_note || '',
    windows,
  }
}

function groupSegments(windows = []) {
  if (!Array.isArray(windows) || !windows.length) return []
  const ordered = [...windows].sort((a, b) => toNum(a?.timestamp_sec) - toNum(b?.timestamp_sec))
  const thresholds = deriveSeverityThresholds(ordered)
  const segments = []
  let current = [ordered[0]]
  let runningSum = toNum(ordered[0]?.score)

  for (let i = 1; i < ordered.length; i++) {
    const prev = ordered[i - 1]
    const win = ordered[i]
    const runningAvg = runningSum / Math.max(1, current.length)
    const issueChanged = String(win?.top_issue ?? '') !== String(prev?.top_issue ?? '')
    const scoreJumped = Math.abs(toNum(win?.score) - runningAvg) > 10
    const bucketFull = (toNum(win?.timestamp_sec) - toNum(current[0]?.timestamp_sec)) >= 5

    if (issueChanged || scoreJumped || bucketFull) {
      segments.push(buildSegment(current, thresholds))
      current = [win]
      runningSum = toNum(win?.score)
      continue
    }
    current.push(win)
    runningSum += toNum(win?.score)
  }
  if (current.length) segments.push(buildSegment(current, thresholds))
  return segments
}

function buildTimelineModules(segments = []) {
  const moduleSize = 4
  const modules = []
  for (let i = 0; i < segments.length; i += moduleSize) {
    const chunk = segments.slice(i, i + moduleSize)
    const moduleIndex = Math.floor(i / moduleSize) + 1
    modules.push({
      id: `module-${moduleIndex}`,
      title: `Module ${String(moduleIndex).padStart(2, '0')}`,
      lessons: chunk.map((segment, idx) => {
        const issue = String(segment.dominant_issue || 'smooth driving').replaceAll('_', ' ')
        const duration = Math.max(1, Math.round(Number(segment.end_sec) - Number(segment.start_sec)))
        return {
          id: `${segment.start_sec}-${segment.end_sec}-${idx}`,
          label: `Lesson ${String(i + idx + 1).padStart(2, '0')}`,
          title: `${segment.coach_note || issue}`,
          segment,
          duration,
        }
      }),
    })
  }
  return modules
}

function decodeJwtPayload(token) {
  try {
    const payloadPart = String(token || '').split('.')[1]
    if (!payloadPart) return null
    const base64 = payloadPart.replace(/-/g, '+').replace(/_/g, '/')
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4)
    return JSON.parse(atob(padded))
  } catch {
    return null
  }
}

function isJwtExpired(token) {
  const payload = decodeJwtPayload(token)
  if (!payload || typeof payload.exp !== 'number') return true
  return (Date.now() / 1000) >= Number(payload.exp)
}

function generateTelemetry(t) {
  return {
    speed: 55 + Math.sin(t * 0.3) * 30,
    rpm: 2000 + Math.sin(t * 0.5) * 800,
    throttle_position: 30 + Math.sin(t * 0.4) * 20,
    gear: Math.floor(3 + Math.sin(t * 0.2) * 1.5),
    acceleration: Math.sin(t * 0.7) * 2,
    fuel_rate: 7 + Math.sin(t * 0.3) * 2,
  }
}

function generateSyntheticFrameB64(t, telemetry) {
  const canvas = document.createElement('canvas')
  canvas.width = 320
  canvas.height = 180
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  const grad = ctx.createLinearGradient(0, 0, 0, 180)
  grad.addColorStop(0, '#60a5fa')
  grad.addColorStop(0.55, '#93c5fd')
  grad.addColorStop(0.56, '#334155')
  grad.addColorStop(1, '#0f172a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 320, 180)

  const laneShift = Math.sin(t * 1.1) * 8
  ctx.strokeStyle = 'rgba(255,255,255,0.7)'
  ctx.setLineDash([10, 10])
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(160 + laneShift - 20, 180)
  ctx.lineTo(145 + laneShift, 95)
  ctx.moveTo(160 + laneShift + 20, 180)
  ctx.lineTo(175 + laneShift, 95)
  ctx.stroke()
  ctx.setLineDash([])

  const speed = Number(telemetry?.speed ?? 50)
  const carW = Math.max(18, Math.min(48, 18 + speed * 0.2))
  const carH = Math.max(10, Math.min(28, 10 + speed * 0.11))
  const carX = 160 + Math.sin(t * 0.7) * 12 - carW / 2
  const carY = 120 - Math.sin(t * 0.45) * 6
  ctx.fillStyle = '#ef4444'
  ctx.fillRect(carX, carY, carW, carH)

  const dataUrl = canvas.toDataURL('image/jpeg', 0.75)
  return dataUrl.split(',')[1]
}

function getLiveInsights(features, score) {
  const insights = []
  if (score >= 80) {
    insights.push({ label: 'Excellent driving behavior', type: 'green' })
  } else if (score < 50) {
    insights.push({ label: 'Eco score is critical', type: 'red' })
  }

  if (features?.braking_flag === 1 || features?.braking_flag_ratio > 0) {
    insights.push({ label: 'Harsh braking detected (-score)', type: 'red' })
  }
  if (Number(features?.lane_change_flag) > 0) {
    insights.push({ label: 'Erratic swerving / lane changes', type: 'yellow' })
  }
  if (Number(features?.proximity_score) > 0.15) {
    insights.push({ label: 'Following distance too close (tailgating)', type: 'red' })
  } else if (Number(features?.proximity_score) > 0.05) {
    insights.push({ label: 'Moderate following distance', type: 'yellow' })
  }
  if (features?.erratic_flag === 1) {
    insights.push({ label: 'High optical velocity changes', type: 'yellow' })
  }

  if (insights.length === 0 || (insights.length === 1 && insights[0].type === 'green')) {
    insights.push({ label: 'Maintaining smooth, safe flow (+score)', type: 'green' })
  }

  const unique = new Map()
  insights.forEach(i => unique.set(i.label, i))
  return Array.from(unique.values())
}

const ZERO_FEATURES = {
  pedestrian_flag: 0,
  vehicle_density: 0,
  braking_flag: 0,
  lane_change_flag: 0,
  proximity_score: 0,
  mean_flow: 0,
  flow_variance: 0,
}

// ─── SVG VISUALIZATION RENDERERS ──────────────────────────────────────────────

function renderScoreGauge(score, container) {
  if (!container) return
  const s = Math.round(score ?? 0)
  let color = 'rgba(234, 234, 234, 0.44)'
  let label = 'Rough'
  if (s >= 75) {
    color = 'rgba(234, 234, 234, 0.9)'
    label = 'Smooth'
  } else if (s >= 50) {
    color = 'rgba(234, 234, 234, 0.66)'
    label = 'Fair'
  }

  const radius = 76
  const circumference = 2 * Math.PI * radius
  const maxArc = (240 / 360) * circumference
  const filledArc = (Math.max(0, Math.min(100, s)) / 100) * maxArc

  container.innerHTML = `
    <div class="card">
      <div class="card-title">Drive Score</div>
      <div class="gauge-wrap">
        <svg viewBox="0 0 200 200" width="100%" height="100%">
          <circle
            cx="100" cy="100" r="${radius}"
            fill="none"
            stroke="rgba(234, 234, 234, 0.08)"
            stroke-width="14"
            stroke-dasharray="${maxArc} ${circumference}"
            stroke-linecap="round"
            transform="rotate(150 100 100)"
          />
          <circle
            cx="100" cy="100" r="${radius}"
            fill="none"
            stroke="${color}"
            stroke-width="14"
            stroke-dasharray="${filledArc} ${circumference}"
            stroke-linecap="round"
            transform="rotate(150 100 100)"
            style="transition: stroke-dasharray 0.5s ease, stroke 0.3s ease"
          />
        </svg>
        <div class="gauge-center">
          <span class="gauge-score" style="color: ${color}">${s}</span>
          <span class="gauge-sub">/ 100</span>
          <span class="gauge-label">${label}</span>
        </div>
      </div>
    </div>
  `
}

function renderTrendChart(points = [], container, emptyMessage = 'No data yet') {
  if (!container) return
  if (!points || !points.length) {
    container.innerHTML = `
      <div class="card">
        <div class="card-title">Score Trend</div>
        <div class="trend-empty">${emptyMessage}</div>
      </div>
    `
    return
  }

  const width = 500
  const height = 180
  const paddingLeft = 36
  const paddingRight = 48
  const paddingTop = 20
  const paddingBottom = 28
  const plotWidth = width - paddingLeft - paddingRight
  const plotHeight = height - paddingTop - paddingBottom

  const getY = (val) => {
    const clamped = Math.max(0, Math.min(100, Number(val) || 0))
    return paddingTop + (1 - clamped / 100) * plotHeight
  }

  const getX = (idx) => {
    if (points.length <= 1) return paddingLeft + plotWidth / 2
    return paddingLeft + (idx / (points.length - 1)) * plotWidth
  }

  const coords = points.map((p, idx) => ({
    x: getX(idx),
    y: getY(p.avg_score ?? p.score ?? 0),
    score: Number(p.avg_score ?? p.score ?? 0),
    time: formatTime(p.start_sec ?? p.timestamp_sec),
    severity: p.severity,
  }))

  const polylinePoints = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' ')
  const polygonPoints = `${coords[0].x.toFixed(1)},${(paddingTop + plotHeight).toFixed(1)} ${polylinePoints} ${coords[coords.length - 1].x.toFixed(1)},${(paddingTop + plotHeight).toFixed(1)}`

  const step = Math.max(1, Math.floor((points.length - 1) / 4))
  const tickIndices = []
  for (let i = 0; i < points.length; i += step) {
    tickIndices.push(i)
  }
  if (tickIndices[tickIndices.length - 1] !== points.length - 1) {
    tickIndices.push(points.length - 1)
  }

  const yTarget = getY(75)
  const yBaseline = getY(50)

  const gridLines = [100, 75, 50, 25, 0].map((val) => {
    const y = getY(val)
    return `
      <g>
        <line x1="${paddingLeft}" y1="${y}" x2="${width - paddingRight}" y2="${y}" stroke="rgba(234, 234, 234, 0.06)" stroke-width="1" />
        <text x="${paddingLeft - 6}" y="${y + 3}" text-anchor="end" fill="rgba(234, 234, 234, 0.4)" font-size="9" font-family="Inter, sans-serif">${val}</text>
      </g>
    `
  }).join('')

  const pointsCircles = coords.map((pt) => {
    let ptColor = 'rgba(234, 234, 234, 0.44)'
    if (pt.severity === 'green') ptColor = 'rgba(234, 234, 234, 0.9)'
    else if (pt.severity === 'yellow') ptColor = 'rgba(234, 234, 234, 0.66)'
    return `
      <circle cx="${pt.x}" cy="${pt.y}" r="3" fill="${ptColor}" stroke="#0e0e0e" stroke-width="1">
        <title>${pt.time} - Score: ${pt.score.toFixed(1)}</title>
      </circle>
    `
  }).join('')

  const xTicks = tickIndices.map((i) => {
    const pt = coords[i]
    if (!pt) return ''
    return `<text x="${pt.x}" y="${height - 6}" text-anchor="middle" fill="rgba(234, 234, 234, 0.45)" font-size="9" font-family="Inter, sans-serif">${pt.time}</text>`
  }).join('')

  container.innerHTML = `
    <div class="card">
      <div class="card-title">Score Trend</div>
      <div class="trend-wrap" style="position: relative;">
        <svg width="100%" height="100%" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" style="overflow: visible; display: block;">
          ${gridLines}
          <line x1="${paddingLeft}" y1="${yTarget}" x2="${width - paddingRight}" y2="${yTarget}" stroke="rgba(234, 234, 234, 0.45)" stroke-dasharray="4 4" stroke-width="1" />
          <text x="${width - paddingRight + 4}" y="${yTarget + 3}" fill="rgba(234, 234, 234, 0.55)" font-size="9" font-family="Inter, sans-serif">Target</text>
          <line x1="${paddingLeft}" y1="${yBaseline}" x2="${width - paddingRight}" y2="${yBaseline}" stroke="rgba(234, 234, 234, 0.3)" stroke-dasharray="4 4" stroke-width="1" />
          <text x="${width - paddingRight + 4}" y="${yBaseline + 3}" fill="rgba(234, 234, 234, 0.4)" font-size="9" font-family="Inter, sans-serif">Base</text>
          <polygon points="${polygonPoints}" fill="rgba(234, 234, 234, 0.05)" />
          <polyline points="${polylinePoints}" fill="none" stroke="rgba(234, 234, 234, 0.35)" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round" />
          ${pointsCircles}
          ${xTicks}
        </svg>
      </div>
    </div>
  `
}

function renderBrakingMeter(ratio, container) {
  if (!container) return
  const isFlash = ratio > 0.3
  const heightPct = Math.min(100, ratio * 200)
  container.innerHTML = `
    <div style="height: 100%; display: flex; flex-direction: column; align-items: center; margin-left: 12px; width: 20px;">
      <span style="font-size: 9px; color: var(--c-white-46); writing-mode: vertical-rl; transform: rotate(180deg); margin-bottom: 8px;">Brake Force</span>
      <div style="flex: 1; width: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 4px; display: flex; align-items: flex-end; overflow: hidden; box-shadow: ${isFlash ? '0 0 10px rgba(255, 77, 77, 0.8)' : 'none'}; transition: box-shadow 0.2s;">
        <div style="width: 100%; height: ${heightPct}%; background: ${isFlash ? '#ff4d4d' : 'var(--c-white-72)'}; transition: height 0.1s linear, background 0.2s;"></div>
      </div>
    </div>
  `
}

function renderProximityHeatstrip(score, container) {
  if (!container) return
  let bg = '#4caf50'
  let label = 'Safe Distance'
  if (score > 0.15) {
    bg = '#ff4d4d'
    label = 'Tailgating Risk'
  } else if (score > 0.05) {
    bg = '#ffcc00'
    label = 'Following Closely'
  }
  const pct = (score * 100).toFixed(0)
  const opacity = Math.min(1, 0.4 + (score * 2))

  container.innerHTML = `
    <div style="width: 100%; margin-bottom: 12px;">
      <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--c-white-46);">Proximity Radar</span>
        <span style="font-size: 11px; color: ${bg}; font-weight: 600;">${label} (${pct}%)</span>
      </div>
      <div style="width: 100%; height: 8px; border-radius: 4px; background: var(--c-white-08); overflow: hidden;">
        <div style="height: 100%; width: 100%; background: ${bg}; opacity: ${opacity}; transition: all 0.5s ease-out;"></div>
      </div>
    </div>
  `
}

function renderSparklineSvg(data, maxProp, color) {
  if (!data || data.length < 2) return ''
  const max = maxProp || Math.max(...data, 1)
  const min = 0
  const range = max - min || 1
  const w = 40
  const h = 12
  const pts = data.map((val, idx) => {
    const x = (idx / (data.length - 1)) * w
    const y = h - ((val - min) / range) * h
    return `${x},${y}`
  }).join(' ')
  return `
    <svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="margin-left: 8px; vertical-align: middle; overflow: visible;">
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1" />
    </svg>
  `
}

function renderFeatureTable(features, history = [], container) {
  if (!container) return
  const fmt = (val, d = 2) => {
    const n = Number(val)
    return Number.isFinite(n) ? n.toFixed(d) : '-'
  }

  const proxHist = (history || []).map(p => Number(p.features?.proximity_score || 0))
  const brakeHist = (history || []).map(p => Number(p.features?.braking_ratio || p.features?.braking_flag || 0))

  const flagLabel = (f) => (Number(f) === 1 ? 'Detected' : 'None')

  const rows = [
    { label: 'Pedestrian Detected', val: flagLabel(features?.pedestrian_flag), flag: Number(features?.pedestrian_flag) === 1 },
    { label: 'Vehicles Density', val: fmt(features?.vehicle_density, 1) },
    { label: 'Braking', val: flagLabel(features?.braking_flag), flag: Number(features?.braking_flag) === 1, spark: renderSparklineSvg(brakeHist, 1, '#ff4d4d') },
    { label: 'Lane Change', val: flagLabel(features?.lane_change_flag), flag: Number(features?.lane_change_flag) === 1 },
    { label: 'Proximity', val: fmt(features?.proximity_score, 4), spark: renderSparklineSvg(proxHist, 0.3, '#ffcc00') },
    { label: 'Mean Flow', val: fmt(features?.mean_flow, 4) },
    { label: 'Flow Variance', val: fmt(features?.flow_variance, 4) },
  ]

  const trs = rows.map(r => `
    <tr>
      <td>${r.label}</td>
      <td class="${r.flag ? 'feature-flag' : ''}" style="display: flex; align-items: center;">
        ${r.val} ${r.spark || ''}
      </td>
    </tr>
  `).join('')

  container.innerHTML = `
    <div class="card">
      <div class="card-title">Feature Snapshot</div>
      <table class="feat-table">
        <tbody>
          ${trs}
        </tbody>
      </table>
    </div>
  `
}

function renderCoachingPanel(tips, severity, source, topIssue, loading, message, container) {
  if (!container) return
  const sev = ['green', 'yellow', 'red'].includes(severity) ? severity : 'yellow'
  const badgeCls = `severity-badge severity-${sev}`
  const sevLabel = { green: 'Good', yellow: 'Fair', red: 'Poor' }[sev]
  const issueLabel = topIssue ? String(topIssue).replaceAll('_', ' ') : null

  const tipsHtml = (tips || []).map((tip, i) => `
    <li class="tip-item">
      <span class="tip-num">${i + 1}</span>
      <span class="tip-copy">${tip}</span>
    </li>
  `).join('')

  container.innerHTML = `
    <div class="card">
      <div class="coach-header">
        <div class="card-title coach-title">Coaching</div>
        <div class="coach-badges">
          ${issueLabel ? `<span class="severity-badge severity-neutral">${issueLabel}</span>` : ''}
          <span class="${badgeCls}">${sevLabel}</span>
        </div>
      </div>
      ${source ? `<div class="coach-source mt-1 mb-2">Source: ${source}</div>` : ''}
      ${message ? `<p class="coach-message">${message}</p>` : ''}
      ${loading ? `<p class="text-mute coach-loading">Generating tips...</p>` : `<ul class="tip-list tip-list-plain">${tipsHtml}</ul>`}
    </div>
  `
}

// ─── STATE MANAGEMENT ─────────────────────────────────────────────────────────

const state = {
  token: localStorage.getItem('driveiq_token') || '',
  currentView: 'live', // 'live' | 'review' | 'history'
  scoringMode: 'xgboost', // 'xgboost' | 'event_rules'

  healthState: 'checking',
  healthMessage: 'Checking backend health...',
  healthMeta: { schema_valid: false, core_models_loaded: false },
  offlineMode: false,
  healthFailCount: 0,
  sessionSaveWarning: '',

  // Live state
  liveScore: 0,
  liveFeatures: { ...ZERO_FEATURES },
  livePoints: [],
  liveEvents: [],
  liveVideoFile: null,
  liveVideoUrl: null,
  streamActive: false,
  streamComplete: false,
  livePlayback: { current: 0, duration: 0 },
  clock: 0,
  sessionId: `sess-${Math.random().toString(36).slice(2)}`,
  sessionStartedAt: Date.now(),
  prevFrameB64: null,
  cleanFrameCount: 0,
  lastPositiveTime: 0,

  // Review state
  reviewFile: null,
  reviewLoading: false,
  reviewResult: null,
  selectedWindow: null,
  expandedModules: {},

  // History state
  historyMetrics: { mean_eco_score: 0, lowest_eco_score: 0, total_trips: 0, trips_this_week: 0 },
  tripHistory: [],
  selectedTrip: null,
  selectedTripTimeline: [],
  selectedTripCoach: null,

  // Auth Modal
  showAuth: false,
  authIsRegister: false,
}

// ─── DOM ELEMENT REFERENCES ───────────────────────────────────────────────────

const el = {
  // Navigation
  navHistory: document.getElementById('nav-history'),
  navReview: document.getElementById('nav-review'),
  navLive: document.getElementById('nav-live'),
  navInsights: document.getElementById('nav-insights'),
  btnModeLive: document.getElementById('btn-mode-live'),
  btnModeHistory: document.getElementById('btn-mode-history'),
  btnModeReview: document.getElementById('btn-mode-review'),

  // Views
  liveView: document.getElementById('live-view'),
  reviewView: document.getElementById('review-view'),
  historyView: document.getElementById('history-view'),

  // Banners & Status
  sidebarStatusDot: document.getElementById('sidebar-status-dot'),
  sidebarStatusText: document.getElementById('sidebar-status-text'),
  sidebarStatusBadge: document.getElementById('sidebar-status-badge'),
  healthBanner: document.getElementById('health-banner'),
  sessionWarning: document.getElementById('session-warning'),
  offlineControls: document.getElementById('offline-controls'),
  btnReconnect: document.getElementById('btn-reconnect'),

  // Topbar
  btnAuth: document.getElementById('btn-auth'),
  profileTitle: document.getElementById('profile-title'),
  profileSub: document.getElementById('profile-sub'),
  profileDot: document.getElementById('profile-dot'),

  // Mini Dashboard
  miniScore: document.getElementById('mini-score'),
  miniScoreSub: document.getElementById('mini-score-sub'),
  miniDuration: document.getElementById('mini-duration'),
  miniWindows: document.getElementById('mini-windows'),
  miniFuel: document.getElementById('mini-fuel'),

  // Live Elements
  btnScoringMode: document.getElementById('btn-scoring-mode'),
  scoringModeText: document.getElementById('scoring-mode-text'),
  liveFileInput: document.getElementById('live-file-input'),
  btnStartStream: document.getElementById('btn-start-stream'),
  liveCompleteActions: document.getElementById('live-complete-actions'),
  btnNewSession: document.getElementById('btn-new-session'),
  liveVideoPlaceholder: document.getElementById('live-video-placeholder'),
  liveVideo: document.getElementById('live-video'),
  liveStatusMeta: document.getElementById('live-status-meta'),
  liveClockMeta: document.getElementById('live-clock-meta'),
  liveEventsMeta: document.getElementById('live-events-meta'),
  liveProgressFill: document.getElementById('live-progress-fill'),
  liveLessonCount: document.getElementById('live-lesson-count'),
  liveTimelineList: document.getElementById('live-timeline-list'),
  evMotionLabel: document.getElementById('ev-motion-label'),
  evBrakingCount: document.getElementById('ev-braking-count'),
  evTailgatingCount: document.getElementById('ev-tailgating-count'),
  evSpeedCount: document.getElementById('ev-speed-count'),
  proximityHeatstripContainer: document.getElementById('proximity-heatstrip-container'),
  brakingMeterContainer: document.getElementById('braking-meter-container'),
  liveTrendChartContainer: document.getElementById('live-trend-chart-container'),

  // Review Elements
  reviewFileInput: document.getElementById('review-file-input'),
  btnRunReview: document.getElementById('btn-run-review'),
  btnExportReport: document.getElementById('btn-export-report'),
  reviewStatus: document.getElementById('review-status'),
  reviewError: document.getElementById('review-error'),
  reviewResultsArea: document.getElementById('review-results-area'),
  reviewVideo: document.getElementById('review-video'),
  reviewClockMeta: document.getElementById('review-clock-meta'),
  reviewWindowsMeta: document.getElementById('review-windows-meta'),
  reviewProgressFill: document.getElementById('review-progress-fill'),
  reviewTimelineModules: document.getElementById('review-timeline-modules'),
  reviewReportCopy: document.getElementById('review-report-copy'),
  reviewTrendChartContainer: document.getElementById('review-trend-chart-container'),

  // History Elements
  historyTripInspection: document.getElementById('history-trip-inspection'),
  histMeanScore: document.getElementById('hist-mean-score'),
  histLowestScore: document.getElementById('hist-lowest-score'),
  histTotalTrips: document.getElementById('hist-total-trips'),
  histTripsWeek: document.getElementById('hist-trips-week'),
  tripsTableBody: document.getElementById('trips-table-body'),

  // Insights Containers
  scoreGaugeContainer: document.getElementById('score-gauge-container'),
  featureTableContainer: document.getElementById('feature-table-container'),
  coachingPanelContainer: document.getElementById('coaching-panel-container'),

  // Auth Modal
  authModal: document.getElementById('auth-modal'),
  authModalTitle: document.getElementById('auth-modal-title'),
  authError: document.getElementById('auth-error'),
  authForm: document.getElementById('auth-form'),
  authEmail: document.getElementById('auth-email'),
  authPassword: document.getElementById('auth-password'),
  authSubmitBtn: document.getElementById('auth-submit-btn'),
  authToggleBtn: document.getElementById('auth-toggle-btn'),
  btnCloseAuth: document.getElementById('btn-close-auth'),
}

// ─── UI RENDER DISPATCHERS ────────────────────────────────────────────────────

function updateViewDisplay() {
  const v = state.currentView
  el.liveView.style.display = v === 'live' ? 'block' : 'none'
  el.reviewView.style.display = v === 'review' ? 'block' : 'none'
  el.historyView.style.display = v === 'history' ? 'block' : 'none'

  // Update nav highlight
  el.navLive.classList.toggle('active', v === 'live')
  el.navReview.classList.toggle('active', v === 'review')
  el.navHistory.classList.toggle('active', v === 'history')

  el.btnModeLive.classList.toggle('active', v === 'live')
  el.btnModeReview.classList.toggle('active', v === 'review')
  el.btnModeHistory.classList.toggle('active', v === 'history')

  updateMiniDashboard()
  updateInsights()
}

function updateAuthDisplay() {
  const loggedIn = Boolean(state.token)
  if (loggedIn) {
    el.profileTitle.textContent = 'Verified Driver'
    el.profileSub.textContent = 'Session saving enabled'
    el.profileDot.style.background = '#22c55e'
    el.btnAuth.textContent = 'Logout'
    el.btnAuth.className = 'btn'
  } else {
    el.profileTitle.textContent = 'Guest Driver'
    el.profileSub.textContent = 'Session saving disabled'
    el.profileDot.style.background = 'var(--c-white-46)'
    el.btnAuth.textContent = 'Login'
    el.btnAuth.className = 'btn btn-primary'
  }
}

function updateHealthBanners() {
  const schemaOk = state.healthMeta.schema_valid
  const modelsOk = state.healthMeta.core_models_loaded
  const backendReady = schemaOk && modelsOk

  el.sidebarStatusDot.className = `status-dot ${backendReady ? 'ok' : 'bad'}`
  el.sidebarStatusText.textContent = backendReady ? 'Backend Ready' : 'Backend Degraded'
  el.sidebarStatusBadge.className = `severity-badge ${backendReady ? 'severity-green' : 'severity-red'}`
  el.sidebarStatusBadge.textContent = `schema: ${String(schemaOk)} | models: ${String(modelsOk)}`

  el.healthBanner.className = `health-banner health-${state.healthState}`
  el.healthBanner.textContent = state.healthMessage

  if (state.sessionSaveWarning) {
    el.sessionWarning.style.display = 'block'
    el.sessionWarning.textContent = state.sessionSaveWarning
  } else {
    el.sessionWarning.style.display = 'none'
  }

  el.offlineControls.style.display = state.offlineMode ? 'block' : 'none'
}

function updateMiniDashboard() {
  const isLive = state.currentView === 'live'
  if (isLive) {
    el.miniScore.textContent = Math.round(state.liveScore || 0)
    el.miniScoreSub.textContent = 'Live streaming'
    el.miniDuration.textContent = 'Live'
    el.miniWindows.textContent = '—'
    el.miniFuel.textContent = '—'
  } else if (state.reviewResult) {
    const avg = Math.round(state.reviewResult.avg_batch_score || 0)
    el.miniScore.textContent = avg
    el.miniScoreSub.textContent = 'Selected video analysis'
    el.miniDuration.textContent = formatClock(state.reviewResult.duration_sec || 0)
    el.miniWindows.textContent = state.reviewResult.window_count || 0
    el.miniFuel.textContent = `${Math.max(0, (avg - 50) / 18).toFixed(1)}L`
  } else {
    el.miniScore.textContent = '—'
    el.miniScoreSub.textContent = 'No trip selected'
    el.miniDuration.textContent = '—'
    el.miniWindows.textContent = '—'
    el.miniFuel.textContent = '—'
  }
}

function updateInsights() {
  const isLive = state.currentView === 'live'
  let score = 0
  let feats = { ...ZERO_FEATURES }
  let coachNote = 'Select a segment to view coaching note.'
  let severity = 'yellow'
  let topIssue = ''

  if (isLive) {
    score = state.liveScore
    feats = state.liveFeatures
    coachNote = state.streamActive ? 'Live mode active. Real-time insights appear above.' : 'Awaiting live stream.'
    severity = severityLabel(score)
  } else if (state.selectedWindow) {
    score = Number(state.selectedWindow.avg_score ?? state.selectedWindow.score ?? 0)
    feats = {
      pedestrian_flag: Number(state.selectedWindow?.pedestrian_ratio ?? state.selectedWindow?.pedestrian_flag ?? 0) > 0 ? 1 : 0,
      vehicle_density: Number(state.selectedWindow?.vehicle_density ?? state.selectedWindow?.vehicle_count ?? 0),
      braking_flag: Number(state.selectedWindow?.braking_flag_ratio ?? 0) > 0 ? 1 : 0,
      lane_change_flag: Number(state.selectedWindow?.lane_change_flag_ratio ?? 0) > 0 ? 1 : 0,
      proximity_score: Number(state.selectedWindow?.proximity_score_mean ?? 0),
      mean_flow: Number(state.selectedWindow?.mean_flow_mean ?? 0),
      flow_variance: Number(state.selectedWindow?.flow_variance ?? 0),
    }
    coachNote = state.selectedWindow.coach_note || 'Analysis complete for selected window.'
    severity = state.selectedWindow.severity || 'yellow'
    topIssue = state.selectedWindow.dominant_issue || state.selectedWindow.top_issue || ''
  }

  renderScoreGauge(score, el.scoreGaugeContainer)
  renderFeatureTable(feats, isLive ? state.livePoints : [], el.featureTableContainer)
  renderCoachingPanel([coachNote], severity, isLive ? 'stream' : 'review', topIssue, false, '', el.coachingPanelContainer)
}

function updateLiveEventCounters() {
  const events = state.liveEvents
  const meanFlow = state.liveFeatures?.mean_flow || 0

  const braking = events.filter(e => e.label.includes('braking')).length
  const tailgating = events.filter(e => e.label.includes('tailgating')).length
  const speed = events.filter(e => e.label.includes('velocity')).length

  let motionLabel = 'Idle'
  let motionColor = 'var(--c-white-46)'
  if (meanFlow >= 20) {
    motionLabel = 'Fast'
    motionColor = 'var(--c-red-bright)'
  } else if (meanFlow >= 8) {
    motionLabel = 'Moderate'
    motionColor = 'var(--c-yellow-bright)'
  } else if (meanFlow >= 1) {
    motionLabel = 'Slow'
    motionColor = 'var(--c-primary)'
  }

  el.evMotionLabel.textContent = motionLabel
  el.evMotionLabel.style.color = motionColor
  el.evBrakingCount.textContent = braking
  el.evTailgatingCount.textContent = tailgating
  el.evSpeedCount.textContent = speed
}

function updateLiveTimelineList() {
  el.liveLessonCount.textContent = `${state.liveEvents.length} lessons`
  if (!state.liveEvents.length) {
    el.liveTimelineList.innerHTML = `
      <div class="empty-state">
        <p class="empty-state-text">${state.streamActive ? 'Analyzing stream with no significant infractions yet.' : 'Start stream to generate timeline lessons.'}</p>
      </div>
    `
    return
  }

  el.liveTimelineList.innerHTML = state.liveEvents.map((insight, idx) => {
    const label = formatClock(insight.timestamp_sec)
    return `
      <div class="timeline-lesson">
        <span class="timeline-lesson-label">Lesson ${String(idx + 1).padStart(2, '0')}</span>
        <button type="button" class="timeline-video-item" data-ts="${insight.timestamp_sec}">
          <span class="line-clamp-2">${insight.label}</span>
          <div class="timeline-video-meta">
            <span>${label}</span>
            ${insight.score != null ? `<span style="font-size: 11px; color: var(--c-white-72); font-weight: 600;">Score: ${insight.score}</span>` : ''}
            <span class="severity-badge severity-${insight.type}">${insight.type}</span>
          </div>
        </button>
      </div>
    `
  }).join('')

  el.liveTimelineList.querySelectorAll('.timeline-video-item').forEach(btn => {
    btn.onclick = () => {
      const ts = Number(btn.dataset.ts) || 0
      if (el.liveVideo) {
        el.liveVideo.currentTime = ts
        el.liveVideo.play().catch(() => {})
      }
    }
  })
}

// ─── API ACTIONS & POLLING ────────────────────────────────────────────────────

async function fetchHealth() {
  if (state.offlineMode) return

  try {
    const res = await fetch(`${API}/api/v1/health`)
    const data = await res.json()
    state.healthFailCount = 0
    state.offlineMode = false
    state.healthMeta = {
      schema_valid: Boolean(data?.schema_valid),
      core_models_loaded: Boolean(data?.core_models_loaded),
    }

    const ready = data?.ready === true || data?.score_ready === true
    if (ready) {
      state.healthState = 'ready'
      state.healthMessage = 'Backend ready for review and scoring.'
    } else {
      state.healthState = 'degraded'
      const reasonParts = []
      if (data?.schema_error) reasonParts.push(data.schema_error)
      if (data?.coach_status === 'loading') reasonParts.push('Coach model warming up')
      if (data?.coach_error) reasonParts.push(`Coach error: ${data.coach_error}`)
      state.healthMessage = `Backend reachable but degraded.${reasonParts.length ? ' ' + reasonParts.join(' | ') : ''}`
    }
  } catch {
    state.healthFailCount += 1
    if (state.healthFailCount >= BACKEND_FAIL_THRESHOLD) {
      state.offlineMode = true
      state.healthState = 'degraded'
      state.healthMessage = 'Backend unavailable. UI is in offline placeholder mode.'
    } else {
      state.healthState = 'error'
      state.healthMessage = 'Cannot reach backend health endpoint.'
    }
  }
  updateHealthBanners()
}

async function fetchScore() {
  if (state.currentView !== 'live' || state.offlineMode || state.healthState === 'error') return
  if (state.streamComplete) return

  let frameB64 = null
  const v = el.liveVideo
  if (v && v.readyState >= 2 && !v.paused && !v.ended) {
    const canvas = document.createElement('canvas')
    canvas.width = 480
    canvas.height = 270
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.drawImage(v, 0, 0, 480, 270)
      frameB64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
      state.clock = v.currentTime
    }
  } else if (v && !v.paused && !v.ended) {
    // waiting
  } else if (state.streamActive) {
    state.clock += POLL_MS / 1000
    frameB64 = generateSyntheticFrameB64(state.clock, generateTelemetry(state.clock))
  } else {
    return
  }

  const hasRealVideo = v && v.readyState >= 2
  const telemetry = hasRealVideo
    ? { speed: 0, rpm: 0, throttle_position: 0, gear: 0, acceleration: 0, fuel_rate: 0 }
    : generateTelemetry(state.clock)

  const tokenExpired = Boolean(state.token) && isJwtExpired(state.token)
  const headers = { 'Content-Type': 'application/json' }
  if (state.token && !tokenExpired) {
    headers.Authorization = `Bearer ${state.token}`
  } else if (state.token && tokenExpired) {
    state.sessionSaveWarning = 'Session expired. Live scoring continues, but this drive is not being saved.'
    updateHealthBanners()
  }

  try {
    const res = await fetch(`${API}/api/v1/score`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        telemetry,
        session_id: state.sessionId,
        session_started_at: state.sessionStartedAt,
        frame_b64: frameB64,
        prev_frame_b64: state.prevFrameB64,
        scoring_mode: state.scoringMode,
      }),
    })
    const data = await res.json()
    state.prevFrameB64 = frameB64

    if (state.token && !tokenExpired) {
      if (data?.auth_failed) {
        state.sessionSaveWarning = 'Authentication failed. Live scoring continues, but this drive is not being saved.'
      } else if (data?.session_saved === false) {
        state.sessionSaveWarning = 'Live scoring active, but session saving is unavailable.'
      } else if (data?.session_saved === true) {
        state.sessionSaveWarning = ''
      }
      updateHealthBanners()
    }

    const s = Number(data.score ?? 0)
    state.liveScore = s
    state.liveFeatures = { ...data.features }

    const insights = getLiveInsights(state.liveFeatures, s)
    const severity = severityLabel(s)

    state.livePoints.push({ timestamp_sec: state.clock, score: s, severity })
    if (state.livePoints.length > 60) state.livePoints.shift()

    const warnings = insights.filter((i) => i.type !== 'green')
    if (warnings.length > 0) {
      const worst = warnings.some(w => w.type === 'red') ? 'red' : 'yellow'
      const summary = warnings.map(w => w.label).join(' | ')
      state.liveEvents.unshift({ label: summary, type: worst, timestamp_sec: state.clock, score: Math.round(s) })
      if (state.liveEvents.length > 30) state.liveEvents.length = 30
      state.cleanFrameCount = 0
    } else {
      state.cleanFrameCount += 1
      const now = Date.now()
      const secSinceLastPositive = (now - state.lastPositiveTime) / 1000
      if (s > 85 && state.cleanFrameCount >= 4 && secSinceLastPositive >= 10) {
        const positiveMessages = [
          'Smooth driving maintained — excellent fuel efficiency.',
          'Good lane discipline and steady speed.',
          'Safe following distance — keep it up!',
          'Consistent eco-driving behavior detected.',
          'No infractions — optimal driving pattern.',
        ]
        const msg = positiveMessages[Math.floor(Math.random() * positiveMessages.length)]
        state.liveEvents.unshift({ label: msg, type: 'green', timestamp_sec: state.clock, score: Math.round(s) })
        if (state.liveEvents.length > 30) state.liveEvents.length = 30
        state.lastPositiveTime = now
        state.cleanFrameCount = 0
      }
    }

    // Update visuals
    updateMiniDashboard()
    updateLiveEventCounters()
    updateLiveTimelineList()
    renderProximityHeatstrip(Number(state.liveFeatures?.proximity_score || 0), el.proximityHeatstripContainer)
    renderBrakingMeter(Number(state.liveFeatures?.braking_ratio || state.liveFeatures?.braking_flag || 0), el.brakingMeterContainer)
    renderTrendChart(state.livePoints, el.liveTrendChartContainer, 'Stream a video to generate live trend mapping')
    updateInsights()

  } catch {
    state.healthState = 'degraded'
    state.healthMessage = 'Backend unstable. Retrying score stream...'
    updateHealthBanners()
  }
}

// ─── REVIEW WORKFLOW ──────────────────────────────────────────────────────────

async function runReviewAnalysis() {
  if (!state.reviewFile) {
    el.reviewError.style.display = 'block'
    el.reviewError.textContent = 'Select an MP4 file first.'
    return
  }

  el.reviewError.style.display = 'none'
  el.reviewStatus.style.display = 'block'
  el.reviewStatus.textContent = 'Uploading and initializing analysis...'
  el.btnRunReview.disabled = true

  try {
    const form = new FormData()
    form.append('video', state.reviewFile)
    form.append('scoring_mode', 'xgboost')

    const initRes = await fetch(`${API}/api/v1/review`, {
      method: 'POST',
      body: form,
    })
    const initData = await initRes.json()
    const taskId = initData.task_id
    if (!taskId) throw new Error('No task ID returned from server.')

    el.reviewStatus.textContent = 'Video uploaded. Analyzing computer vision features...'

    const pollTask = async () => {
      try {
        const statusRes = await fetch(`${API}/api/v1/review/status/${taskId}`)
        const statusData = await statusRes.json()
        const { status: taskStatus, error: taskError, result: taskResult } = statusData

        if (taskStatus === 'completed') {
          const segments = groupSegments(taskResult?.windows || [])
          state.reviewResult = { ...taskResult, segments, task_id: taskId }
          el.btnRunReview.disabled = false
          el.reviewStatus.style.display = 'none'
          displayReviewResults(state.reviewResult)
        } else if (taskStatus === 'failed') {
          el.reviewError.style.display = 'block'
          el.reviewError.textContent = taskError || 'Analysis failed in background.'
          el.reviewStatus.style.display = 'none'
          el.btnRunReview.disabled = false
        } else {
          const pct = statusData.progress || 0
          const msg = statusData.message || (pct > 0 ? `Analyzing video frames (${pct}%)...` : 'Processing video frames (running YOLO & optical flow)...')
          el.reviewStatus.textContent = msg
          setTimeout(pollTask, 1200)
        }
      } catch (err) {
        el.reviewError.style.display = 'block'
        el.reviewError.textContent = 'Error polling analysis status.'
        el.reviewStatus.style.display = 'none'
        el.btnRunReview.disabled = false
      }
    }

    setTimeout(pollTask, 1500)
  } catch (err) {
    el.reviewError.style.display = 'block'
    el.reviewError.textContent = err.message || 'Upload failed.'
    el.reviewStatus.style.display = 'none'
    el.btnRunReview.disabled = false
  }
}

function displayReviewResults(result) {
  el.reviewResultsArea.style.display = 'block'
  el.btnExportReport.disabled = false

  const videoUrl = URL.createObjectURL(state.reviewFile)
  el.reviewVideo.src = videoUrl

  const segments = result.segments || []
  if (segments.length) {
    state.selectedWindow = segments[0]
  }

  // Render Detailed Trip Report Copy
  const hasRed = segments.some(s => s.severity === 'red')
  el.reviewReportCopy.innerHTML = `
    <p><strong>Overall Journey Score:</strong> ${result.avg_batch_score?.toFixed(1) || 'N/A'}</p>
    <p><strong>Total Duration:</strong> ${formatClock(result.duration_sec)}</p>
    <p>
      ${result.window_count} extraction windows were evaluated.
      ${hasRed ? 'Critical drops are clustered around abrupt velocity shifts and proximity spikes.' : 'Session flow remained stable with smooth transitions across extraction windows.'}
    </p>
  `

  // Render Structured Timeline
  const modules = buildTimelineModules(segments)
  el.reviewTimelineModules.innerHTML = modules.map((m, idx) => {
    const isExpanded = idx === 0
    return `
      <div class="timeline-module ${isExpanded ? 'open' : ''}" data-mod-id="${m.id}">
        <button type="button" class="timeline-module-head ${isExpanded ? 'expanded' : ''}" data-mod-toggle="${m.id}">
          <span class="timeline-module-title">${m.title}</span>
          <span class="timeline-module-count">${m.lessons.length} lessons</span>
        </button>
        <div class="timeline-module-body ${isExpanded ? 'expanded' : ''}">
          ${m.lessons.map(l => `
            <div class="timeline-lesson">
              <span class="timeline-lesson-label">${l.label}</span>
              <button type="button" class="timeline-video-item" data-start="${l.segment.start_sec}">
                <span class="line-clamp-2">${l.title}</span>
                <div class="timeline-video-meta">
                  <span>${formatClock(l.segment.start_sec)} - ${formatClock(l.segment.end_sec)}</span>
                  <span class="severity-badge severity-${l.segment.severity}">${l.segment.severity}</span>
                </div>
              </button>
            </div>
          `).join('')}
        </div>
      </div>
    `
  }).join('')

  // Accordion toggle handlers
  el.reviewTimelineModules.querySelectorAll('[data-mod-toggle]').forEach(head => {
    head.onclick = () => {
      const parent = head.closest('.timeline-module')
      const body = parent.querySelector('.timeline-module-body')
      head.classList.toggle('expanded')
      body.classList.toggle('expanded')
      parent.classList.toggle('open')
    }
  })

  // Lesson seek handlers
  el.reviewTimelineModules.querySelectorAll('.timeline-video-item').forEach(item => {
    item.onclick = () => {
      el.reviewTimelineModules.querySelectorAll('.timeline-video-item').forEach(b => b.classList.remove('selected'))
      item.classList.add('selected')
      const start = Number(item.dataset.start) || 0
      el.reviewVideo.currentTime = start
      el.reviewVideo.play().catch(() => {})
      const found = segments.find(s => s.start_sec === start)
      if (found) {
        state.selectedWindow = found
        updateInsights()
      }
    }
  })

  // Render Trend Chart
  renderTrendChart(segments, el.reviewTrendChartContainer, 'Upload a clip to see score trend')
  updateMiniDashboard()
  updateInsights()
}

// ─── HISTORY & ANALYTICS WORKFLOW ─────────────────────────────────────────────

async function fetchHistoryData() {
  if (!state.token || state.offlineMode) return

  try {
    const [metricsRes, tripsRes] = await Promise.all([
      fetch(`${API}/api/v1/dashboard/metrics`, { headers: { Authorization: `Bearer ${state.token}` } }),
      fetch(`${API}/api/v1/trips/history`, { headers: { Authorization: `Bearer ${state.token}` } }),
    ])

    if (metricsRes.ok) {
      state.historyMetrics = await metricsRes.json()
      el.histMeanScore.textContent = state.historyMetrics.mean_eco_score ?? '—'
      el.histLowestScore.textContent = state.historyMetrics.lowest_eco_score ?? '—'
      el.histTotalTrips.textContent = state.historyMetrics.total_trips ?? '—'
      el.histTripsWeek.textContent = state.historyMetrics.trips_this_week ?? '—'
    }

    if (tripsRes.ok) {
      state.tripHistory = await tripsRes.json()
      renderTripsTable(state.tripHistory)
    }
  } catch (e) {
    console.error('Failed to load history data', e)
  }
}

function renderTripsTable(trips = []) {
  if (!trips.length) {
    el.tripsTableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align:center; padding: 24px 0; color: var(--c-white-46);">
          No saved trips found. Record drives to build history.
        </td>
      </tr>
    `
    return
  }

  el.tripsTableBody.innerHTML = trips.map(t => {
    const s = Math.round(t.final_score || 0)
    const cls = severityClass(s)
    const dt = t.created_at ? new Date(t.created_at).toLocaleString() : 'Recent'
    return `
      <tr>
        <td style="padding: 10px 0;">${dt}</td>
        <td style="padding: 10px 0;">
          <span class="severity-badge ${cls}">${s} / 100</span>
        </td>
        <td style="padding: 10px 0;">${formatClock(t.duration_sec || 0)}</td>
        <td style="padding: 10px 0; color: var(--c-white-72);">${(t.top_event || 'Smooth Flow').replaceAll('_', ' ')}</td>
        <td style="padding: 10px 0; text-align: right;">
          <button type="button" class="btn btn-ghost" data-trip-id="${t.session_id}" style="padding: 4px 10px; font-size: 11px;">
            Inspect
          </button>
        </td>
      </tr>
    `
  }).join('')

  el.tripsTableBody.querySelectorAll('[data-trip-id]').forEach(btn => {
    btn.onclick = () => loadTripDetails(btn.dataset.tripId)
  })
}

async function loadTripDetails(sessionId) {
  const trip = state.tripHistory.find(t => t.session_id === sessionId)
  if (!trip) return

  el.historyTripInspection.style.display = 'block'
  el.historyTripInspection.scrollIntoView({ behavior: 'smooth', block: 'start' })

  el.historyTripInspection.innerHTML = `
    <article class="card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <div>
          <div class="card-title">Inspecting Trip: ${sessionId}</div>
          <span style="font-size: 11px; color: var(--c-white-46);">Score: ${Math.round(trip.final_score)} · Duration: ${formatClock(trip.duration_sec)}</span>
        </div>
        <button type="button" class="btn btn-ghost" id="btn-close-trip-inspect" style="padding: 4px 8px;">✕ Close</button>
      </div>
      <div class="grid-2">
        <div id="trip-coach-container" class="card" style="padding: 16px;">
          <span style="font-size: 11px; text-transform: uppercase; color: var(--c-white-46);">AI Coaching Insights</span>
          <p style="margin-top: 8px; font-size: 12px; color: var(--c-white-72);">Loading AI coaching notes...</p>
        </div>
        <div id="trip-timeline-trend-container"></div>
      </div>
    </article>
  `

  document.getElementById('btn-close-trip-inspect').onclick = () => {
    el.historyTripInspection.style.display = 'none'
  }

  try {
    const res = await fetch(`${API}/api/v1/trips/${sessionId}/timeline`, {
      headers: { Authorization: `Bearer ${state.token}` },
    })
    const timeline = await res.json()
    renderTrendChart(timeline, document.getElementById('trip-timeline-trend-container'), 'No timeline available')

    if (timeline.length) {
      let totalVehicles = 0
      let totalPedestrians = 0
      timeline.forEach(f => {
        const feats = f.features || {}
        totalVehicles += Number(feats.vehicle_density || feats.vehicle_count || 0)
        if (Number(feats.pedestrian_ratio || feats.pedestrian_flag || 0) > 0) totalPedestrians += 1
      })
      const avgVehicles = totalVehicles / timeline.length

      const coachRes = await fetch(`${API}/api/v1/coach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${state.token}` },
        body: JSON.stringify({
          score: trip.final_score,
          features: { vehicle_density: avgVehicles, pedestrian_flag: totalPedestrians > 0 ? 1 : 0 },
          events: [trip.top_event],
          session_id: sessionId,
          is_summary: true,
        }),
      })
      const coachData = await coachRes.json()
      const tips = coachData.tips || []
      document.getElementById('trip-coach-container').innerHTML = `
        <span style="font-size: 11px; text-transform: uppercase; color: var(--c-white-46); letter-spacing: 0.05em;">AI Coaching Insights</span>
        <ul style="margin: 8px 0 0 16px; font-size: 12px; color: var(--c-white-92);">
          ${tips.map(t => `<li style="margin-bottom: 6px;">${t}</li>`).join('')}
        </ul>
      `
    }
  } catch (err) {
    console.error('Error fetching trip details', err)
  }
}

// ─── AUTHENTICATION WORKFLOW ──────────────────────────────────────────────────

function openAuthModal(isRegister = false) {
  state.showAuth = true
  state.authIsRegister = isRegister
  el.authModalTitle.textContent = isRegister ? 'Create Account' : 'Sign In'
  el.authSubmitBtn.textContent = isRegister ? 'Create Account' : 'Sign In'
  el.authToggleBtn.textContent = isRegister ? 'Already have an account? Sign in.' : 'Need an account? Register.'
  el.authError.style.display = 'none'
  el.authEmail.value = ''
  el.authPassword.value = ''
  el.authModal.style.display = 'flex'
}

function closeAuthModal() {
  state.showAuth = false
  el.authModal.style.display = 'none'
}

async function handleAuthSubmit(e) {
  e.preventDefault()
  el.authError.style.display = 'none'
  el.authSubmitBtn.disabled = true
  el.authSubmitBtn.textContent = 'Working...'

  const endpoint = state.authIsRegister ? `${API}/api/v1/auth/register` : `${API}/api/v1/auth/login`
  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: el.authEmail.value, password: el.authPassword.value }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data?.error || data?.detail || 'Authentication failed.')
    if (data.token) {
      state.token = data.token
      localStorage.setItem('driveiq_token', data.token)
      state.sessionSaveWarning = ''
      closeAuthModal()
      updateAuthDisplay()
      updateHealthBanners()
      fetchHistoryData()
    }
  } catch (err) {
    el.authError.style.display = 'block'
    el.authError.textContent = err.message || 'Authentication error.'
  } finally {
    el.authSubmitBtn.disabled = false
    el.authSubmitBtn.textContent = state.authIsRegister ? 'Create Account' : 'Sign In'
  }
}

// ─── ATTACH EVENT HANDLERS & INIT ─────────────────────────────────────────────

function initEventListeners() {
  // Navigation tabs
  el.navLive.onclick = () => {
    state.currentView = 'live'
    updateViewDisplay()
    el.liveView.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  el.navReview.onclick = () => {
    state.currentView = 'review'
    updateViewDisplay()
    el.reviewView.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
  el.navHistory.onclick = () => {
    state.currentView = 'history'
    updateViewDisplay()
    el.historyView.scrollIntoView({ behavior: 'smooth', block: 'start' })
    fetchHistoryData()
  }
  el.navInsights.onclick = () => {
    document.getElementById('insights').scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  // Mode buttons
  el.btnModeLive.onclick = () => { state.currentView = 'live'; updateViewDisplay() }
  el.btnModeReview.onclick = () => { state.currentView = 'review'; updateViewDisplay() }
  el.btnModeHistory.onclick = () => { state.currentView = 'history'; updateViewDisplay(); fetchHistoryData() }

  // Reconnect
  el.btnReconnect.onclick = () => {
    state.healthFailCount = 0
    state.offlineMode = false
    fetchHealth()
  }

  // Topbar Auth
  el.btnAuth.onclick = () => {
    if (state.token) {
      state.token = ''
      localStorage.removeItem('driveiq_token')
      state.sessionSaveWarning = ''
      updateAuthDisplay()
      updateHealthBanners()
    } else {
      openAuthModal(false)
    }
  }

  // Auth modal
  el.btnCloseAuth.onclick = closeAuthModal
  el.authToggleBtn.onclick = () => openAuthModal(!state.authIsRegister)
  el.authForm.onsubmit = handleAuthSubmit
  el.authModal.onclick = (e) => { if (e.target === el.authModal) closeAuthModal() }

  // Live stream controls
  el.btnScoringMode.onclick = () => {
    state.scoringMode = state.scoringMode === 'xgboost' ? 'event_rules' : 'xgboost'
    el.scoringModeText.textContent = state.scoringMode.replace('_', ' ')
  }

  el.liveFileInput.onchange = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    state.liveVideoFile = file
    if (state.liveVideoUrl) URL.revokeObjectURL(state.liveVideoUrl)
    state.liveVideoUrl = URL.createObjectURL(file)
    el.liveVideo.src = state.liveVideoUrl
    el.liveVideoPlaceholder.style.display = 'none'
    el.liveVideo.style.display = 'block'
    el.btnStartStream.style.display = 'inline-block'
    el.liveCompleteActions.style.display = 'none'

    // Reset stream telemetry
    state.livePoints = []
    state.liveEvents = []
    state.liveScore = 0
    state.liveFeatures = { ...ZERO_FEATURES }
    state.streamActive = false
    state.streamComplete = false
    state.clock = 0
    state.sessionStartedAt = Date.now()
    state.sessionId = `sess-${Math.random().toString(36).slice(2)}`
    updateLiveEventCounters()
    updateLiveTimelineList()
  }

  el.btnStartStream.onclick = () => {
    state.streamActive = true
    state.sessionStartedAt = Date.now()
    el.btnStartStream.style.display = 'none'
    el.liveVideo.play().catch(() => {})
    el.liveStatusMeta.textContent = 'Streaming active'
  }

  el.liveVideo.ontimeupdate = () => {
    const cur = el.liveVideo.currentTime || 0
    const dur = el.liveVideo.duration || 0
    state.livePlayback = { current: cur, duration: dur }
    el.liveClockMeta.textContent = `${formatClock(cur)} / ${formatClock(dur)}`
    const pct = dur > 0 ? (cur / dur) * 100 : 0
    el.liveProgressFill.style.width = `${pct}%`
  }

  el.liveVideo.onended = () => {
    state.streamActive = false
    state.streamComplete = true
    el.liveStatusMeta.textContent = 'Stream finished'
    el.liveCompleteActions.style.display = 'inline-flex'

    if (state.livePoints.length > 0) {
      const avg = state.livePoints.reduce((acc, p) => acc + p.score, 0) / state.livePoints.length
      state.liveScore = Math.round(avg)
      state.liveEvents.unshift({
        label: `Stream complete. Final score: ${Math.round(avg)}.`,
        type: 'green',
        timestamp_sec: state.clock,
      })
      updateLiveTimelineList()
      updateInsights()
    }
  }

  el.btnNewSession.onclick = () => {
    state.liveVideoFile = null
    if (state.liveVideoUrl) URL.revokeObjectURL(state.liveVideoUrl)
    state.liveVideoUrl = null
    el.liveVideo.src = ''
    el.liveVideo.style.display = 'none'
    el.liveVideoPlaceholder.style.display = 'block'
    el.liveFileInput.value = ''
    el.liveCompleteActions.style.display = 'none'
    el.btnStartStream.style.display = 'none'
    state.livePoints = []
    state.liveEvents = []
    state.liveScore = 0
    state.liveFeatures = { ...ZERO_FEATURES }
    state.streamActive = false
    state.streamComplete = false
    state.clock = 0
    state.sessionStartedAt = Date.now()
    state.sessionId = `sess-${Math.random().toString(36).slice(2)}`
    updateViewDisplay()
    renderTrendChart([], el.liveTrendChartContainer, 'Stream a video to generate live trend mapping')
  }

  // Review Controls
  el.reviewFileInput.onchange = (e) => {
    state.reviewFile = e.target.files?.[0] || null
    el.reviewError.style.display = 'none'
    el.reviewStatus.style.display = 'none'
  }

  el.btnRunReview.onclick = runReviewAnalysis

  el.btnExportReport.onclick = () => {
    if (state.reviewResult?.task_id) {
      window.open(`${API}/api/v1/review/report/${state.reviewResult.task_id}`, '_blank')
    } else {
      alert('No report available. Please complete an analysis first.')
    }
  }

  el.reviewVideo.ontimeupdate = () => {
    const cur = el.reviewVideo.currentTime || 0
    const dur = el.reviewVideo.duration || 0
    el.reviewClockMeta.textContent = `${formatClock(cur)} / ${formatClock(dur)}`
    const pct = dur > 0 ? (cur / dur) * 100 : 0
    el.reviewProgressFill.style.width = `${pct}%`
  }
}

// ─── INITIALIZATION ───────────────────────────────────────────────────────────

function init() {
  initEventListeners()
  updateAuthDisplay()
  updateViewDisplay()
  fetchHealth()

  // Initial charts
  renderTrendChart([], el.liveTrendChartContainer, 'Stream a video to generate live trend mapping')
  renderBrakingMeter(0, el.brakingMeterContainer)
  renderProximityHeatstrip(0, el.proximityHeatstripContainer)

  // Intervals
  setInterval(fetchHealth, HEALTH_POLL_MS)
  setInterval(() => {
    if (state.currentView === 'live') {
      fetchScore()
    }
  }, POLL_MS)
}

// Run on DOM load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init)
} else {
  init()
}
