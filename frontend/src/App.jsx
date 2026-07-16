import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import axios from 'axios'

import ScoreGauge from './components/ScoreGauge'
import CoachingPanel from './components/CoachingPanel'
import TrendChart from './components/TrendChart'

import FeatureTable from './components/FeatureTable'
import ReviewPanel from './components/ReviewPanel'
import LoginPanel from './components/LoginPanel'
import MainDashboard from './components/MainDashboard'
import MiniDashboard from './components/MiniDashboard'
import EventCounterPanel from './components/EventCounterPanel'
import ProximityHeatstrip from './components/ProximityHeatstrip'
import BrakingMeter from './components/BrakingMeter'
import SpeedProxyMiniChart from './components/SpeedProxyMiniChart'
import { Bar } from 'react-chartjs-2'
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js'

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend)

const API = ''   // empty = same origin (Vite proxy to Flask)
const POLL_MS = 2500
const HEALTH_POLL_MS = 15000
const BACKEND_FAIL_THRESHOLD = 3

// Simulate telemetry that changes over time for demo
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

  // Sky + road background
  const grad = ctx.createLinearGradient(0, 0, 0, 180)
  grad.addColorStop(0, '#60a5fa')
  grad.addColorStop(0.55, '#93c5fd')
  grad.addColorStop(0.56, '#334155')
  grad.addColorStop(1, '#0f172a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 320, 180)

  // Lane markers with subtle motion
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

  // Front vehicle proxy based on speed/proximity-like movement
  const speed = Number(telemetry?.speed ?? 50)
  const carW = Math.max(18, Math.min(48, 18 + speed * 0.2))
  const carH = Math.max(10, Math.min(28, 10 + speed * 0.11))
  const carX = 160 + Math.sin(t * 0.7) * 12 - carW / 2
  const carY = 120 - Math.sin(t * 0.45) * 6
  ctx.fillStyle = '#ef4444'
  ctx.fillRect(carX, carY, carW, carH)

  // Export as base64 payload body only (without data URL prefix)
  const dataUrl = canvas.toDataURL('image/jpeg', 0.75)
  return dataUrl.split(',')[1]
}

const ZERO_FEATURES = {
  rpm: 0,
  throttle_position: 0,
  braking_flag: 0,
  lane_change_flag: 0,
  proximity_score: 0,
  mean_flow: 0,
  flow_variance: 0,
}

function severityClass(score) {
  if (score === 'green' || score === 'yellow' || score === 'red') {
    return `severity-${score}`
  }
  if (score >= 75) return 'severity-green'
  if (score >= 50) return 'severity-yellow'
  return 'severity-red'
}

function severityLabel(score) {
  if (score >= 75) return 'green'
  if (score >= 50) return 'yellow'
  return 'red'
}

function formatClock(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins}:${String(secs).padStart(2, '0')}`
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

  // Deduplicate using Map
  const unique = new Map()
  insights.forEach(i => unique.set(i.label, i))
  return Array.from(unique.values())
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

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('driveiq_token') || '')
  const [showAuthDialog, setShowAuthDialog] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')

  const [currentView, setCurrentView] = useState('live')
  const isLiveMode = currentView === 'live'
  const [liveScore, setLiveScore] = useState(0)
  const [liveFeatures, setLiveFeatures] = useState({ ...ZERO_FEATURES })
  const [reviewResult, setReviewResult] = useState(null)
  const [selectedWindow, setSelectedWindow] = useState(null)
  const [healthState, setHealthState] = useState('checking')
  const [healthMessage, setHealthMessage] = useState('Checking backend health...')
  const [offlineMode, setOfflineMode] = useState(false)
  const [sessionSaveWarning, setSessionSaveWarning] = useState('')
  const [healthMeta, setHealthMeta] = useState({ schema_valid: false, core_models_loaded: false })
  const [livePoints, setLivePoints] = useState([])
  const [liveEvents, setLiveEvents] = useState([])
  const [liveVideoFile, setLiveVideoFile] = useState(null)
  const [streamActive, setStreamActive] = useState(false)
  const [streamComplete, setStreamComplete] = useState(false)
  const [scoringMode, setScoringMode] = useState('xgboost')
  const [livePlayback, setLivePlayback] = useState({ current: 0, duration: 0 })

  const liveVideoUrl = useMemo(() => {
    if (!liveVideoFile) return null
    return URL.createObjectURL(liveVideoFile)
  }, [liveVideoFile])

  const clockRef = useRef(0)
  const sessionIdRef = useRef(`sess-${Math.random().toString(36).slice(2)}`)
  const sessionStartedAtRef = useRef(Date.now())
  const prevFrameRef = useRef(null)
  const healthFailCountRef = useRef(0)
  const liveVideoRef = useRef(null)
  const streamCompleteRef = useRef(false)
  const cleanFrameCountRef = useRef(0)
  const lastPositiveTimeRef = useRef(0)

  // Cleanup object URLs
  useEffect(() => {
    return () => {
      if (liveVideoUrl) URL.revokeObjectURL(liveVideoUrl)
    }
  }, [liveVideoUrl])

  const fetchHealth = useCallback(async () => {
    if (offlineMode) {
      return
    }

    try {
      const { data } = await axios.get(`${API}/api/health`)
      healthFailCountRef.current = 0
      setOfflineMode(false)
      setHealthMeta({
        schema_valid: Boolean(data?.schema_valid),
        core_models_loaded: Boolean(data?.core_models_loaded),
      })
      const scoreReady = data?.score_ready === true
      const ready = data?.ready === true || scoreReady
      if (ready) {
        setHealthState('ready')
        setHealthMessage('Backend ready for review and scoring.')
      } else {
        setHealthState('degraded')
        const reasonParts = []
        if (data?.schema_error) reasonParts.push(data.schema_error)
        if (data?.coach_status === 'loading') reasonParts.push('Coach model warming up')
        if (data?.coach_error) reasonParts.push(`Coach error: ${data.coach_error}`)
        const reason = reasonParts.length ? ` ${reasonParts.join(' | ')}` : ''
        setHealthMessage(`Backend reachable but degraded.${reason}`.trim())
      }
    } catch {
      healthFailCountRef.current += 1
      if (healthFailCountRef.current >= BACKEND_FAIL_THRESHOLD) {
        setOfflineMode(true)
        setHealthState('degraded')
        setHealthMessage('Backend unavailable. UI is in offline placeholder mode.')
      } else {
        setHealthState('error')
        setHealthMessage('Cannot reach backend health endpoint.')
      }
    }
  }, [offlineMode])

  // /api/score polling (Live Mode only)
  const fetchScore = useCallback(async () => {
    if (!isLiveMode) {
      return
    }

    if (offlineMode || healthState === 'error') {
      return
    }

    // If stream already completed full pass, don't score again
    if (streamCompleteRef.current) {
      return
    }

    let frameB64 = null
    const v = liveVideoRef.current
    if (v && v.readyState >= 2 && !v.paused && !v.ended) {
      const canvas = document.createElement('canvas')
      canvas.width = 480
      canvas.height = 270
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(v, 0, 0, 480, 270)
      frameB64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
      clockRef.current = v.currentTime
    } else if (v) {
      // Video element exists but is paused/ended - stop scoring
      return
    } else if (streamActive) {
      // No video element at all, but stream active - use synthetic fallback
      clockRef.current += POLL_MS / 1000
      frameB64 = generateSyntheticFrameB64(clockRef.current, generateTelemetry(clockRef.current))
    } else {
      return // Not active
    }

    // When real video is playing, send neutral/empty telemetry so the backend
    // relies purely on CV-extracted features (not fake sine-wave acceleration).
    const hasRealVideo = liveVideoRef.current && liveVideoRef.current.readyState >= 2
    const telemetry = hasRealVideo
      ? { speed: 0, rpm: 0, throttle_position: 0, gear: 0, acceleration: 0, fuel_rate: 0 }
      : generateTelemetry(clockRef.current)
    const prevFrameB64 = prevFrameRef.current
    const tokenExpired = Boolean(token) && isJwtExpired(token)
    const authHeaders = {}
    if (token && !tokenExpired) {
      authHeaders.Authorization = `Bearer ${token}`
    } else if (token && tokenExpired) {
      setSessionSaveWarning('Session expired. Live scoring continues, but this drive is not being saved. Please log in again.')
    }

    try {
      const { data } = await axios.post(`${API}/api/score`, {
        telemetry,
        session_id: sessionIdRef.current,
        session_started_at: sessionStartedAtRef.current,
        frame_b64: frameB64,
        prev_frame_b64: prevFrameB64,
        scoring_mode: scoringMode,
      }, {
        headers: authHeaders
      })
      prevFrameRef.current = frameB64

      if (token && !tokenExpired) {
        if (data?.auth_failed) {
          setSessionSaveWarning('Authentication failed. Live scoring continues, but this drive is not being saved. Please log in again.')
        } else if (data?.session_saved === false) {
          setSessionSaveWarning('Live scoring is active, but session saving is currently unavailable.')
        } else if (data?.session_saved === true) {
          setSessionSaveWarning('')
        }
      }

      const s = Number(data.score ?? 0)
      setLiveScore(s)

      const featuresMerged = { ...data.features }
      setLiveFeatures(featuresMerged)

      const insights = getLiveInsights(featuresMerged, s)
      const severity = severityLabel(s)

      setLivePoints(prev => {
        const next = [...prev, { timestamp_sec: clockRef.current, score: s, severity }]
        if (next.length > 60) next.shift()
        return next
      })

      // Pin one consolidated event per timestamp to the live log
      const warnings = insights.filter((i) => i.type !== 'green')
      if (warnings.length > 0) {
        // Pick the worst severity and join all labels
        const worst = warnings.some(w => w.type === 'red') ? 'red' : 'yellow'
        const summary = warnings.map((w) => w.label).join(' | ')
        setLiveEvents(prev => {
          const entry = { label: summary, type: worst, timestamp_sec: clockRef.current, score: Math.round(s) }
          const next = [entry, ...prev]
          if (next.length > 30) next.length = 30
          return next
        })
        cleanFrameCountRef.current = 0  // reset clean streak
      } else {
        // No warnings — track consecutive clean frames for positive feedback
        cleanFrameCountRef.current += 1
        const now = Date.now()
        const secSinceLastPositive = (now - lastPositiveTimeRef.current) / 1000

        // Push a positive event every ~10 seconds of clean driving (score > 85)
        if (s > 85 && cleanFrameCountRef.current >= 4 && secSinceLastPositive >= 10) {
          const positiveMessages = [
            'Smooth driving maintained — excellent fuel efficiency.',
            'Good lane discipline and steady speed.',
            'Safe following distance — keep it up!',
            'Consistent eco-driving behavior detected.',
            'No infractions — optimal driving pattern.',
          ]
          const msg = positiveMessages[Math.floor(Math.random() * positiveMessages.length)]
          setLiveEvents(prev => {
            const entry = { label: msg, type: 'green', timestamp_sec: clockRef.current, score: Math.round(s) }
            const next = [entry, ...prev]
            if (next.length > 30) next.length = 30
            return next
          })
          lastPositiveTimeRef.current = now
          cleanFrameCountRef.current = 0
        }
      }

    } catch {
      setHealthState('degraded')
      setHealthMessage('Backend unstable. Retrying score stream with backoff...')
    }
  }, [healthState, offlineMode, isLiveMode, streamActive, token, scoringMode])

  const reconnectBackend = useCallback(async () => {
    healthFailCountRef.current = 0
    setOfflineMode(false)
    await fetchHealth()
  }, [fetchHealth])

  // Start polling
  useEffect(() => {
    fetchHealth()
    const healthId = setInterval(fetchHealth, HEALTH_POLL_MS)
    let scoreId = null
    if (isLiveMode) {
      fetchScore()
      scoreId = setInterval(fetchScore, POLL_MS)
    }
    return () => {
      if (scoreId) clearInterval(scoreId)
      clearInterval(healthId)
    }
  }, [fetchHealth, fetchScore, isLiveMode])

  const reviewPoints = useMemo(() => {
    if (isLiveMode) return livePoints
    return reviewResult?.segments || []
  }, [isLiveMode, livePoints, reviewResult])
  const displayedScore = isLiveMode
    ? liveScore
    : Number(selectedWindow?.avg_score ?? selectedWindow?.score ?? 0)
  const displayedFeatures = isLiveMode
    ? {
      pedestrian_flag: liveFeatures?.pedestrian_flag ?? 0,
      vehicle_density: liveFeatures?.vehicle_density ?? 0,
      braking_flag: liveFeatures?.braking_flag ?? 0,
      lane_change_flag: liveFeatures?.lane_change_flag ?? 0,
      proximity_score: liveFeatures?.proximity_score ?? 0,
      mean_flow: liveFeatures?.mean_flow ?? 0,
      flow_variance: liveFeatures?.flow_variance ?? 0,
    }
    : {
      pedestrian_flag: Number(selectedWindow?.pedestrian_ratio ?? selectedWindow?.pedestrian_flag ?? 0) > 0 ? 1 : 0,
      vehicle_density: Number(selectedWindow?.vehicle_density ?? selectedWindow?.vehicle_count ?? 0),
      braking_flag: Number(selectedWindow?.braking_flag_ratio ?? 0) > 0 ? 1 : 0,
      lane_change_flag: Number(selectedWindow?.lane_change_flag_ratio ?? 0) > 0 ? 1 : 0,
      proximity_score: Number(selectedWindow?.proximity_score_mean ?? 0),
      mean_flow: Number(selectedWindow?.mean_flow_mean ?? 0),
      flow_variance: Number(selectedWindow?.flow_variance ?? 0),
    }

  const selectedCoach = selectedWindow?.coach_note || 'Select a segment to view coaching note.'
  const selectedSeverity = selectedWindow?.severity || 'yellow'
  const healthClass = `health-banner health-${healthState}`
  const schemaOk = healthMeta.schema_valid
  const modelsOk = healthMeta.core_models_loaded
  const backendReady = schemaOk && modelsOk
  const activeNav = isLiveMode ? 'live' : 'review'
  const liveProgressPct = livePlayback.duration > 0
    ? Math.min(100, (livePlayback.current / livePlayback.duration) * 100)
    : 0
  const todayScore = Math.round(
    Number(isLiveMode ? liveScore : reviewResult?.avg_batch_score ?? displayedScore ?? 67),
  )
  const bestScore = Math.max(82, Math.round(Number(reviewResult?.avg_batch_score ?? 0)), Math.round(liveScore))
  const tripCount = reviewResult?.window_count ? Math.max(1, Math.ceil(reviewResult.window_count / 6)) : 4
  const fuelSaved = reviewResult?.avg_batch_score
    ? `${Math.max(0, (Number(reviewResult.avg_batch_score) - 50) / 18).toFixed(1)}L`
    : '2.3L'

  const activateReviewMode = () => {
    setCurrentView('review')
    setStreamActive(false)
  }

  const activateLiveMode = () => {
    setCurrentView('live')
    setLivePoints([])
    setLiveEvents([])
    setLivePlayback({ current: 0, duration: 0 })
    clockRef.current = 0
    sessionStartedAtRef.current = Date.now()
    prevFrameRef.current = null
    streamCompleteRef.current = false
    setStreamComplete(false)
  }

  const startNewSession = () => {
    setLiveVideoFile(null)
    setLivePoints([])
    setLiveEvents([])
    setLiveScore(0)
    setLiveFeatures({ ...ZERO_FEATURES })
    setStreamActive(false)
    setStreamComplete(false)
    setLivePlayback({ current: 0, duration: 0 })
    clockRef.current = 0
    sessionStartedAtRef.current = Date.now()
    prevFrameRef.current = null
    streamCompleteRef.current = false
    sessionIdRef.current = `sess-${Math.random().toString(36).slice(2)}`
  }

  const scrollToSection = (sectionId) => {
    const target = document.getElementById(sectionId)
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-mark">DQ</span>
          <div className="brand-copy">
            <span className="brand-name">DriveIQ</span>
            <span className="brand-sub">Coaching Console</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button className="nav-item" onClick={() => scrollToSection('overview')}>
            <span className="nav-icon">01</span>
            <span>Overview</span>
          </button>
          <button
            className={`nav-item ${activeNav === 'review' ? 'active' : ''}`}
            onClick={() => {
              activateReviewMode()
              scrollToSection('review')
            }}
          >
            <span className="nav-icon">02</span>
            <span>Post-Drive Review</span>
          </button>
          <button
            className={`nav-item ${activeNav === 'live' ? 'active' : ''}`}
            onClick={() => {
              activateLiveMode()
              scrollToSection('live')
            }}
          >
            <span className="nav-icon">03</span>
            <span>Live Stream</span>
          </button>
          <button className="nav-item" onClick={() => scrollToSection('insights')}>
            <span className="nav-icon">04</span>
            <span>Insights</span>
          </button>

        </nav>

        <div className="sidebar-footer">
          <div className="status-indicator">
            <span className={`status-dot ${backendReady ? 'ok' : 'bad'}`} />
            <span>{backendReady ? 'Backend Ready' : 'Backend Degraded'}</span>
          </div>
          <span className={`severity-badge ${backendReady ? 'severity-green' : 'severity-red'}`}>
            schema: {String(schemaOk)} | models: {String(modelsOk)}
          </span>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div className="topbar-left">
            <label className="search-shell" htmlFor="app-search">
              <input
                id="app-search"
                type="search"
                placeholder="Search modules, lessons, sessions"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </label>
          </div>

          <div className="topbar-right">
            <button type="button" className="icon-btn" aria-label="Notifications">N</button>
            <div className="profile-chip">
              <span className="profile-dot" />
              <div className="profile-meta">
                <span className="profile-title">{token ? 'Verified Driver' : 'Guest Driver'}</span>
                <span className="profile-sub">{token ? 'Session saving enabled' : 'Session saving disabled'}</span>
              </div>
            </div>
            {token ? (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  localStorage.removeItem('driveiq_token')
                  setToken('')
                  setSessionSaveWarning('')
                }}
              >
                Logout
              </button>
            ) : (
              <button type="button" className="btn btn-primary" onClick={() => setShowAuthDialog(true)}>
                Login
              </button>
            )}
          </div>
        </header>

        <div className={healthClass}>{healthMessage}</div>
        {sessionSaveWarning ? <div className="health-banner health-degraded">{sessionSaveWarning}</div> : null}
        {offlineMode ? (
          <div className="offline-controls">
            <button className="btn" onClick={reconnectBackend}>Retry Backend Connection</button>
          </div>
        ) : null}

        <main className="main-content">
          <section className="mode-toggle">
            <button
              type="button"
              className={`mode-btn ${currentView === 'live' ? 'active' : ''}`}
              onClick={activateLiveMode}
            >
              Real-Time Live Mode
            </button>
            <button
              type="button"
              className={`mode-btn ${currentView === 'history' ? 'active' : ''}`}
              onClick={() => setCurrentView('history')}
            >
              Trip History
            </button>
            <button
              type="button"
              className={`mode-btn ${currentView === 'review' ? 'active' : ''}`}
              onClick={activateReviewMode}
            >
              Post-Drive Full Analysis
            </button>
          </section>

          {currentView === 'history' && (
            <MainDashboard token={token} offlineMode={offlineMode} />
          )}

          {currentView !== 'history' && (
            <MiniDashboard isLiveMode={isLiveMode} liveScore={liveScore} reviewResult={reviewResult} />
          )}

          {currentView === 'review' && (
            <>
              <section className="panel-card" id="review">
                <div className="panel-head">
                  <div>
                    <h2 className="section-title">Trip Review</h2>
                    <p className="panel-subtitle">Upload a drive and navigate structured modules, lessons, and video moments.</p>
                  </div>
                  <button className="btn" onClick={() => window.alert('Report export coming soon')}>
                    Export Report
                  </button>
                </div>
                <ReviewPanel
                  onAnalysisComplete={setReviewResult}
                  onWindowSelect={setSelectedWindow}
                  selectedTimestampSec={selectedWindow?.start_sec ?? selectedWindow?.timestamp_sec}
                />
              </section>

              {reviewResult && (
                <section className="grid-2">
                  <article className="card">
                    <div className="card-title">Drive Segment Breakdown</div>
                    <div className="chart-frame">
                      <Bar
                        data={{
                          labels: reviewResult.segments?.map((s) => formatClock(s.start_sec)) || [],
                          datasets: [{
                            label: 'Score per Segment',
                            data: reviewResult.segments?.map((s) => s.avg_score) || [],
                            backgroundColor: reviewResult.segments?.map((s) => {
                              const cls = severityClass(s.severity || s.avg_score)
                              if (cls === 'severity-green') return 'rgba(234, 234, 234, 0.88)'
                              if (cls === 'severity-yellow') return 'rgba(234, 234, 234, 0.62)'
                              return 'rgba(234, 234, 234, 0.36)'
                            }),
                            borderRadius: 8,
                            borderSkipped: false,
                          }],
                        }}
                        options={{
                          maintainAspectRatio: false,
                          scales: {
                            x: {
                              grid: { display: false },
                              ticks: { color: 'rgba(234, 234, 234, 0.58)', font: { size: 11 } },
                              border: { color: 'transparent' },
                            },
                            y: {
                              suggestedMax: 100,
                              grid: { color: 'rgba(234, 234, 234, 0.08)' },
                              ticks: { color: 'rgba(234, 234, 234, 0.46)', font: { size: 10 } },
                              border: { color: 'transparent' },
                            },
                          },
                          plugins: { legend: { display: false } },
                        }}
                      />
                    </div>
                  </article>

                  <div className="stack-col">
                    <article className="card">
                      <div className="card-title">Detailed Trip Report</div>
                      <div className="report-copy">
                        <p><strong>Overall Journey Score:</strong> {reviewResult.avg_batch_score?.toFixed(1) || 'N/A'}</p>
                        <p><strong>Total Duration:</strong> {formatClock(reviewResult.duration_sec)}</p>
                        <p>
                          {reviewResult.window_count} extraction windows were evaluated.
                          {reviewResult.segments?.some((s) => s.severity === 'red')
                            ? ' Critical drops are clustered around abrupt velocity shifts and proximity spikes.'
                            : ' Session flow remained stable with smooth transitions across extraction windows.'}
                        </p>
                      </div>
                    </article>
                    <TrendChart points={reviewPoints} emptyMessage="Upload a clip to see score trend" />
                  </div>
                </section>
              )}
            </>
          )}

          {currentView === 'live' && (
            <>
              <section className="panel-card panel-live" id="live">
                <div className="panel-head">
                  <div>
                    <h2 className="section-title">Live Dynamic Streaming</h2>
                    <p className="panel-subtitle">Run a direct frame stream and monitor live timeline coaching output.</p>
                    <button
                      className="btn btn-ghost"
                      style={{ marginTop: '8px', border: '1px solid var(--c-white-08)', padding: '4px 12px', fontSize: '11px' }}
                      onClick={() => setScoringMode(m => m === 'xgboost' ? 'event_rules' : 'xgboost')}
                    >
                      Mode: <span style={{ color: 'var(--c-primary)', fontWeight: 'bold' }}>{scoringMode.replace('_', ' ')}</span>
                    </button>
                  </div>
                  <div className="file-input-wrap">
                    {!streamComplete && (
                      <input
                        type="file"
                        accept="video/mp4"
                        onChange={(e) => {
                          setLiveVideoFile(e.target.files?.[0] || null)
                          setLivePoints([])
                          setLiveEvents([])
                          setLiveScore(0)
                          setLiveFeatures({ ...ZERO_FEATURES })
                          setStreamActive(false)
                          setStreamComplete(false)
                          setLivePlayback({ current: 0, duration: 0 })
                          clockRef.current = 0
                          sessionStartedAtRef.current = Date.now()
                          prevFrameRef.current = null
                          streamCompleteRef.current = false
                          sessionIdRef.current = `sess-${Math.random().toString(36).slice(2)}`
                        }}
                      />
                    )}
                    {liveVideoUrl && !streamActive && !streamComplete ? (
                      <button
                        className="btn btn-primary"
                        onClick={() => {
                          sessionStartedAtRef.current = Date.now()
                          setStreamActive(true)
                          liveVideoRef.current?.play()
                        }}
                      >
                        Start Stream
                      </button>
                    ) : null}
                    {streamComplete && (
                      <>
                        <span className="severity-badge severity-green" style={{ padding: '6px 14px', fontSize: '12px' }}>✓ Stream Complete — Session Saved</span>
                        <button className="btn" onClick={startNewSession}>New Session</button>
                      </>
                    )}
                  </div>
                </div>

                <EventCounterPanel events={liveEvents} meanFlow={liveFeatures?.mean_flow || 0} />

                <div className="grid-2">
                  <article className="video-card">
                    <ProximityHeatstrip score={Number(displayedFeatures?.proximity_score || 0)} />
                    <div style={{ display: 'flex' }}>
                      <div className="video-card-media" style={{ flex: 1 }}>
                        {liveVideoUrl ? (
                          <video
                            ref={liveVideoRef}
                            src={liveVideoUrl}
                            controls={!streamComplete}
                            muted
                            onLoadedMetadata={(e) => {
                              const duration = Number(e.currentTarget.duration) || 0
                              setLivePlayback({ current: 0, duration })
                            }}
                            onTimeUpdate={(e) => {
                              const current = Number(e.currentTarget.currentTime) || 0
                              const duration = Number(e.currentTarget.duration) || 0
                              setLivePlayback({ current, duration })
                            }}
                            onPlay={(e) => {
                              // Block replay after stream is done
                              if (streamCompleteRef.current) {
                                e.currentTarget.pause()
                                return
                              }
                            }}
                            onEnded={() => {
                              setStreamActive(false)
                              streamCompleteRef.current = true
                              setStreamComplete(true)
                              // Pause and lock the video
                              if (liveVideoRef.current) {
                                liveVideoRef.current.pause()
                              }
                              if (livePoints.length > 0) {
                                const avg = livePoints.reduce((acc, p) => acc + p.score, 0) / livePoints.length
                                setLiveScore(Math.round(avg))
                                setLiveEvents((prev) => [
                                  { label: `Stream complete. Final score: ${Math.round(avg)}.`, type: 'green', timestamp_sec: clockRef.current },
                                  ...prev,
                                ])
                              }
                            }}
                            style={streamComplete ? { pointerEvents: 'none', opacity: 0.6 } : {}}
                          />
                        ) : (
                          <div className="video-placeholder">
                            Upload an MP4 file to begin live frame-by-frame analysis.
                          </div>
                        )}
                      </div>
                      <BrakingMeter ratio={Number(displayedFeatures?.braking_ratio || displayedFeatures?.braking_flag || 0)} />
                    </div>
                    <div className="video-card-body">
                      <h3 className="video-card-title line-clamp-2">
                        Real-Time Drive Session - Adaptive stream diagnostics
                      </h3>
                      <div className="video-card-meta">
                        <span>{streamComplete ? 'Stream finished' : streamActive ? 'Streaming active' : 'Awaiting stream start'}</span>
                        <span>{formatClock(livePlayback.current)} / {formatClock(livePlayback.duration)}</span>
                        <span>{liveEvents.length} events</span>
                      </div>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${liveProgressPct}%` }} />
                      </div>
                    </div>
                  </article>
                  <div className="stack-col">
                    <TrendChart points={reviewPoints} emptyMessage="Stream a video to generate live trend mapping" />
                    <SpeedProxyMiniChart history={livePoints.map(p => Number(p.features?.mean_flow || 0)).slice(-30)} />
                  </div>
                </div>

                <article className="card timeline-card" style={{ marginTop: '16px' }}>
                  <div className="card-title">Live Timeline</div>
                  <div className="timeline-module">
                    <div className="timeline-module-head static">
                      <span className="timeline-module-title">Module 01 - Live Coaching Events</span>
                      <span className="timeline-module-count">{liveEvents.length || 0} lessons</span>
                    </div>
                    <div className="timeline-module-body expanded" style={{ maxHeight: '400px', overflowY: 'auto', paddingRight: '4px' }}>
                      {liveEvents.length > 0 ? (
                        liveEvents.map((insight, idx) => {
                          const label = formatClock(insight.timestamp_sec)
                          return (
                            <div className="timeline-lesson" key={`${label}-${idx}`}>
                              <span className="timeline-lesson-label">Lesson {String(idx + 1).padStart(2, '0')}</span>
                              <button
                                type="button"
                                className="timeline-video-item"
                                onClick={() => {
                                  if (liveVideoRef.current) {
                                    liveVideoRef.current.currentTime = insight.timestamp_sec
                                    liveVideoRef.current.play()
                                  }
                                }}
                              >
                                <span className="line-clamp-2">{insight.label}</span>
                                <div className="timeline-video-meta">
                                  <span>{label}</span>
                                  {insight.score != null && <span style={{ fontSize: '11px', color: 'var(--c-white-72)', fontWeight: 600 }}>Score: {insight.score}</span>}
                                  <span className={`severity-badge severity-${insight.type}`}>{insight.type}</span>
                                </div>
                              </button>
                            </div>
                          )
                        })
                      ) : (
                        <div className="empty-state">
                          <p className="empty-state-text">
                            {streamActive ? 'Analyzing stream with no significant infractions yet.' : 'Start stream to generate timeline lessons.'}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              </section>
            </>
          )}

          <section className="score-feature-grid" id="insights">
            <ScoreGauge score={displayedScore} />
            <FeatureTable features={displayedFeatures} history={isLiveMode ? livePoints : []} />
          </section>

          <section>
            <CoachingPanel
              tips={[selectedCoach]}
              loading={false}
              message={isLiveMode ? 'Live mode active. Real-time insights appear above.' : 'Coaching for selected review segment.'}
              severity={selectedSeverity}
              source={selectedWindow?.score_source || 'review'}
              fallback={false}
              debugReason=""
              warning=""
              topIssue={selectedWindow?.dominant_issue || selectedWindow?.top_issue || ''}
            />
          </section>


        </main>
      </div>

      {showAuthDialog ? (
        <LoginPanel
          onClose={() => setShowAuthDialog(false)}
          onLogin={(t) => {
            localStorage.setItem('driveiq_token', t)
            setToken(t)
            setSessionSaveWarning('')
            setShowAuthDialog(false)
          }}
        />
      ) : null}
    </div>
  )
}
