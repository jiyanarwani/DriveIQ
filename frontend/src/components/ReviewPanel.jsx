import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

function formatTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatRange(start, end) {
  const s = formatTime(start)
  const e = formatTime(end)
  return s === e ? `${s} - ${formatTime(Number(start) + 2)}` : `${s} - ${e}`
}

function severityBadge(sev) {
  const map = { green: 'green', yellow: 'yellow', red: 'red' }
  return `severity-badge severity-${map[sev] || 'yellow'}`
}

function toNum(value, fallback = 0) {
  const n = Number(value)
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
  const worst = windows.reduce(
    (w, cur) => (toNum(cur?.score) < toNum(w?.score, Infinity) ? cur : w),
    windows[0],
  )
  const mean = (key, fb = 0) =>
    windows.reduce((acc, w) => acc + toNum(w?.[key], fb), 0) / Math.max(1, windows.length)

  return {
    start_sec: toNum(first?.timestamp_sec),
    end_sec: toNum(last?.timestamp_sec),
    avg_score: Number(avgScore.toFixed(2)),
    dominant_issue: dominantIssue,
    severity,
    coach_note: worst?.coach_note || 'Maintain smooth, consistent driving.',
    window_count: windows.length,
    timestamp_sec: toNum(first?.timestamp_sec),
    score: Number(avgScore.toFixed(2)),
    top_issue: dominantIssue,
    score_source: mostFrequent(windows, 'score_source', 'xgb'),
    braking_flag_ratio: mean('braking_flag_ratio', mean('braking_flag')),
    lane_change_flag_ratio: mean('lane_change_flag_ratio', mean('lane_change_flag')),
    proximity_score_mean: mean('proximity_score_mean', mean('proximity_score')),
    mean_flow_mean: mean('mean_flow_mean', mean('mean_flow')),
    flow_variance: mean('flow_variance'),
    _windows: windows,
  }
}

function buildTimelineModules(segments = []) {
  if (!segments.length) return []
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

export function groupSegments(windows = []) {
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

export default function ReviewPanel({ onAnalysisComplete, onWindowSelect, selectedTimestampSec }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [expandedModules, setExpandedModules] = useState({})
  const [previewPlayback, setPreviewPlayback] = useState({ current: 0, duration: 0 })

  const videoRef = useRef(null)

  const videoUrl = useMemo(() => {
    if (!file) return null
    return URL.createObjectURL(file)
  }, [file])

  const timelineModules = useMemo(() => buildTimelineModules(result?.segments || []), [result])
  const previewProgress = previewPlayback.duration > 0
    ? Math.min(100, (previewPlayback.current / previewPlayback.duration) * 100)
    : 0

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl)
  }, [videoUrl])

  useEffect(() => {
    if (!timelineModules.length) {
      setExpandedModules({})
      return
    }

    setExpandedModules((prev) => {
      const next = { ...prev }
      timelineModules.forEach((module, idx) => {
        if (typeof next[module.id] !== 'boolean') next[module.id] = idx === 0
      })
      return next
    })
  }, [timelineModules])

  const onFileChange = (e) => {
    const picked = e.target.files?.[0] || null
    setFile(picked)
    setResult(null)
    setError('')
    setExpandedModules({})
    setPreviewPlayback({ current: 0, duration: 0 })
    if (onAnalysisComplete) onAnalysisComplete(null)
  }

  const analyse = async () => {
    if (!file) {
      setError('Select an mp4 file first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const form = new FormData()
      form.append('video', file)
      const { data } = await axios.post('/api/review', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      const segments = groupSegments(data?.windows || [])
      const nextResult = { ...data, segments }
      setResult(nextResult)
      if (onAnalysisComplete) onAnalysisComplete(nextResult)
      if (onWindowSelect && segments.length) onWindowSelect(segments[0])
    } catch (e) {
      const msg = e?.response?.data?.message || e?.response?.data?.error || 'Analysis failed.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const seekTo = (windowItem) => {
    if (!videoRef.current) return
    const target = Number(windowItem?.start_sec ?? windowItem?.timestamp_sec) || 0
    videoRef.current.currentTime = target
    videoRef.current.play().catch(() => {})
    if (onWindowSelect) onWindowSelect(windowItem)
  }

  const toggleModule = (moduleId) => {
    setExpandedModules((prev) => ({ ...prev, [moduleId]: !prev[moduleId] }))
  }

  return (
    <div className="review-panel">
      <div className="file-input-wrap review-upload-row">
        <input type="file" accept="video/mp4" onChange={onFileChange} id="review-file-input" />
        <button
          className="btn btn-primary"
          onClick={analyse}
          disabled={loading || !file}
          id="review-analyse-btn"
        >
          {loading ? 'Analysing...' : 'Analyse'}
        </button>
        {loading ? <span className="text-mute">Processing extraction windows...</span> : null}
      </div>

      {error ? <div className="form-error">{error}</div> : null}

      {result ? (
        <>
          <div className="grid-2-asym review-results-grid">
            <article className="video-card">
              <div className="video-card-media">
                {videoUrl ? (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    onLoadedMetadata={(e) => {
                      const duration = Number(e.currentTarget.duration) || 0
                      setPreviewPlayback({ current: 0, duration })
                    }}
                    onTimeUpdate={(e) => {
                      const current = Number(e.currentTarget.currentTime) || 0
                      const duration = Number(e.currentTarget.duration) || 0
                      setPreviewPlayback({ current, duration })
                    }}
                  />
                ) : (
                  <div className="video-placeholder">Video preview unavailable.</div>
                )}
              </div>
              <div className="video-card-body">
                <h3 className="video-card-title line-clamp-2">
                  {file?.name || 'Uploaded Drive Session'}
                </h3>
                <div className="video-card-meta">
                  <span>{formatTime(result.duration_sec)} duration</span>
                  <span>{result.window_count} windows</span>
                  <span>{formatTime(previewPlayback.current)} watched</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${previewProgress}%` }} />
                </div>
              </div>
            </article>

            <article className="card timeline-card">
              <div className="card-title">Structured Timeline</div>
              <div className="timeline-collection" style={{ overflowY: 'auto', maxHeight: '400px', paddingRight: '4px' }}>
                {timelineModules.map((module) => {
                  const expanded = Boolean(expandedModules[module.id])
                  return (
                    <div className={`timeline-module ${expanded ? 'open' : ''}`} key={module.id}>
                      <button
                        type="button"
                        className={`timeline-module-head ${expanded ? 'expanded' : ''}`}
                        onClick={() => toggleModule(module.id)}
                      >
                        <span className="timeline-module-title">{module.title}</span>
                        <span className="timeline-module-count">{module.lessons.length} lessons</span>
                      </button>

                      <div
                        className={`timeline-module-body ${expanded ? 'expanded' : ''}`}
                        style={{ maxHeight: expanded ? `${module.lessons.length * 144 + 32}px` : '0px' }}
                      >
                        {module.lessons.map((lesson) => {
                          const seg = lesson.segment
                          const isSelected = Number(selectedTimestampSec) === Number(seg.start_sec)
                          return (
                            <div className="timeline-lesson" key={lesson.id}>
                              <span className="timeline-lesson-label">{lesson.label}</span>
                              <button
                                type="button"
                                className={`timeline-video-item ${isSelected ? 'selected' : ''}`}
                                onClick={() => seekTo(seg)}
                              >
                                <span className="line-clamp-2">{lesson.title}</span>
                                <div className="timeline-video-meta">
                                  <span>{formatRange(seg.start_sec, seg.end_sec)} | {lesson.duration}s</span>
                                  <span>Score: {seg.score}</span>
                                  <span className={severityBadge(seg.severity)}>{seg.severity}</span>
                                </div>
                              </button>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </article>
          </div>

          {/* AI Session Summary Card */}
          {result?.session_summary?.error && !result?.session_summary?.summary ? (
            <article className="card" style={{ marginTop: '16px', padding: '16px' }}>
              <span style={{ color: 'var(--c-white-46)', fontSize: '12px' }}>AI summary unavailable</span>
            </article>
          ) : null}

          {result?.session_summary?.summary ? (() => {
            const ss = result.session_summary.summary
            const ratingColorMap = {
              'Excellent': 'var(--c-green-bright)',
              'Good': 'var(--c-primary)',
              'Needs Improvement': 'var(--c-yellow-bright)',
              'Poor': 'var(--c-red-bright)',
            }
            const ratingColor = ratingColorMap[ss.overall_rating] || 'var(--c-white-46)'

            return (
              <article className="card" style={{ marginTop: '16px', padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <div className="card-title" style={{ margin: 0 }}>AI Session Summary</div>
                  <span style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    background: 'var(--c-white-08)',
                    color: ratingColor,
                  }}>
                    {ss.overall_rating}
                  </span>
                </div>

                {ss.summary_paragraph ? (
                  <p style={{ color: 'var(--c-white-72)', fontSize: '13px', lineHeight: '1.6', marginBottom: '16px' }}>
                    {ss.summary_paragraph}
                  </p>
                ) : null}

                {ss.what_went_well?.length ? (
                  <div style={{ marginBottom: '12px' }}>
                    <div style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>What went well</div>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {ss.what_went_well.map((item, i) => (
                        <li key={i} style={{ fontSize: '12px', color: 'var(--c-white-72)', padding: '3px 0' }}>
                          <span style={{ color: 'var(--c-green-bright)', marginRight: '6px' }}>✓</span>{item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {ss.areas_to_improve?.length ? (
                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>Areas to improve</div>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {ss.areas_to_improve.map((item, i) => (
                        <li key={i} style={{ fontSize: '12px', color: 'var(--c-white-72)', padding: '3px 0' }}>
                          <span style={{ color: 'var(--c-yellow-bright)', marginRight: '6px' }}>⚠</span>{item}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </article>
            )
          })() : null}
        </>
      ) : null}
    </div>
  )
}
