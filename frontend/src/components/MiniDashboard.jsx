function formatClock(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins}:${String(secs).padStart(2, '0')}`
}

export default function MiniDashboard({ isLiveMode, liveScore, reviewResult }) {
  // Determine metrics based on mode
  let avgScore = 0
  let duration = '0:00'
  let windows = 0
  let fuelSaved = '0.0L'

  if (isLiveMode) {
    avgScore = Math.round(liveScore || 0)
    duration = 'Live'
    windows = '—'
    fuelSaved = '—'
  } else if (reviewResult) {
    avgScore = Math.round(reviewResult.avg_batch_score || 0)
    duration = formatClock(reviewResult.duration_sec || 0)
    windows = reviewResult.window_count || 0
    fuelSaved = `${Math.max(0, (Number(reviewResult.avg_batch_score || 0) - 50) / 18).toFixed(1)}L`
  } else {
    // No data yet
    avgScore = '—'
    duration = '—'
    windows = '—'
    fuelSaved = '—'
  }

  return (
    <section className="stat-strip mt-3" id="mini-dashboard">
      <article className="card stat-card">
        <span className="card-title">Current Trip Score</span>
        <strong className="card-value">{avgScore}</strong>
        <span className="card-sub">{isLiveMode ? 'Live streaming' : 'Selected video analysis'}</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Trip Duration</span>
        <strong className="card-value">{duration}</strong>
        <span className="card-sub">Time analyzed</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Extraction Windows</span>
        <strong className="card-value">{windows}</strong>
        <span className="card-sub">Analyzed segments</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Estimated Fuel Saved</span>
        <strong className="card-value">{fuelSaved}</strong>
        <span className="card-sub">For this trip</span>
      </article>
    </section>
  )
}
