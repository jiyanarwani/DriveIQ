export default function CoachingPanel({ tips, loading, message, severity, source, topIssue }) {
  const sev = ['green', 'yellow', 'red'].includes(severity) ? severity : 'yellow'
  const badgeCls = `severity-badge severity-${sev}`
  const sevLabel = { green: 'Good', yellow: 'Fair', red: 'Poor' }[sev]
  const issueLabel = topIssue ? String(topIssue).replaceAll('_', ' ') : null

  return (
    <div className="card">
      <div className="coach-header">
        <div className="card-title coach-title">Coaching</div>
        <div className="coach-badges">
          {issueLabel && (
            <span className="severity-badge severity-neutral">{issueLabel}</span>
          )}
          <span className={badgeCls}>{sevLabel}</span>
        </div>
      </div>

      {source && (
        <div className="coach-source mt-1 mb-2">
          Source: {source}
        </div>
      )}

      {message && (
        <p className="coach-message">{message}</p>
      )}

      {loading ? (
        <p className="text-mute coach-loading">
          Generating tips...
        </p>
      ) : (
        <ul className="tip-list tip-list-plain">
          {(tips || []).map((tip, i) => (
            <li key={i} className="tip-item">
              <span className="tip-num">{i + 1}</span>
              <span className="tip-copy">{tip}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
