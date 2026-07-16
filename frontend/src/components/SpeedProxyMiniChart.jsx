import React from 'react'

export default function SpeedProxyMiniChart({ history = [] }) {
  // history is an array of mean_flow values
  const maxFlow = Math.max(10, ...history)
  const minFlow = Math.min(0, ...history)
  const range = maxFlow - minFlow || 1
  
  // Create an SVG path
  // SVG viewBox is 0 0 100 30
  // width = 100, height = 30
  const w = 100
  const h = 30
  
  const points = history.map((val, idx) => {
    const x = (idx / Math.max(1, history.length - 1)) * w
    const y = h - ((val - minFlow) / range) * h
    return `${x},${y}`
  }).join(' ')

  return (
    <div style={{ marginTop: '16px', background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-white-46)', letterSpacing: '0.05em' }}>Speed Proxy Trend</span>
        <span style={{ fontSize: '11px', color: 'var(--c-white-72)' }}>Last 30s</span>
      </div>
      <svg width="100%" height="40" viewBox="0 0 100 30" preserveAspectRatio="none" style={{ overflow: 'visible' }}>
        <polyline
          points={points}
          fill="none"
          stroke="rgba(234, 234, 234, 0.6)"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Fill area underneath */}
        <polygon
          points={`0,30 ${points} 100,30`}
          fill="rgba(234, 234, 234, 0.05)"
        />
      </svg>
    </div>
  )
}
