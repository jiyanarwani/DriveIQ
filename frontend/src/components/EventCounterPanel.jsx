import React from 'react'

export default function EventCounterPanel({ events = [], meanFlow = 0 }) {
  const counts = {
    braking: events.filter(e => e.label.includes('braking')).length,
    tailgating: events.filter(e => e.label.includes('tailgating')).length,
    speed: events.filter(e => e.label.includes('velocity')).length,
  }

  // Determine motion state pill
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

  } else if (meanFlow <= 0.2) {
    motionLabel = 'Idle'
    motionColor = 'var(--c-white-46)'
  }

  return (
    <div className="event-counter-panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', marginBottom: '16px' }}>
      <div className="card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Motion State</span>
        <strong style={{ fontSize: '18px', color: motionColor, fontWeight: 700, textTransform: 'uppercase' }}>{motionLabel}</strong>
      </div>
      <div className="card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Hard Braking</span>
        <strong style={{ fontSize: '18px', color: 'var(--c-white-92)', fontWeight: 600 }}>{counts.braking}</strong>
      </div>
      <div className="card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Tailgating</span>
        <strong style={{ fontSize: '18px', color: 'var(--c-white-92)', fontWeight: 600 }}>{counts.tailgating}</strong>
      </div>
      <div className="card" style={{ padding: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <span style={{ fontSize: '11px', color: 'var(--c-white-46)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Erratic Speed</span>
        <strong style={{ fontSize: '18px', color: 'var(--c-white-92)', fontWeight: 600 }}>{counts.speed}</strong>
      </div>
    </div>
  )
}
