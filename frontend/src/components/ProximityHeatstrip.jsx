import React from 'react'

export default function ProximityHeatstrip({ score = 0 }) {
  // score is 0.0 to 1.0 roughly. >0.15 is red, >0.05 is yellow
  let bg = '#4caf50'
  let label = 'Safe Distance'
  if (score > 0.15) {
    bg = '#ff4d4d'
    label = 'Tailgating Risk'
  } else if (score > 0.05) {
    bg = '#ffcc00'
    label = 'Following Closely'
  }

  // Create a gradient that is mostly the color but with some intensity
  return (
    <div style={{ width: '100%', marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--c-white-46)' }}>Proximity Radar</span>
        <span style={{ fontSize: '11px', color: bg, fontWeight: 600 }}>{label} ({(score * 100).toFixed(0)}%)</span>
      </div>
      <div style={{ width: '100%', height: '8px', borderRadius: '4px', background: 'var(--c-white-08)', overflow: 'hidden' }}>
        <div 
          style={{ 
            height: '100%', 
            width: '100%', 
            background: bg, 
            opacity: Math.min(1, 0.4 + (score * 2)),
            transition: 'all 0.5s ease-out' 
          }} 
        />
      </div>
    </div>
  )
}
