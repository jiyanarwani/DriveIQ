import React, { useEffect, useState } from 'react'

export default function BrakingMeter({ ratio = 0 }) {
  const [flash, setFlash] = useState(false)

  useEffect(() => {
    if (ratio > 0.3) {
      setFlash(true)
      const t = setTimeout(() => setFlash(false), 500)
      return () => clearTimeout(t)
    }
  }, [ratio])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', marginLeft: '12px', width: '20px' }}>
      <span style={{ fontSize: '9px', color: 'var(--c-white-46)', writingMode: 'vertical-rl', transform: 'rotate(180deg)', marginBottom: '8px' }}>Brake Force</span>
      <div style={{ 
        flex: 1, 
        width: '8px', 
        background: 'rgba(255, 255, 255, 0.05)', 
        borderRadius: '4px', 
        display: 'flex', 
        alignItems: 'flex-end',
        overflow: 'hidden',
        boxShadow: flash ? '0 0 10px rgba(255, 77, 77, 0.8)' : 'none',
        transition: 'box-shadow 0.2s'
      }}>
        <div style={{
          width: '100%',
          height: `${Math.min(100, ratio * 200)}%`, // amplify to make it visible
          background: flash ? '#ff4d4d' : 'var(--c-white-72)',
          transition: 'height 0.1s linear, background 0.2s'
        }} />
      </div>
    </div>
  )
}
