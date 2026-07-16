const FLAG_LABEL = { 0: 'None', 1: 'Detected' }

function fmt(val, digits = 2) {
  const n = Number(val)
  return Number.isFinite(n) ? n.toFixed(digits) : '-'
}

function InlineSparkline({ data, maxProp, color }) {
  if (!data || data.length < 2) return null
  const max = maxProp || Math.max(...data, 1)
  const min = 0
  const range = max - min || 1
  const w = 40
  const h = 12
  const points = data.map((val, idx) => {
    const x = (idx / (data.length - 1)) * w
    const y = h - ((val - min) / range) * h
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ marginLeft: '8px', verticalAlign: 'middle', overflow: 'visible' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1" />
    </svg>
  )
}

export default function FeatureTable({ features, history = [] }) {
  const proxHist = history.map(p => Number(p.features?.proximity_score || 0))
  const brakeHist = history.map(p => Number(p.features?.braking_ratio || p.features?.braking_flag || 0))

  const rows = [
    { label: 'Pedestrian Detected', value: FLAG_LABEL[Number(features?.pedestrian_flag ?? 0)] ?? 'None',
      flag: Number(features?.pedestrian_flag ?? 0) === 1 },
    { label: 'Vehicles Density', value: fmt(features?.vehicle_density ?? 0, 1) },
    { label: 'Braking',          value: FLAG_LABEL[Number(features?.braking_flag ?? 0)] ?? 'None',
      flag: Number(features?.braking_flag ?? 0) === 1,
      sparkline: <InlineSparkline data={brakeHist} maxProp={1} color="#ff4d4d" /> },
    { label: 'Lane Change',      value: FLAG_LABEL[Number(features?.lane_change_flag ?? 0)] ?? 'None',
      flag: Number(features?.lane_change_flag ?? 0) === 1 },
    { label: 'Proximity',        value: fmt(features?.proximity_score ?? 0, 4),
      sparkline: <InlineSparkline data={proxHist} maxProp={0.3} color="#ffcc00" /> },
    { label: 'Mean Flow',        value: fmt(features?.mean_flow ?? 0, 4) },
    { label: 'Flow Variance',    value: fmt(features?.flow_variance ?? 0, 4) },
  ]

  return (
    <div className="card">
      <div className="card-title">Feature Snapshot</div>
      <table className="feat-table">
        <tbody>
          {rows.map(({ label, value, flag, sparkline }) => (
            <tr key={label}>
              <td>{label}</td>
              <td className={flag ? 'feature-flag' : ''} style={{ display: 'flex', alignItems: 'center' }}>
                {value} {sparkline}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
