import { Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip } from 'chart.js'

ChartJS.register(ArcElement, Tooltip)

function scoreColor(s) {
  if (s >= 75) return 'rgba(234, 234, 234, 0.9)'
  if (s >= 50) return 'rgba(234, 234, 234, 0.66)'
  return 'rgba(234, 234, 234, 0.44)'
}

function scoreLabel(s) {
  if (s >= 75) return 'Smooth'
  if (s >= 50) return 'Fair'
  return 'Rough'
}

export default function ScoreGauge({ score }) {
  const s = Math.round(score ?? 0)
  const color = scoreColor(s)

  const data = {
    datasets: [{
      data: [s, 100 - s],
      backgroundColor: [color, 'rgba(234, 234, 234, 0.08)'],
      borderWidth: 0,
      borderRadius: 10,
      circumference: 240,
      rotation: -120,
    }]
  }

  const options = {
    cutout: '82%',
    plugins: { tooltip: { enabled: false } },
    animation: { animateRotate: true, duration: 500 },
    responsive: true,
    maintainAspectRatio: false,
  }

  return (
    <div className="card">
      <div className="card-title">Drive Score</div>
      <div className="gauge-wrap">
        <Doughnut data={data} options={options} />
        <div className="gauge-center">
          <span className="gauge-score" style={{ color }}>{s}</span>
          <span className="gauge-sub">/ 100</span>
          <span className="gauge-label">{scoreLabel(s)}</span>
        </div>
      </div>
    </div>
  )
}
