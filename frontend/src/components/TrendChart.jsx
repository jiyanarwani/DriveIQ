import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
} from 'chart.js'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip)

// Inline reference lines plugin - avoids re-registration issues
const referenceLinesPlugin = {
  id: 'diq_referenceLines',
  afterDraw(chart, _args, opts) {
    const yScale = chart.scales?.y
    const area = chart.chartArea
    if (!yScale || !area) return
    const { ctx } = chart
    ctx.save();
    (opts?.lines || []).forEach((line) => {
      const y = yScale.getPixelForValue(line.value)
      ctx.strokeStyle = line.color
      ctx.setLineDash([4, 6])
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.4
      ctx.beginPath()
      ctx.moveTo(area.left, y)
      ctx.lineTo(area.right, y)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.globalAlpha = 0.5
      ctx.fillStyle = line.color
      ctx.font = '10px Inter, sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'bottom'
      ctx.fillText(line.label, area.right - 4, y - 3)
    })
    ctx.restore()
  },
}

ChartJS.register(referenceLinesPlugin)

function formatTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function pointColor(severity) {
  if (severity === 'green') return 'rgba(234, 234, 234, 0.9)'
  if (severity === 'yellow') return 'rgba(234, 234, 234, 0.66)'
  return 'rgba(234, 234, 234, 0.44)'
}

export default function TrendChart({ points = [], emptyMessage = 'No data yet' }) {
  if (!points.length) {
    return (
      <div className="card">
        <div className="card-title">Score Trend</div>
        <div className="trend-empty">{emptyMessage}</div>
      </div>
    )
  }

  const labels = points.map((p) => formatTime(p.start_sec ?? p.timestamp_sec))

  const data = {
    labels,
    datasets: [{
      label: 'Eco Score',
      data: points.map((p) => Number(p.avg_score ?? p.score ?? 0)),
      borderColor: 'rgba(234, 234, 234, 0.28)',
      backgroundColor: 'rgba(234, 234, 234, 0.06)',
      fill: true,
      tension: 0.45,
      pointRadius: 3,
      pointBackgroundColor: points.map((p) => pointColor(p.severity)),
      pointBorderColor: points.map((p) => pointColor(p.severity)),
      pointHoverRadius: 5,
      borderWidth: 1.5,
    }]
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0,
        max: 100,
        grid: { color: 'rgba(234, 234, 234, 0.06)' },
        ticks: { color: 'rgba(234, 234, 234, 0.48)', font: { size: 10, family: 'Inter' }, stepSize: 25 },
        border: { color: 'transparent' },
      },
      x: {
        grid: { display: false },
        ticks: { color: 'rgba(234, 234, 234, 0.48)', font: { size: 10, family: 'Inter' }, maxTicksLimit: 8 },
        border: { color: 'transparent' },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: '#1a1a1a',
        borderColor: '#2a2a2a',
        borderWidth: 1,
        titleColor: 'rgba(234, 234, 234, 0.7)',
        bodyColor: '#eaeaea',
        titleFont: { size: 10, family: 'Inter' },
        bodyFont: { size: 12, family: 'Inter', weight: '600' },
        padding: 10,
      },
      diq_referenceLines: {
        lines: [
          { value: 75, label: 'Target', color: 'rgba(234, 234, 234, 0.52)' },
          { value: 50, label: 'Baseline', color: 'rgba(234, 234, 234, 0.35)' },
        ],
      },
    },
    animation: { duration: 250 },
  }

  return (
    <div className="card">
      <div className="card-title">Score Trend</div>
      <div className="trend-wrap">
        <Line data={data} options={options} />
      </div>
    </div>
  )
}
