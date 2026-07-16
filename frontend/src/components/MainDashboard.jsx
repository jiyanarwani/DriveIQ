import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import TrendChart from './TrendChart'

export default function MainDashboard({ token, offlineMode }) {
  const [metrics, setMetrics] = useState({
    mean_eco_score: 0,
    lowest_eco_score: 0,
    total_trips: 0,
    trips_this_week: 0
  })
  
  const [tripHistory, setTripHistory] = useState([])
  const [selectedTrip, setSelectedTrip] = useState(null)
  const [selectedTripTimeline, setSelectedTripTimeline] = useState([])
  const [selectedTripCoach, setSelectedTripCoach] = useState(null)
  const tripDetailsRef = useRef(null)

  useEffect(() => {
    if (!token || offlineMode) return

    axios.get('/api/dashboard/metrics', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setMetrics(res.data)
    })
    .catch(err => {
      console.error('Failed to fetch dashboard metrics', err)
    })

    axios.get('/api/trips/history', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setTripHistory(res.data)
    })
    .catch(err => {
      console.error('Failed to fetch trip history', err)
    })
  }, [token, offlineMode])

  // If no token, show placeholder metrics
  const displayMetrics = token ? metrics : {
    mean_eco_score: '—',
    lowest_eco_score: '—',
    total_trips: '—',
    trips_this_week: '—'
  }

  const loadTripDetails = (sessionId) => {
    const trip = tripHistory.find(t => t.session_id === sessionId)
    if (trip) {
      setSelectedTrip(trip)
      setSelectedTripTimeline([])
      setSelectedTripCoach(null)

      // Scroll to the analytics section after a tick so it renders first
      setTimeout(() => {
        tripDetailsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }, 100)
      
      axios.get(`/api/trips/${sessionId}/timeline`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then(res => {
        const timeline = res.data
        setSelectedTripTimeline(timeline)
        
        if (timeline.length > 0) {
          let totalVehicles = 0
          let totalPedestrians = 0
          
          timeline.forEach(f => {
            const feats = f.features || {}
            totalVehicles += Number(feats.vehicle_density || feats.vehicle_count || 0)
            if (Number(feats.pedestrian_ratio || feats.pedestrian_flag || 0) > 0) {
              totalPedestrians += 1
            }
          })
          
          const avgVehicles = totalVehicles / timeline.length
          const aggFeatures = {
            vehicle_density: avgVehicles,
            pedestrian_flag: totalPedestrians > 0 ? 1 : 0
          }
          
          axios.post('/api/coach', {
            score: trip.final_score,
            features: aggFeatures,
            events: [trip.top_event],
            session_id: sessionId,
            is_summary: true
          }, {
            headers: { Authorization: `Bearer ${token}` }
          })
          .then(coachRes => {
            const tips = coachRes.data.tips || []
            setSelectedTripCoach({
              tips: tips,
              pedestrians: totalPedestrians,
              avgVehicles: avgVehicles
            })
          })
          .catch(err => console.error('Coach fetch error', err))
        }
      })
      .catch(err => {
        console.error('Failed to fetch trip timeline', err)
      })
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', marginTop: '24px' }}>
      
      {/* 1. Trip History Card */}
      {token && !offlineMode && (
        <section className="card">
          <div className="card-title">Trip History</div>
          <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: '320px', marginTop: '16px' }}>
            <table className="feat-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Date</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Score</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Top Event</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Duration (frames)</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tripHistory.map(trip => (
                  <tr key={trip.session_id} style={selectedTrip?.session_id === trip.session_id ? { backgroundColor: 'var(--c-white-04)' } : {}}>
                    <td>{new Date(trip.date).toLocaleDateString()}</td>
                    <td>{typeof trip.final_score === 'number' ? trip.final_score.toFixed(1) : trip.final_score}</td>
                    <td style={{ textTransform: 'capitalize' }}>{trip.top_event.replace(/_/g, ' ')}</td>
                    <td>{trip.frame_count}</td>
                    <td>
                      <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => loadTripDetails(trip.session_id)}>
                        {selectedTrip?.session_id === trip.session_id ? 'Viewing' : 'View'}
                      </button>
                    </td>
                  </tr>
                ))}
                {tripHistory.length === 0 && (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', padding: '16px', color: 'var(--c-white-46)' }}>No trips recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* 2. Analytics Section (Selected Trip Details + Account Metrics) */}
      <div id="analytics-section" ref={tripDetailsRef} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {/* Selected Trip Details (Inline Analytics) */}
        {selectedTrip && (
          <section className="card">
            <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div className="card-title" style={{ margin: 0 }}>Trip Analytics: {selectedTrip.session_id}</div>
              <button className="btn btn-ghost" onClick={() => setSelectedTrip(null)}>&times; Close</button>
            </div>
            
            <div className="grid-2" style={{ gap: '24px' }}>
              <div className="stack-col">
                <div className="form-group" style={{ gap: '12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                    <span className="label" style={{ marginBottom: 0 }}>Date</span>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>{new Date(selectedTrip.date).toLocaleString()}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                    <span className="label" style={{ marginBottom: 0 }}>Final Score</span>
                    <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--c-white-92)' }}>{typeof selectedTrip.final_score === 'number' ? selectedTrip.final_score.toFixed(1) : selectedTrip.final_score}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                    <span className="label" style={{ marginBottom: 0 }}>Duration</span>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>{selectedTrip.frame_count} frames</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span className="label" style={{ marginBottom: 0 }}>Dominant Event</span>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-72)', textTransform: 'capitalize' }}>{selectedTrip.top_event.replace(/_/g, ' ')}</span>
                  </div>
                </div>

                {selectedTripCoach && (
                  <div style={{ marginTop: '24px', borderTop: '1px solid var(--c-white-08)', paddingTop: '16px' }}>
                    <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-white-46)', marginBottom: '12px', letterSpacing: '0.06em' }}>CV Feature Snapshot</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>Pedestrian Frames Detected</span>
                      <span style={{ fontSize: '12px', color: 'var(--c-white-92)' }}>{selectedTripCoach.pedestrians}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                      <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>Avg Vehicle Density</span>
                      <span style={{ fontSize: '12px', color: 'var(--c-white-92)' }}>{selectedTripCoach.avgVehicles.toFixed(1)}</span>
                    </div>

                    <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-white-46)', marginBottom: '12px', letterSpacing: '0.06em' }}>AI Coaching Insights</div>
                    <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '12px', color: 'var(--c-white-92)' }}>
                      {selectedTripCoach.tips.map((tip, idx) => (
                        <li key={idx} style={{ marginBottom: '6px' }}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="stack-col">
                <div className="card-title" style={{ fontSize: '14px', marginBottom: '12px' }}>Score Trend Timeline</div>
                {selectedTripTimeline.length > 0 ? (
                  <TrendChart points={selectedTripTimeline} emptyMessage="No timeline data available" />
                ) : (
                  <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--c-white-04)', borderRadius: '8px', color: 'var(--c-white-46)', fontSize: '12px' }}>
                    Loading timeline graph...
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* Account Level Metrics */}
        <section className="stat-strip" id="main-dashboard">
          <article className="card stat-card">
            <span className="card-title">Mean Eco Score</span>
            <strong className="card-value">{displayMetrics.mean_eco_score}</strong>
            <span className="card-sub">Overall account average</span>
          </article>
          <article className="card stat-card">
            <span className="card-title">Lowest Score</span>
            <strong className="card-value">{displayMetrics.lowest_eco_score}</strong>
            <span className="card-sub">Lowest recorded session</span>
          </article>
          <article className="card stat-card">
            <span className="card-title">Total Trips</span>
            <strong className="card-value">{displayMetrics.total_trips}</strong>
            <span className="card-sub">Lifetime drive sessions</span>
          </article>
          <article className="card stat-card">
            <span className="card-title">Trips This Week</span>
            <strong className="card-value">{displayMetrics.trips_this_week}</strong>
            <span className="card-sub">Last 7 days</span>
          </article>
        </section>
      </div>
    </div>
  )
}
