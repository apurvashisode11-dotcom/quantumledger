import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/live'

function App() {
  const [trades, setTrades] = useState([])
  const [alerts, setAlerts] = useState([])
  const [stats, setStats] = useState(null)
  const wsRef = useRef(null)

  // Fetch recent trades once on load, and refresh every 5 seconds
  useEffect(() => {
    const fetchTrades = () => {
      fetch(`${API_BASE}/trades/recent?limit=20`)
        .then(res => res.json())
        .then(data => setTrades(data))
        .catch(err => console.error('Failed to fetch trades:', err))
    }
    fetchTrades()
    const interval = setInterval(fetchTrades, 5000)
    return () => clearInterval(interval)
  }, [])

  // Fetch live stats, refresh every 5 seconds
  useEffect(() => {
    const fetchStats = () => {
      fetch(`${API_BASE}/alerts/stats`)
        .then(res => res.json())
        .then(data => setStats(data))
        .catch(err => console.error('Failed to fetch stats:', err))
    }
    fetchStats()
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [])

  // Connect to WebSocket for live alerts
  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => console.log('WebSocket connected')
    ws.onmessage = (event) => {
      const alert = JSON.parse(event.data)
      setAlerts(prev => [alert, ...prev].slice(0, 30))
    }
    ws.onerror = (err) => console.error('WebSocket error:', err)
    ws.onclose = () => console.log('WebSocket disconnected')

    return () => ws.close()
  }, [])

  return (
    <div className="dashboard">
      <h1>QuantumLedger — Live Fraud Detection</h1>

      {stats && (
        <div className="stats-bar">
          <div className="stat"><span>Accuracy</span><strong>{(stats.accuracy * 100).toFixed(1)}%</strong></div>
          <div className="stat"><span>Precision</span><strong>{(stats.precision * 100).toFixed(1)}%</strong></div>
          <div className="stat"><span>Recall</span><strong>{(stats.recall * 100).toFixed(1)}%</strong></div>
          <div className="stat"><span>F1 Score</span><strong>{stats.f1.toFixed(3)}</strong></div>
          <div className="stat"><span>Total Alerts</span><strong>{stats.total_alerts}</strong></div>
        </div>
      )}

      <div className="panels">
        <div className="panel">
          <h2>Recent Trades</h2>
          <div className="trade-list">
            {trades.map(t => (
              <div key={t.id} className={`trade-row ${t.is_fraud_actual ? 'fraud' : ''}`}>
                <span>{t.trader_id} → {t.counterparty_id}</span>
                <span>vol: {t.volume}</span>
                <span>${t.price}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Live Fraud Alerts</h2>
          <div className="alert-list">
            {alerts.map((a, i) => (
              <div key={i} className={`alert-row ${a.predicted_fraud ? 'flagged' : ''}`}>
                <span>{a.trader_id} → {a.counterparty_id}</span>
                <span>vol: {a.volume}</span>
                <span>conf: {(a.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App