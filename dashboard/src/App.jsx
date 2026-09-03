import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/live'

function App() {
  const [trades, setTrades] = useState([])
  const [alerts, setAlerts] = useState([])
  const [stats, setStats] = useState(null)
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef(null)

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

  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => setWsConnected(true)
    ws.onmessage = (event) => {
      const alert = JSON.parse(event.data)
      setAlerts(prev => [{ ...alert, _key: `${alert.id}-${Date.now()}` }, ...prev].slice(0, 30))
    }
    ws.onerror = () => setWsConnected(false)
    ws.onclose = () => setWsConnected(false)

    return () => ws.close()
  }, [])

  const confidenceLevel = (conf) => {
    if (conf >= 0.8) return 'high'
    if (conf >= 0.5) return 'medium'
    return 'low'
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>QuantumLedger</h1>
          <span className="subtitle">Real-time GNN fraud detection</span>
        </div>
        <div className={`live-badge ${wsConnected ? 'connected' : 'disconnected'}`}>
          <span className="pulse-dot"></span>
          {wsConnected ? 'Live' : 'Disconnected'}
        </div>
      </header>

      {stats && (
        <div className="stats-grid">
          <div className="stat-card">
            <span className="stat-label">Accuracy</span>
            <span className="stat-value">{(stats.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Precision</span>
            <span className="stat-value">{(stats.precision * 100).toFixed(1)}%</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Recall</span>
            <span className="stat-value">{(stats.recall * 100).toFixed(1)}%</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">F1 Score</span>
            <span className="stat-value">{stats.f1.toFixed(3)}</span>
          </div>
          <div className="stat-card highlight">
            <span className="stat-label">Total Alerts</span>
            <span className="stat-value">{stats.total_alerts.toLocaleString()}</span>
          </div>
        </div>
      )}

      <div className="panels">
        <div className="panel">
          <div className="panel-header">
            <h2>Recent Trades</h2>
            <span className="badge-count">{trades.length}</span>
          </div>
          <div className="list">
            {trades.length === 0 && <div className="empty-state">Waiting for trades…</div>}
            {trades.map(t => (
              <div key={t.id} className={`row ${t.is_fraud_actual ? 'row-fraud' : ''}`}>
                <div className="row-main">
                  <span className="trader-pair">{t.trader_id} <span className="arrow">→</span> {t.counterparty_id}</span>
                  {t.is_fraud_actual && <span className="tag tag-fraud">{t.fraud_type}</span>}
                </div>
                <div className="row-meta">
                  <span>vol {t.volume.toLocaleString()}</span>
                  <span className="price">${t.price.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>Live Fraud Alerts</h2>
            <span className="badge-count">{alerts.length}</span>
          </div>
          <div className="list">
            {alerts.length === 0 && <div className="empty-state">Listening for alerts…</div>}
            {alerts.map(a => (
              <div key={a._key} className={`row alert-enter ${a.predicted_fraud ? `row-alert ${confidenceLevel(a.confidence)}` : ''}`}>
                <div className="row-main">
                  <span className="trader-pair">{a.trader_id} <span className="arrow">→</span> {a.counterparty_id}</span>
                  {a.predicted_fraud && <span className="tag tag-alert">FLAGGED</span>}
                </div>
                <div className="row-meta">
                  <span>vol {a.volume.toLocaleString()}</span>
                  <span className="confidence-bar-wrap">
                    <span className="confidence-bar" style={{ width: `${a.confidence * 100}%` }}></span>
                  </span>
                  <span className="conf-text">{(a.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App