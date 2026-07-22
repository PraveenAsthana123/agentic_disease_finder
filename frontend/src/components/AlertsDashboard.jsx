import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#dc2626', '#ef4444', '#f59e0b', '#22c55e', '#1e88e5', '#7c4dff', '#ec4899', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '—')

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

const severityColor = (s) => {
  const v = (s || '').toLowerCase()
  if (v === 'critical') return '#dc2626'
  if (v === 'high') return '#ef4444'
  if (v === 'medium') return '#f59e0b'
  if (v === 'low') return '#22c55e'
  return '#94a3b8'
}

const categoryIcon = (c) => {
  const v = (c || '').toLowerCase()
  if (v === 'assessment') return '📋'
  if (v === 'seizure') return '⚡'
  if (v === 'medication') return '💊'
  if (v === 'vitals') return '❤️'
  return '🚨'
}

const categoryLabel = (c) => {
  const v = (c || '').toLowerCase()
  if (v === 'assessment') return 'Assessment'
  if (v === 'seizure') return 'Seizure'
  if (v === 'medication') return 'Medication'
  if (v === 'vitals') return 'Vitals'
  return c
}

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

export default function AlertsDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [defs, setDefs] = useState(null)
  const [filter, setFilter] = useState('all')
  const [sevFilter, setSevFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/alerts`),
      axios.get(`${API_URL}/api/alerts/definitions`),
    ])
      .then(([overviewRes, defsRes]) => {
        setData(overviewRes.data)
        setDefs(defsRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Alerts...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data || !data.available) return <div style={{ padding: 40 }}>Alert data not available</div>

  const alerts = data.alerts || []
  let filtered = filter === 'all' ? alerts : alerts.filter(a => a.category === filter)
  if (sevFilter !== 'all') filtered = filtered.filter(a => a.severity === sevFilter)

  // Chart data
  const sevData = Object.entries(data.by_severity || {}).map(([k, v]) => ({
    name: k, value: v, fill: severityColor(k)
  }))
  const catData = Object.entries(data.by_category || {}).map(([k, v]) => ({
    name: categoryLabel(k), value: v
  }))

  const criticalCount = data.by_severity?.critical || 0
  const highCount = data.by_severity?.high || 0

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>🚨 Clinical Alerts</h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
            Real-time clinical alerts from clinical.db — assessments, seizures, medications, vitals
          </p>
        </div>
        <button
          onClick={() => setShowDefs(!showDefs)}
          style={{ padding: '6px 14px', borderRadius: 8, border: '1px solid #cbd5e1', background: showDefs ? '#e2e8f0' : '#fff', cursor: 'pointer', fontSize: 13 }}
        >
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {/* Definitions panel */}
      {showDefs && defs && defs.definitions && (
        <div style={{ ...cardStyle, marginBottom: 20, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>Severity Levels</h4>
              {Object.entries(defs.definitions.severity_levels || {}).map(([k, v]) => (
                <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                  <span style={badgeStyle(severityColor(k))}>{k}</span> {v}
                </div>
              ))}
            </div>
            <div>
              <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>Categories</h4>
              {Object.entries(defs.definitions.categories || {}).map(([k, v]) => (
                <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                  <strong>{categoryIcon(k)} {categoryLabel(k)}:</strong> {v}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <div style={{ ...cardStyle, borderLeft: '4px solid #dc2626' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Total Alerts</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{fmt(data.total_alerts)}</div>
        </div>
        <div style={{ ...cardStyle, borderLeft: '4px solid #ef4444' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Critical</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#dc2626' }}>{fmt(criticalCount)}</div>
        </div>
        <div style={{ ...cardStyle, borderLeft: '4px solid #f59e0b' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>High</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{fmt(highCount)}</div>
        </div>
        <div style={{ ...cardStyle, borderLeft: '4px solid #1e88e5' }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Patients Affected</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{fmt(data.patients_affected)}</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>By Severity</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                {sevData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>By Category</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={catData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                {catData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, color: '#64748b', alignSelf: 'center', marginRight: 4 }}>Category:</span>
        {['all', 'assessment', 'seizure', 'medication', 'vitals'].map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            style={{
              padding: '4px 12px', borderRadius: 16, border: '1px solid #cbd5e1',
              background: filter === cat ? '#1e293b' : '#fff',
              color: filter === cat ? '#fff' : '#475569',
              cursor: 'pointer', fontSize: 12, fontWeight: 500,
            }}
          >
            {cat === 'all' ? 'All' : `${categoryIcon(cat)} ${categoryLabel(cat)}`}
          </button>
        ))}
        <span style={{ fontSize: 13, color: '#64748b', alignSelf: 'center', marginLeft: 12, marginRight: 4 }}>Severity:</span>
        {['all', 'critical', 'high', 'medium', 'low'].map(sev => (
          <button
            key={sev}
            onClick={() => setSevFilter(sev)}
            style={{
              padding: '4px 12px', borderRadius: 16, border: '1px solid #cbd5e1',
              background: sevFilter === sev ? severityColor(sev) || '#1e293b' : '#fff',
              color: sevFilter === sev ? '#fff' : '#475569',
              cursor: 'pointer', fontSize: 12, fontWeight: 500,
            }}
          >
            {sev === 'all' ? 'All' : sev.charAt(0).toUpperCase() + sev.slice(1)}
          </button>
        ))}
      </div>

      {/* Alert list */}
      <div style={{ fontSize: 13, color: '#64748b', marginBottom: 8 }}>
        Showing {filtered.length} of {alerts.length} alerts
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {filtered.map(a => (
          <div
            key={a.id}
            style={{
              ...cardStyle,
              borderLeft: `4px solid ${severityColor(a.severity)}`,
              padding: '14px 18px',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                  <span style={badgeStyle(severityColor(a.severity))}>{a.severity}</span>
                  <span style={{ fontSize: 11, color: '#94a3b8' }}>{categoryIcon(a.category)} {categoryLabel(a.category)}</span>
                  {a.type && <span style={{ fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>{a.type.replace(/_/g, ' ')}</span>}
                </div>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 2 }}>{a.title}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{a.body}</div>
                {a.action_required && (
                  <div style={{ fontSize: 12, color: '#7c2d12', marginTop: 4, fontWeight: 500 }}>
                    Action: {a.action_required}
                  </div>
                )}
              </div>
              <div style={{ textAlign: 'right', minWidth: 120 }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>{a.patient_id}</div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>{a.timestamp ? new Date(a.timestamp).toLocaleDateString() : '—'}</div>
              </div>
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ ...cardStyle, textAlign: 'center', color: '#94a3b8', padding: 40 }}>
            No alerts match the current filters
          </div>
        )}
      </div>
    </div>
  )
}
