import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#ef4444', '#f59e0b', '#1e88e5', '#22c55e', '#94a3b8', '#7c4dff', '#ec4899', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '—')

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

const sectionHeadingStyle = {
  fontSize: 16,
  fontWeight: 600,
  color: '#1e293b',
  marginBottom: 12,
  marginTop: 24,
}

const priorityColor = (p) => {
  const v = (p || '').toLowerCase()
  if (v === 'critical') return '#dc2626'
  if (v === 'high') return '#ef4444'
  if (v === 'medium') return '#f59e0b'
  if (v === 'low') return '#22c55e'
  return '#94a3b8'
}

const categoryIcon = (c) => {
  const v = (c || '').toLowerCase()
  if (v === 'result') return '📊'
  if (v === 'form') return '📋'
  if (v === 'seizure') return '⚡'
  if (v === 'medication') return '💊'
  if (v === 'activity') return '📝'
  if (v === 'alert') return '🚨'
  return '🔔'
}

const categoryLabel = (c) => {
  const v = (c || '').toLowerCase()
  if (v === 'result') return 'Results'
  if (v === 'form') return 'Forms'
  if (v === 'seizure') return 'Seizure'
  if (v === 'medication') return 'Medication'
  if (v === 'activity') return 'Activity'
  if (v === 'alert') return 'Alerts'
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

export default function NotificationDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [defs, setDefs] = useState(null)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/notifications`),
      axios.get(`${API_URL}/notifications/definitions`),
    ])
      .then(([overviewRes, defsRes]) => {
        setData(overviewRes.data)
        setDefs(defsRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Notifications...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data || !data.available) return <div style={{ padding: 40 }}>Notification data not available</div>

  const notifications = data.notifications || []
  const filtered = filter === 'all' ? notifications : notifications.filter(n => n.category === filter)

  // Charts data
  const catData = Object.entries(data.by_category || {}).map(([k, v]) => ({
    name: categoryLabel(k), value: v
  }))
  const prioData = Object.entries(data.by_priority || {}).map(([k, v]) => ({
    name: k, value: v, fill: priorityColor(k)
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>🔔 Notification Centre</h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
            Real-time notifications from clinical.db — assessments, forms, seizure events, medications
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
      {showDefs && defs && (
        <div style={{ ...cardStyle, marginBottom: 20, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>Categories</h4>
              {Object.entries(defs.categories || {}).map(([k, v]) => (
                <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                  <strong>{categoryIcon(k)} {k}:</strong> {v}
                </div>
              ))}
            </div>
            <div>
              <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>Priorities</h4>
              {Object.entries(defs.priorities || {}).map(([k, v]) => (
                <div key={k} style={{ fontSize: 12, marginBottom: 4 }}>
                  <span style={badgeStyle(priorityColor(k))}>{k}</span> {v}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <div style={cardStyle}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Total</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{fmt(data.total)}</div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Unread</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{fmt(data.unread)}</div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Critical / High</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#dc2626' }}>
            {fmt((data.by_priority?.critical || 0) + (data.by_priority?.high || 0))}
          </div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Categories</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e88e5' }}>{Object.keys(data.by_category || {}).length}</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>By Category</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={catData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                {catData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div style={cardStyle}>
          <h3 style={sectionHeadingStyle}>By Priority</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={prioData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                {prioData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {['all', 'alert', 'result', 'form', 'seizure', 'medication', 'activity'].map(cat => (
          <button
            key={cat}
            onClick={() => setFilter(cat)}
            style={{
              padding: '5px 14px', borderRadius: 20, border: '1px solid #cbd5e1',
              background: filter === cat ? '#1e293b' : '#fff',
              color: filter === cat ? '#fff' : '#475569',
              cursor: 'pointer', fontSize: 13, fontWeight: filter === cat ? 600 : 400,
            }}
          >
            {cat === 'all' ? '🔔 All' : `${categoryIcon(cat)} ${categoryLabel(cat)}`}
            {cat !== 'all' && ` (${data.by_category?.[cat] || 0})`}
          </button>
        ))}
      </div>

      {/* Notification list */}
      <div style={cardStyle}>
        <h3 style={{ ...sectionHeadingStyle, marginTop: 0 }}>
          {filter === 'all' ? 'All Notifications' : `${categoryLabel(filter)} Notifications`}
          <span style={{ fontWeight: 400, fontSize: 13, color: '#94a3b8', marginLeft: 8 }}>
            ({filtered.length})
          </span>
        </h3>
        {filtered.length === 0 ? (
          <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No notifications in this category</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {filtered.map(n => (
              <div
                key={n.id}
                style={{
                  display: 'flex', alignItems: 'flex-start', gap: 12,
                  padding: '12px 16px', borderRadius: 8,
                  background: n.read ? '#f8fafc' : '#fffbeb',
                  border: `1px solid ${n.priority === 'critical' ? '#fca5a5' : n.priority === 'high' ? '#fde68a' : '#e2e8f0'}`,
                }}
              >
                <span style={{ fontSize: 20, flexShrink: 0 }}>{categoryIcon(n.category)}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                    <span style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{n.title}</span>
                    <span style={badgeStyle(priorityColor(n.priority))}>{n.priority}</span>
                    {!n.read && <span style={{ ...badgeStyle('#1e88e5'), fontSize: 10 }}>NEW</span>}
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{n.body}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>
                    {n.patient_id && <span>Patient: {n.patient_id} · </span>}
                    {n.timestamp && <span>{new Date(n.timestamp).toLocaleString()}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ marginTop: 16, fontSize: 11, color: '#94a3b8', textAlign: 'right' }}>
        Generated: {data.generated_at ? new Date(data.generated_at).toLocaleString() : '—'}
      </div>
    </div>
  )
}
