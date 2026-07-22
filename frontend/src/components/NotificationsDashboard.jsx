import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#ef4444', '#f59e0b', '#6366f1', '#10b981', '#8b5cf6', '#ec4899']

const card = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  marginBottom: 16,
}

const badge = (bg) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: bg,
})

const priorityColor = (p) => {
  if (p === 'critical') return '#ef4444'
  if (p === 'high') return '#f59e0b'
  if (p === 'medium') return '#6366f1'
  if (p === 'low') return '#10b981'
  return '#94a3b8'
}

const catColor = (c) => {
  if (c === 'seizure') return '#ef4444'
  if (c === 'alert') return '#f59e0b'
  if (c === 'medication') return '#8b5cf6'
  if (c === 'form') return '#10b981'
  if (c === 'activity') return '#6366f1'
  if (c === 'result') return '#64748b'
  return '#94a3b8'
}

const catLabel = (c) => {
  if (c === 'seizure') return 'Seizure'
  if (c === 'alert') return 'Alert'
  if (c === 'medication') return 'Medication'
  if (c === 'form') return 'Form'
  if (c === 'activity') return 'Activity'
  if (c === 'result') return 'Result'
  return c
}

export default function NotificationsDashboard() {
  const [tab, setTab] = useState('all')
  const [overview, setOverview] = useState(null)
  const [catNotifications, setCatNotifications] = useState(null)
  const [unread, setUnread] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [selectedCat, setSelectedCat] = useState('seizure')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    if (tab === 'all' || tab === 'summary') {
      axios.get(API + '/api/notifications')
        .then(r => { setOverview(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'unread') {
      axios.get(API + '/api/notifications/unread')
        .then(r => { setUnread(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'category') {
      axios.get(API + '/api/notifications/category/' + selectedCat)
        .then(r => { setCatNotifications(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'definitions') {
      axios.get(API + '/api/notifications/definitions')
        .then(r => { setDefinitions(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, selectedCat])

  const tabs = [
    { id: 'all', label: 'All Notifications' },
    { id: 'summary', label: 'Summary' },
    { id: 'unread', label: 'Unread' },
    { id: 'category', label: 'By Category' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const renderNotification = (n) => (
    <div key={n.id} style={{ ...card, borderLeft: `4px solid ${priorityColor(n.priority)}`, opacity: n.read ? 0.7 : 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={badge(priorityColor(n.priority))}>{n.priority}</span>
          <span style={badge(catColor(n.category))}>{catLabel(n.category)}</span>
          {!n.read && <span style={{ ...badge('#1e88e5'), fontSize: 10 }}>NEW</span>}
        </div>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>{n.patient_id}</span>
      </div>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{n.title}</div>
      <div style={{ fontSize: 13, color: '#64748b', marginBottom: 6 }}>{n.body}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', display: 'flex', gap: 16 }}>
        <span>Source: {n.source_table}</span>
        <span>{new Date(n.timestamp).toLocaleString()}</span>
      </div>
    </div>
  )

  const renderAll = () => {
    if (!overview) return null
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 20 }}>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700 }}>{overview.total}</div><div style={{ color: '#64748b', fontSize: 13 }}>Total</div></div>
          <div style={{ ...card, borderLeft: '4px solid #ef4444' }}><div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{overview.unread}</div><div style={{ color: '#64748b', fontSize: 13 }}>Unread</div></div>
          <div style={card}><div style={{ fontSize: 28, fontWeight: 700, color: '#10b981' }}>{overview.total - overview.unread}</div><div style={{ color: '#64748b', fontSize: 13 }}>Read</div></div>
        </div>
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          {overview.notifications?.slice(0, 50).map(renderNotification)}
          {overview.notifications?.length > 50 && <div style={{ textAlign: 'center', color: '#94a3b8', padding: 12 }}>Showing 50 of {overview.notifications.length}</div>}
        </div>
      </div>
    )
  }

  const renderSummary = () => {
    if (!overview) return null
    const catData = Object.entries(overview.by_category || {}).map(([k, v]) => ({ name: catLabel(k), value: v, fill: catColor(k) }))
    const priData = Object.entries(overview.by_priority || {}).map(([k, v]) => ({ name: k, value: v, fill: priorityColor(k) }))
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>By Category</h4>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={catData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {catData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>By Priority</h4>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={priData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value">{priData.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    )
  }

  const renderUnread = () => {
    if (!unread) return null
    return (
      <div>
        <div style={{ ...card, borderLeft: '4px solid #ef4444' }}>
          <div style={{ fontSize: 24, fontWeight: 700 }}>{unread.total} unread notifications</div>
        </div>
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          {unread.notifications?.slice(0, 50).map(renderNotification)}
          {unread.notifications?.length > 50 && <div style={{ textAlign: 'center', color: '#94a3b8', padding: 12 }}>Showing 50 of {unread.notifications.length}</div>}
        </div>
      </div>
    )
  }

  const renderCategory = () => {
    const categories = ['seizure', 'alert', 'medication', 'form', 'activity', 'result']
    return (
      <div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          {categories.map(c => (
            <button key={c} onClick={() => setSelectedCat(c)}
              style={{ padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
                background: selectedCat === c ? catColor(c) : '#f1f5f9', color: selectedCat === c ? '#fff' : '#334155' }}>
              {catLabel(c)}
            </button>
          ))}
        </div>
        {catNotifications && (
          <div>
            <div style={card}><strong>{catNotifications.total}</strong> notifications in <span style={badge(catColor(selectedCat))}>{catLabel(selectedCat)}</span></div>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              {catNotifications.notifications?.slice(0, 40).map(renderNotification)}
            </div>
          </div>
        )}
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        {['categories', 'priorities', 'source_tables'].map(section => (
          <div key={section} style={card}>
            <h4 style={{ marginTop: 0, textTransform: 'capitalize' }}>{section.replace('_', ' ')}</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(definitions[section] || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, width: '25%' }}>{k}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginTop: 0 }}>Notification Centre</h2>
      <p style={{ color: '#64748b', marginBottom: 20 }}>
        Clinical notifications from assessments, seizure events, medications, forms, and activity logs — sourced from clinical.db
      </p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ padding: '8px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600,
              background: tab === t.id ? '#1e88e5' : '#f1f5f9', color: tab === t.id ? '#fff' : '#334155' }}>
            {t.label}
          </button>
        ))}
      </div>
      {loading && <div style={card}>Loading...</div>}
      {error && <div style={{ ...card, borderLeft: '4px solid #ef4444', color: '#ef4444' }}>Error: {error}</div>}
      {!loading && !error && tab === 'all' && renderAll()}
      {!loading && !error && tab === 'summary' && renderSummary()}
      {!loading && !error && tab === 'unread' && renderUnread()}
      {!loading && !error && tab === 'category' && renderCategory()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
