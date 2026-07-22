import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#ec4899']

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

const catColor = (cat) => {
  if (cat === 'team_message') return '#6366f1'
  if (cat === 'clinical_decision') return '#ef4444'
  if (cat === 'expert_review') return '#f59e0b'
  if (cat === 'form_assignment') return '#10b981'
  if (cat === 'system') return '#8b5cf6'
  return '#64748b'
}

export default function InboxDashboard() {
  const [tab, setTab] = useState('all')
  const [overview, setOverview] = useState(null)
  const [catMessages, setCatMessages] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [selectedCat, setSelectedCat] = useState('team_message')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    if (tab === 'all' || tab === 'summary') {
      axios.get(API + '/api/inbox')
        .then(r => { setOverview(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'category') {
      axios.get(API + '/api/inbox/category/' + selectedCat)
        .then(r => { setCatMessages(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    } else if (tab === 'definitions') {
      axios.get(API + '/api/inbox/definitions')
        .then(r => { setDefinitions(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, selectedCat])

  const tabs = [
    { id: 'all', label: 'All Messages' },
    { id: 'summary', label: 'Summary' },
    { id: 'category', label: 'By Category' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const renderMessage = (msg) => (
    <div key={msg.id} style={{ ...card, borderLeft: `4px solid ${catColor(msg.category)}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={badge(catColor(msg.category))}>
            {msg.category === 'team_message' ? 'Care Team' :
             msg.category === 'clinical_decision' ? 'Clinical Decision' :
             msg.category === 'expert_review' ? 'Expert Review' :
             msg.category === 'form_assignment' ? 'Form' : msg.category}
          </span>
          {msg.is_bot && <span style={badge('#8b5cf6')}>Bot</span>}
          {msg.patient_id && <span style={{ fontSize: 12, color: '#64748b' }}>Patient: {msg.patient_id}</span>}
        </div>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>
          {msg.timestamp ? new Date(msg.timestamp).toLocaleString() : ''}
        </span>
      </div>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{msg.subject}</div>
      <div style={{ fontSize: 14, color: '#475569' }}>{msg.body}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
        From: {msg.from_role}{msg.channel ? ` | Channel: ${msg.channel}` : ''}
      </div>
    </div>
  )

  const renderAll = () => {
    if (!overview) return null
    const { messages, total, summary } = overview
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12, marginBottom: 20 }}>
          <div style={{ ...card, textAlign: 'center' }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#6366f1' }}>{total}</div>
            <div style={{ fontSize: 13, color: '#64748b' }}>Total Messages</div>
          </div>
          {(summary?.by_category || []).map((c, i) => (
            <div key={c.category} style={{ ...card, textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700, color: catColor(c.category) }}>{c.count}</div>
              <div style={{ fontSize: 13, color: '#64748b' }}>{c.icon} {c.label}</div>
            </div>
          ))}
        </div>
        <h3 style={{ marginBottom: 12, color: '#1e293b' }}>All Messages ({total})</h3>
        {messages.map(renderMessage)}
      </div>
    )
  }

  const renderSummary = () => {
    if (!overview) return null
    const { summary, total } = overview
    const catData = (summary?.by_category || []).map(c => ({
      name: c.label,
      value: c.count,
      category: c.category,
    }))
    const channelData = (summary?.by_channel || []).map(c => ({
      name: c.channel,
      count: c.count,
    }))

    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div style={card}>
            <h3 style={{ marginBottom: 12 }}>Messages by Category</h3>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={catData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {catData.map((d, i) => (
                    <Cell key={i} fill={catColor(d.category)} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <h3 style={{ marginBottom: 12 }}>Messages by Channel</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={channelData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div style={card}>
          <h3 style={{ marginBottom: 8 }}>Inbox Stats</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: 8, textAlign: 'left' }}>Category</th>
                <th style={{ padding: 8, textAlign: 'right' }}>Count</th>
                <th style={{ padding: 8, textAlign: 'right' }}>% of Total</th>
              </tr>
            </thead>
            <tbody>
              {catData.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8 }}>
                    <span style={{ ...badge(catColor(d.category)), marginRight: 8 }}>{d.name}</span>
                  </td>
                  <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{d.value}</td>
                  <td style={{ padding: 8, textAlign: 'right', color: '#64748b' }}>
                    {total ? ((d.value / total) * 100).toFixed(1) : 0}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 8, fontSize: 12, color: '#94a3b8' }}>
            Source: {overview.source}
          </div>
        </div>
      </div>
    )
  }

  const renderCategory = () => {
    const categories = [
      { id: 'team_message', label: 'Care Team' },
      { id: 'clinical_decision', label: 'Clinical Decision' },
      { id: 'expert_review', label: 'Expert Review' },
      { id: 'form_assignment', label: 'Form / Questionnaire' },
    ]
    return (
      <div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
          {categories.map(c => (
            <button key={c.id} onClick={() => setSelectedCat(c.id)}
              style={{
                padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
                background: selectedCat === c.id ? catColor(c.id) : '#f1f5f9',
                color: selectedCat === c.id ? '#fff' : '#475569',
                fontWeight: 600, fontSize: 13,
              }}>
              {c.label}
            </button>
          ))}
        </div>
        {catMessages && (
          <div>
            <h3 style={{ marginBottom: 12, color: '#1e293b' }}>
              {catMessages.label} ({catMessages.total})
            </h3>
            {(catMessages.messages || []).map(renderMessage)}
            {catMessages.total === 0 && (
              <div style={{ ...card, textAlign: 'center', color: '#94a3b8' }}>
                No messages in this category
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        <div style={card}>
          <h3 style={{ marginBottom: 12 }}>Message Categories</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: 10, textAlign: 'left' }}>Icon</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Category</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(definitions.categories || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 10, fontSize: 20 }}>{c.icon}</td>
                  <td style={{ padding: 10 }}>
                    <span style={badge(catColor(c.id))}>{c.label}</span>
                  </td>
                  <td style={{ padding: 10, color: '#475569' }}>{c.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12, fontSize: 12, color: '#94a3b8' }}>
            Source: {definitions.source}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ marginBottom: 4, color: '#1e293b' }}>Message Inbox</h2>
      <p style={{ color: '#64748b', marginBottom: 16, fontSize: 14 }}>
        Secure messages: care team conversations, clinical decisions, expert reviews, form assignments
      </p>
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              borderBottom: tab === t.id ? '3px solid #6366f1' : '3px solid transparent',
              background: 'transparent',
              color: tab === t.id ? '#6366f1' : '#64748b',
              fontWeight: tab === t.id ? 700 : 500,
              fontSize: 14, marginBottom: -2,
            }}>
            {t.label}
          </button>
        ))}
      </div>
      {loading && <div style={{ textAlign: 'center', padding: 40, color: '#94a3b8' }}>Loading...</div>}
      {error && <div style={{ ...card, background: '#fef2f2', color: '#ef4444' }}>Error: {error}</div>}
      {!loading && !error && tab === 'all' && renderAll()}
      {!loading && !error && tab === 'summary' && renderSummary()}
      {!loading && !error && tab === 'category' && renderCategory()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
