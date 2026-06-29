import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#22c55e', '#f59e0b', '#ef4444', '#7c4dff', '#ec4899', '#14b8a6', '#94a3b8']
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

const typeIcon = (t) => {
  const v = (t || '').toLowerCase()
  if (v === 'screening') return '\uD83D\uDD2C'
  if (v === 'adherence') return '\uD83D\uDC8A'
  if (v === 'safety') return '\uD83D\uDEE1\uFE0F'
  if (v === 'form_completion') return '\uD83D\uDCCB'
  if (v === 'education') return '\uD83C\uDF93'
  return '\uD83D\uDCE3'
}

const typeLabel = (t) => {
  const v = (t || '').toLowerCase()
  if (v === 'screening') return 'Screening'
  if (v === 'adherence') return 'Adherence'
  if (v === 'safety') return 'Safety'
  if (v === 'form_completion') return 'Forms'
  if (v === 'education') return 'Education'
  return t
}

const statusColor = (s) => {
  const v = (s || '').toLowerCase()
  if (v === 'active') return '#22c55e'
  if (v === 'completed') return '#1e88e5'
  if (v === 'pending') return '#f59e0b'
  return '#94a3b8'
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

export default function CampaignsDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [defs, setDefs] = useState(null)
  const [typeFilter, setTypeFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/campaigns`),
      axios.get(`${API_URL}/campaigns/definitions`),
    ])
      .then(([overviewRes, defsRes]) => {
        setData(overviewRes.data)
        setDefs(defsRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Campaigns...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data || !data.available) return <div style={{ padding: 40 }}>Campaign data not available</div>

  const campaigns = data.campaigns || []
  let filtered = typeFilter === 'all' ? campaigns : campaigns.filter(c => c.type === typeFilter)

  // Chart data
  const categoryData = Object.entries(data.by_category || {}).map(([name, count]) => ({ name, count }))
  const typeData = Object.entries(data.by_type || {}).map(([name, count]) => ({
    name: typeLabel(name), count
  }))

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
            Health Campaigns & Education
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
            Real health campaigns from clinical.db — screening programs, medication adherence, safety education, and patient engagement
          </p>
        </div>
        <button onClick={() => setShowDefs(!showDefs)}
          style={{ padding: '6px 14px', borderRadius: 8, border: '1px solid #cbd5e1',
                   background: showDefs ? '#1e88e5' : '#fff', color: showDefs ? '#fff' : '#475569',
                   cursor: 'pointer', fontSize: 13 }}>
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {/* Definitions panel */}
      {showDefs && defs && defs.definitions && (
        <div style={{ ...cardStyle, marginBottom: 20, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#0369a1' }}>Metric Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {Object.entries(defs.definitions).map(([key, val]) => (
              <div key={key} style={{ fontSize: 13 }}>
                <strong style={{ color: '#0c4a6e' }}>{key.replace(/_/g, ' ')}:</strong>{' '}
                <span style={{ color: '#334155' }}>{val}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPI cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <div style={cardStyle}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Total Campaigns</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{fmt(data.total_campaigns)}</div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Active</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#22c55e' }}>{fmt(data.active)}</div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Completed</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#1e88e5' }}>{fmt(data.completed)}</div>
        </div>
        <div style={cardStyle}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Pending</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{fmt(data.pending)}</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <div style={cardStyle}>
          <div style={sectionHeadingStyle}>Campaigns by Category</div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={categoryData} dataKey="count" nameKey="name" cx="50%" cy="50%"
                   outerRadius={90} label={({ name, count }) => `${name}: ${count}`}>
                {categoryData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div style={cardStyle}>
          <div style={sectionHeadingStyle}>Campaigns by Type</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={typeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Filter */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {['all', 'screening', 'adherence', 'safety', 'form_completion', 'education'].map(t => (
          <button key={t} onClick={() => setTypeFilter(t)}
            style={{
              padding: '6px 14px', borderRadius: 8,
              border: typeFilter === t ? '2px solid #1e88e5' : '1px solid #e2e8f0',
              background: typeFilter === t ? '#eff6ff' : '#fff',
              color: typeFilter === t ? '#1e88e5' : '#475569',
              cursor: 'pointer', fontSize: 13, fontWeight: typeFilter === t ? 600 : 400,
            }}>
            {t === 'all' ? 'All' : `${typeIcon(t)} ${typeLabel(t)}`} ({
              t === 'all' ? campaigns.length : campaigns.filter(c => c.type === t).length
            })
          </button>
        ))}
      </div>

      {/* Campaign cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {filtered.map((c, i) => (
          <div key={c.id || i} style={{ ...cardStyle, borderLeft: `4px solid ${statusColor(c.status)}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
              <div>
                <span style={{ fontSize: 18, marginRight: 8 }}>{typeIcon(c.type)}</span>
                <strong style={{ fontSize: 15, color: '#1e293b' }}>{c.name}</strong>
              </div>
              <span style={badgeStyle(statusColor(c.status))}>
                {(c.status || '').toUpperCase()}
              </span>
            </div>

            <div style={{ fontSize: 13, color: '#64748b', marginBottom: 8 }}>
              {c.category} &middot; Source: <code style={{ fontSize: 12 }}>{c.source}</code>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 8 }}>
              {c.participants !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Participants</span>
                  <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(c.participants)}</div>
                </div>
              )}
              {c.completion_rate !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Completion</span>
                  <div style={{ fontWeight: 600, color: '#1e293b' }}>{c.completion_rate}%</div>
                </div>
              )}
              {c.assessments_total !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Assessments</span>
                  <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(c.assessments_completed)}/{fmt(c.assessments_total)}</div>
                </div>
              )}
              {c.total_events !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Events</span>
                  <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(c.total_events)}</div>
                </div>
              )}
              {c.injuries_reported !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Injuries</span>
                  <div style={{ fontWeight: 600, color: c.injuries_reported > 0 ? '#ef4444' : '#22c55e' }}>
                    {fmt(c.injuries_reported)}
                  </div>
                </div>
              )}
              {c.medications_tracked !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Medications</span>
                  <div style={{ fontWeight: 600, color: '#1e293b' }}>{fmt(c.medications_tracked)}</div>
                </div>
              )}
              {c.forms_pending !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Pending</span>
                  <div style={{ fontWeight: 600, color: c.forms_pending > 0 ? '#f59e0b' : '#22c55e' }}>
                    {fmt(c.forms_pending)}
                  </div>
                </div>
              )}
              {c.flagged_assessments !== undefined && (
                <div style={{ fontSize: 12 }}>
                  <span style={{ color: '#94a3b8' }}>Flagged</span>
                  <div style={{ fontWeight: 600, color: '#ef4444' }}>{fmt(c.flagged_assessments)}</div>
                </div>
              )}
            </div>

            {c.education_topic && (
              <div style={{ fontSize: 12, color: '#475569', background: '#f8fafc', padding: '6px 10px',
                            borderRadius: 6, marginTop: 4 }}>
                <strong>Education:</strong> {c.education_topic}
              </div>
            )}

            {c.top_triggers && c.top_triggers.length > 0 && (
              <div style={{ fontSize: 12, marginTop: 6 }}>
                <span style={{ color: '#94a3b8' }}>Top triggers: </span>
                {c.top_triggers.map((t, j) => (
                  <span key={j} style={{ ...badgeStyle('#64748b'), marginRight: 4, fontSize: 11 }}>
                    {t.trigger} ({t.count})
                  </span>
                ))}
              </div>
            )}

            {c.severity_distribution && Object.keys(c.severity_distribution).length > 0 && (
              <div style={{ fontSize: 12, marginTop: 6 }}>
                <span style={{ color: '#94a3b8' }}>Severity: </span>
                {Object.entries(c.severity_distribution).map(([sev, cnt], j) => (
                  <span key={j} style={{ marginRight: 8 }}>
                    {sev}: <strong>{cnt}</strong>
                  </span>
                ))}
              </div>
            )}

            {c.latest_date && (
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 6, textAlign: 'right' }}>
                Latest: {c.latest_date.split('T')[0]}
              </div>
            )}
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div style={{ ...cardStyle, textAlign: 'center', color: '#94a3b8', padding: 40 }}>
          No campaigns found for this filter.
        </div>
      )}
    </div>
  )
}
