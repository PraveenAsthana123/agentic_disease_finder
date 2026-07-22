import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#8b5cf6' }

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function StoriesTestsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [statusFilter, setStatusFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/stories-tests/overview`),
      axios.get(`${API_URL}/api/stories-tests/breakdown`),
      axios.get(`${API_URL}/api/stories-tests/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Stories & Tests...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'user-stories', 'demo-stories', 'testing', 'definitions']
  const k = overview.summary || {}

  /* ── Overview Tab ── */
  const renderOverview = () => {
    const statusDist = overview.status_distribution || []
    const personaDist = (k.personas || []).map(p => ({ name: p, value: 1 }))
    const dims = overview.dimension_table || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Key Metrics" span={2}>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <KPI label="User Stories" value={k.total_user_stories} />
            <KPI label="Demo Stories" value={k.total_demo_stories} />
            <KPI label="Test Dimensions" value={k.total_test_dimensions} />
            <KPI label="Built Dims" value={k.built} />
            <KPI label="Partial Dims" value={k.partial} />
          </div>
        </Card>

        <Card title="Status Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={statusDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
                {statusDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Stories per Persona">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={personaDist} layout="vertical" margin={{ left: 100, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" name="Stories" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="All Test Dimensions" span={2}>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Dimension</th>
                <th style={thStyle}>Tests</th>
                <th style={thStyle}>How</th>
                <th style={thStyle}>Status</th>
              </tr></thead>
              <tbody>
                {dims.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{d.dim}</td>
                    <td style={tdStyle}>{d.tests}</td>
                    <td style={tdStyle}>{d.how}</td>
                    <td style={tdStyle}><StatusBadge status={d.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  /* ── User Stories Tab ── */
  const renderUserStories = () => {
    const bd = breakdown || {}
    const stories = bd.user_stories || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
        {stories.map((s, i) => (
          <Card key={i} title={s.title || `Story ${i + 1}`}>
            <div style={{ marginBottom: 8 }}>
              <span style={{
                background: '#3b82f622', color: '#3b82f6', border: '1px solid #3b82f655',
                borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
              }}>{s.persona}</span>
            </div>
            <p style={{ fontSize: 12, color: '#334155', margin: '8px 0', lineHeight: 1.5 }}>{s.story}</p>
            {s.endpoint && (
              <p style={{ fontSize: 11, color: '#64748b', margin: '4px 0' }}>
                <strong>Endpoint:</strong>{' '}
                <code style={{ background: '#f1f5f9', padding: '1px 4px', borderRadius: 3, fontSize: 11 }}>{s.endpoint}</code>
              </p>
            )}
          </Card>
        ))}
      </div>
    )
  }

  /* ── Demo Stories Tab ── */
  const renderDemoStories = () => {
    const bd = breakdown || {}
    const demos = bd.demo_stories || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
        {demos.map((d, i) => (
          <Card key={i} title={d.title || `Demo ${i + 1}`}>
            <p style={{ fontSize: 12, color: '#334155', margin: '8px 0', lineHeight: 1.5 }}>{d.script}</p>
            {d.shows && (
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 11, color: '#64748b' }}>Shows:</strong>
                <p style={{ fontSize: 12, color: '#334155', margin: '4px 0' }}>{d.shows}</p>
              </div>
            )}
          </Card>
        ))}
      </div>
    )
  }

  /* ── Testing Tab ── */
  const renderTesting = () => {
    const bd = breakdown || {}
    const dims = bd.testing || []
    const statuses = [...new Set(dims.map(d => d.status))]
    const filtered = statusFilter === 'all' ? dims : dims.filter(d => d.status === statusFilter)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Test Dimensions (${filtered.length})`}>
          <div style={{ marginBottom: 12 }}>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              style={{
                padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0',
                fontSize: 12, cursor: 'pointer', color: '#334155', background: '#fff'
              }}
            >
              <option value="all">All Statuses</option>
              {statuses.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>Dimension</th>
                <th style={thStyle}>Tests</th>
                <th style={thStyle}>How</th>
                <th style={thStyle}>Status</th>
              </tr></thead>
              <tbody>
                {filtered.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{d.dim}</td>
                    <td style={tdStyle}>{d.tests}</td>
                    <td style={tdStyle}>{d.how}</td>
                    <td style={tdStyle}><StatusBadge status={d.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    const d = defs || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Description</th></tr></thead>
            <tbody>
              {(d.status_legend || []).map((s, i) => (
                <tr key={i}><td style={tdStyle}><StatusBadge status={s.status} /></td><td style={tdStyle}>{s.meaning || s.description || s.label}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Glossary" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
            <tbody>
              {(d.glossary || []).map((g, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                  <td style={tdStyle}>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {(d.notes || d.clinical_notes || []).length > 0 && (
          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(d.notes || d.clinical_notes || []).map((n, i) => <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{n}</li>)}
            </ul>
          </Card>
        )}

        {d.references && d.references.length > 0 && (
          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.references.map((r, i) => <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>{typeof r === 'string' ? r : `${r.ref} — ${r.detail}`}</li>)}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Stories & Tests</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {k.total_user_stories || 0} user stories, {k.total_demo_stories || 0} demo stories, {k.total_test_dimensions || 0} test dimensions ({k.pct_built || 0}% built)
      </p>

      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', background: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t ? 600 : 400,
            color: tab === t ? '#3b82f6' : '#64748b',
            borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            marginBottom: -2, textTransform: 'capitalize'
          }}>{t.replace(/-/g, ' ')}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'user-stories' && renderUserStories()}
      {tab === 'demo-stories' && renderDemoStories()}
      {tab === 'testing' && renderTesting()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
