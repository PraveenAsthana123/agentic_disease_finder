import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { built: '#4caf50', partial: '#ff9800', planned: '#f44336', unknown: '#94a3b8' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
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

export default function StoriesTestsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/stories-tests/overview`),
      axios.get(`${API_URL}/stories-tests/breakdown`),
      axios.get(`${API_URL}/stories-tests/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Stories &amp; Tests...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Stories &amp; Tests data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'stories', label: 'User Stories' },
    { id: 'demos', label: 'Demo Stories' },
    { id: 'testing', label: 'Testing Matrix' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4, color: '#0f172a' }}>Stories &amp; Tests Dashboard</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
        User stories, demo scripts, and the 9-dimension testing matrix from stories_and_tests.json
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="User Stories" value={s.total_user_stories} />
              <KPI label="Demo Stories" value={s.total_demo_stories} />
              <KPI label="Test Dimensions" value={s.total_test_dimensions} />
              <KPI label="Built" value={s.built} sub={`${s.pct_built}%`} />
              <KPI label="Partial" value={s.partial} />
              <KPI label="Planned" value={s.planned} />
              <KPI label="Personas" value={s.personas?.length || 0} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Test Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name} (${value})`}>
                    {overview.status_distribution.map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[overview.status_distribution[i].name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Dimensions by Status">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.dimension_table} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="dim" width={70} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="status" fill="#1e88e5">
                    {overview.dimension_table.map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.status] || '#94a3b8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Testing Matrix Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Dimension</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Tests</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>How</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.dimension_table.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{row.dim}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{row.tests}</td>
                      <td style={{ padding: '6px 10px', color: '#64748b', fontFamily: 'monospace', fontSize: 11 }}>{row.how}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ═══ USER STORIES TAB ═══ */}
      {tab === 'stories' && breakdown && (
        <Card title={`User Stories (${breakdown.user_stories.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>#</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Persona</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Story</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Endpoint</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.user_stories.map((us, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{i + 1}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{us.persona}</td>
                    <td style={{ padding: '6px 10px', color: '#475569', maxWidth: 400 }}>{us.story}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11, color: '#1e88e5' }}>{us.endpoint}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ═══ DEMO STORIES TAB ═══ */}
      {tab === 'demos' && breakdown && (
        <Card title={`Demo Stories (${breakdown.demo_stories.length})`}>
          {breakdown.demo_stories.map((ds, i) => (
            <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, marginBottom: 12, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, color: '#1e293b' }}>
                {i + 1}. {ds.title}
              </div>
              <div style={{ fontSize: 12, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>
                <strong>Script:</strong> {ds.script}
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                <strong>Shows:</strong> {ds.shows}
              </div>
            </div>
          ))}
        </Card>
      )}

      {/* ═══ TESTING MATRIX TAB ═══ */}
      {tab === 'testing' && breakdown && (
        <Card title={`9-Dimension Testing Matrix (${breakdown.testing.length} dimensions)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>#</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Dimension</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Tests</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>How</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.testing.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{i + 1}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{t.dim}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{t.tests}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11, color: '#64748b' }}>{t.how}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={t.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Meaning</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.status_legend?.map((sl, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={sl.status} /></td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{sl.meaning}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Term</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.glossary?.map((g, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.term}</td>
                      <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {defs.notes && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.notes.map((n, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{n}</li>
                ))}
              </ul>
            </Card>
          )}

          {defs.references && (
            <Card title="References">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.references.map((r, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{r}</li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
