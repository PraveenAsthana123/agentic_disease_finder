import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308', '#06b6d4', '#84cc16', '#f43f5e']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#3b82f6', unknown: '#cbd5e1' }
const PRIORITY_COLORS = { P0: '#ef4444', P1: '#f97316', P2: '#3b82f6', 'N/A': '#94a3b8' }

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

function PriorityBadge({ priority }) {
  const bg = PRIORITY_COLORS[priority] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {priority}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function ExpertDashboardsCatalogDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/expert-dashboards-catalog/overview`),
      axios.get(`${API_URL}/api/expert-dashboards-catalog/breakdown`),
      axios.get(`${API_URL}/api/expert-dashboards-catalog/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Expert Dashboards Catalog...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-role', 'must-have', 'libraries', 'definitions']
  const k = overview.kpis || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
          <KPI label="Total Dashboards" value={k.total_dashboards} />
          <KPI label="Built" value={k.built} sub="verified" />
          <KPI label="Partial" value={k.partial} />
          <KPI label="Planned" value={k.planned} />
          <KPI label="Unique Roles" value={k.unique_roles} />
          <KPI label="Total Endpoints" value={k.total_endpoints} />
          <KPI label="With Endpoints" value={k.dashboards_with_endpoints} />
          <KPI label="Libraries" value={k.libraries_count} />
        </div>
      </Card>

      <Card title="Status Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.charts?.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(overview.charts?.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Priority Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.charts?.priority_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(overview.charts?.priority_distribution || []).map((e, i) => <Cell key={i} fill={PRIORITY_COLORS[e.name] || COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Dashboards per Role" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={overview.charts?.dashboards_per_role || []} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Dashboards" span={2}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Role</th>
                <th style={thStyle}>Visualization</th>
                <th style={thStyle}>Priority</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Endpoints</th>
              </tr>
            </thead>
            <tbody>
              {(overview.dashboards_table || []).map((d, i) => (
                <tr key={i}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{d.name}</td>
                  <td style={tdStyle}>{d.role}</td>
                  <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{d.viz}</td>
                  <td style={tdStyle}><PriorityBadge priority={d.priority} /></td>
                  <td style={tdStyle}><StatusBadge status={d.status} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{d.endpoints || 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderByRole = () => {
    const perRole = breakdown?.per_role || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
        {perRole.map((group, gi) => (
          <Card key={gi} title={`${group.role} (${group.count})`}>
            {group.dashboards.map((d, di) => (
              <div key={di} style={{
                border: '1px solid #e2e8f0', borderRadius: 6, padding: 10, marginBottom: 8,
                background: '#f8fafc'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.name}</span>
                  <StatusBadge status={d.status} />
                </div>
                {d.viz && <div style={{ fontSize: 11, color: '#3b82f6', marginBottom: 2 }}>{d.viz}</div>}
                {d.feature && <div style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>{d.feature}</div>}
                {d.why && <div style={{ fontSize: 10, color: '#94a3b8', fontStyle: 'italic' }}>{d.why}</div>}
                {d.priority && d.priority !== 'N/A' && (
                  <div style={{ marginTop: 4 }}><PriorityBadge priority={d.priority} /></div>
                )}
                {d.endpoints && d.endpoints.length > 0 && (
                  <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {d.endpoints.map((ep, ei) => (
                      <span key={ei} style={{
                        background: '#dbeafe', color: '#1e40af', borderRadius: 3,
                        padding: '1px 6px', fontSize: 10, fontFamily: 'monospace'
                      }}>{ep}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }

  const renderMustHave = () => {
    const mustHave = breakdown?.must_have_p0 || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Must-Have P0 Visualizations & Dashboards" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Critical clinical visualizations required for safe EEG interpretation.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 8 }}>
            {mustHave.map((m, i) => (
              <div key={i} style={{
                border: '1px solid #e2e8f0', borderRadius: 6, padding: 10,
                background: m.type === 'dashboard' ? '#f0fdf4' : '#f8fafc'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{m.name}</span>
                  {m.status && <StatusBadge status={m.status} />}
                </div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{m.type}</div>
                {m.note && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{m.note}</div>}
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const renderLibraries = () => {
    const libs = overview.libraries || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Technology Stack" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Libraries and frameworks powering the expert dashboard visualizations.
          </p>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Purpose</th>
                <th style={thStyle}>Library / Framework</th>
              </tr>
            </thead>
            <tbody>
              {libs.map((l, i) => (
                <tr key={i}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{l.purpose}</td>
                  <td style={tdStyle}>
                    <span style={{
                      background: '#dbeafe', color: '#1e40af', borderRadius: 3,
                      padding: '2px 8px', fontSize: 11, fontFamily: 'monospace'
                    }}>{l.library}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!defs) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          {(defs.status_legend || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{
                width: 14, height: 14, borderRadius: 3,
                background: s.color || '#94a3b8', display: 'inline-block', flexShrink: 0
              }} />
              <span style={{ fontWeight: 600, fontSize: 12, minWidth: 60 }}>{s.status}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{s.meaning}</span>
            </div>
          ))}
        </Card>
        <Card title="Glossary" span={2}>
          <div style={{ maxHeight: 350, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
        <Card title="Clinical Notes">
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#334155', lineHeight: 1.7 }}>
            {(defs.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
          </ul>
        </Card>
        <Card title="References">
          {(defs.references || []).map((r, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{r.label}</span>
              <span style={{ fontSize: 11, color: '#64748b', marginLeft: 6 }}>{r.detail}</span>
            </div>
          ))}
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
        Expert Dashboards Catalog
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>{overview.note}</p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t ? '2px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t ? '#eff6ff' : '#fff',
            color: tab === t ? '#2563eb' : '#64748b'
          }}>
            {t === 'overview' ? 'Overview' : t === 'by-role' ? 'By Role' : t === 'must-have' ? 'Must-Have P0' : t === 'libraries' ? 'Libraries' : 'Definitions'}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'by-role' && renderByRole()}
      {tab === 'must-have' && renderMustHave()}
      {tab === 'libraries' && renderLibraries()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
