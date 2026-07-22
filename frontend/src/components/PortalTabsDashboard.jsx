import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308', '#06b6d4', '#84cc16', '#f43f5e']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#94a3b8', unknown: '#cbd5e1' }

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

export default function PortalTabsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/portal-tabs/overview`),
      axios.get(`${API_URL}/api/portal-tabs/breakdown`),
      axios.get(`${API_URL}/api/portal-tabs/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Portal Tabs...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#94a3b8' }}>{overview.note}</div>

  const tabs = ['overview', 'tabs', 'endpoints', 'definitions']
  const kpis = overview.kpis || {}
  const charts = overview.charts || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note} &middot; Updated {overview.updated_at}</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#64748b'
          }}>
            {t === 'overview' ? 'Overview' : t === 'tabs' ? 'Portal Tabs' : t === 'endpoints' ? 'Endpoints' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Tabs" value={kpis.total_tabs} />
              <KPI label="Built" value={kpis.built} sub="live & verified" />
              <KPI label="Planned" value={kpis.planned} />
              <KPI label="Partial" value={kpis.partial} />
              <KPI label="Total Endpoints" value={kpis.total_endpoints} />
            </div>
          </Card>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            <Card title="Tab Status Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Endpoints per Tab">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={charts.endpoints_per_tab || []} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-25} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Purpose Description Length">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={charts.purpose_lengths || []} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-25} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Summary table */}
          <Card title="All Portal Tabs">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Label</th>
                    <th style={thStyle}>Purpose</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Maps To</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.summary_table || []).map((r, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{r.label}</td>
                      <td style={{ ...tdStyle, maxWidth: 350 }}>{r.purpose}</td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                      <td style={{ ...tdStyle, fontSize: 10, color: '#94a3b8', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.maps_to}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── PORTAL TABS TAB ── */}
      {tab === 'tabs' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
            {(breakdown.tabs || []).map((t, i) => (
              <Card key={i} title={t.label}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <StatusBadge status={t.status} />
                  <span style={{ fontSize: 11, color: '#94a3b8' }}>ID: {t.id}</span>
                </div>
                <p style={{ fontSize: 12, color: '#475569', margin: '8px 0', lineHeight: 1.5 }}>{t.purpose}</p>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
                  <strong>Endpoints ({t.endpoint_count}):</strong>
                  <div style={{ marginTop: 4 }}>
                    {(t.endpoints || []).map((ep, j) => (
                      <span key={j} style={{
                        display: 'inline-block', background: '#eff6ff', color: '#3b82f6',
                        borderRadius: 4, padding: '2px 6px', fontSize: 10, marginRight: 4, marginBottom: 2,
                        fontFamily: 'monospace'
                      }}>{ep}</span>
                    ))}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* ── ENDPOINTS TAB ── */}
      {tab === 'endpoints' && breakdown && (
        <>
          <Card title="Endpoint Mapping">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Tab</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Endpoint Count</th>
                    <th style={thStyle}>Endpoints</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.tabs || []).map((t, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{t.label}</td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>{t.endpoint_count}</td>
                      <td style={tdStyle}>
                        {(t.endpoints || []).map((ep, j) => (
                          <div key={j} style={{ fontSize: 10, fontFamily: 'monospace', color: '#3b82f6', marginBottom: 2 }}>{ep}</div>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* By status grouping */}
          {Object.entries(breakdown.by_status || {}).map(([status, items]) => (
            <Card key={status} title={`${status.charAt(0).toUpperCase() + status.slice(1)} Tabs (${items.length})`}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {items.map((t, i) => (
                  <div key={i} style={{
                    background: `${STATUS_COLORS[status] || '#94a3b8'}11`,
                    border: `1px solid ${STATUS_COLORS[status] || '#94a3b8'}33`,
                    borderRadius: 6, padding: '8px 12px', fontSize: 12
                  }}>
                    <strong>{t.label}</strong>
                    <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{t.id}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <strong style={{ fontSize: 12, color: '#1e293b' }}>{g.term}</strong>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
              {(defs.references || []).map((r, i) => (
                <div key={i} style={{ fontSize: 12, padding: '4px 0' }}>
                  <strong style={{ color: '#1e293b' }}>{r.label}</strong>
                  <span style={{ color: '#94a3b8' }}> — {r.note}</span>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
