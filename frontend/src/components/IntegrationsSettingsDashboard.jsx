import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', 'needs-credentials': '#ef4444', 'needs credentials': '#ef4444', planned: '#94a3b8', unknown: '#cbd5e1' }
const CAT_ICONS = { storage: '\u{1F4C1}', messaging: '\u{1F4AC}', email: '\u{2709}\uFE0F' }

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
  const key = (status || '').toLowerCase()
  const bg = STATUS_COLORS[key] || '#94a3b8'
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

export default function IntegrationsSettingsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/integrations-settings/overview`),
      axios.get(`${API_URL}/integrations-settings/breakdown`),
      axios.get(`${API_URL}/integrations-settings/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Integrations Settings...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#94a3b8' }}>{overview.note}</div>

  const tabs = ['overview', 'integrations', 'channels', 'definitions']
  const kpis = overview.kpis || {}
  const charts = overview.charts || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note} &middot; Updated {overview.updated_at}</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t ? 700 : 500, cursor: 'pointer',
            border: 'none', borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t ? '#1e293b' : '#64748b', marginBottom: -2
          }}>
            {t === 'overview' ? 'Overview' : t === 'integrations' ? 'Integrations' : t === 'channels' ? 'Delivery Channels' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Items" value={kpis.total} />
              <KPI label="Integrations" value={kpis.integrations} />
              <KPI label="Delivery Channels" value={kpis.delivery_channels} />
              <KPI label="Built" value={kpis.built} sub="live" />
              <KPI label="Partial" value={kpis.partial} />
              <KPI label="Needs Credentials" value={kpis.needs_credentials} sub="adapter ready" />
              <KPI label="Planned" value={kpis.planned} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Status distribution pie */}
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Category distribution pie */}
            <Card title="Integration Categories">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.category_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.category_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Summary table */}
          {breakdown && (
            <Card title="All Integrations & Channels" span={2}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Name</th>
                      <th style={thStyle}>Type</th>
                      <th style={thStyle}>Purpose</th>
                      <th style={thStyle}>Status</th>
                      <th style={thStyle}>Config</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.integrations || []).map(item => (
                      <tr key={item.id}>
                        <td style={tdStyle}><strong>{item.name}</strong></td>
                        <td style={tdStyle}>{CAT_ICONS[item.category] || ''} {item.category}</td>
                        <td style={tdStyle}>{item.purpose}</td>
                        <td style={tdStyle}><StatusBadge status={item.status} /></td>
                        <td style={{ ...tdStyle, fontSize: 11, fontFamily: 'monospace', color: '#64748b' }}>{item.config}</td>
                      </tr>
                    ))}
                    {(breakdown.delivery_channels || []).map(item => (
                      <tr key={item.id}>
                        <td style={tdStyle}><strong>{item.name}</strong></td>
                        <td style={tdStyle}>channel</td>
                        <td style={tdStyle}>{item.purpose}</td>
                        <td style={tdStyle}><StatusBadge status={item.status} /></td>
                        <td style={{ ...tdStyle, fontSize: 11, fontFamily: 'monospace', color: '#64748b' }}>{item.config}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Honest note */}
          {overview.honest_note && (
            <Card title="Honest Status Note">
              <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.6, margin: 0 }}>{overview.honest_note}</p>
            </Card>
          )}
        </>
      )}

      {/* INTEGRATIONS TAB */}
      {tab === 'integrations' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340, 1fr))', gap: 16 }}>
          {(breakdown.integrations || []).map(item => (
            <Card key={item.id} title={`${CAT_ICONS[item.category] || ''} ${item.name}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <StatusBadge status={item.status} />
                <span style={{ fontSize: 11, color: '#94a3b8', textTransform: 'uppercase' }}>{item.category}</span>
              </div>
              <p style={{ fontSize: 12, color: '#475569', margin: '0 0 8px', lineHeight: 1.5 }}>{item.purpose}</p>
              <div style={{ fontSize: 11, color: '#64748b', background: '#f8fafc', padding: '6px 10px', borderRadius: 4, fontFamily: 'monospace' }}>
                {item.config}
              </div>
              {item.scope && (
                <div style={{ marginTop: 6 }}>
                  <span style={{
                    background: '#eff6ff', color: '#3b82f6', border: '1px solid #bfdbfe',
                    borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 600
                  }}>
                    scope: {item.scope}
                  </span>
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* DELIVERY CHANNELS TAB */}
      {tab === 'channels' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340, 1fr))', gap: 16 }}>
          {(breakdown.delivery_channels || []).map(item => (
            <Card key={item.id} title={item.name}>
              <div style={{ marginBottom: 10 }}>
                <StatusBadge status={item.status} />
              </div>
              <p style={{ fontSize: 12, color: '#475569', margin: '0 0 8px', lineHeight: 1.5 }}>{item.purpose}</p>
              <div style={{ fontSize: 11, color: '#64748b', background: '#f8fafc', padding: '6px 10px', borderRadius: 4, fontFamily: 'monospace' }}>
                {item.config}
              </div>
              {item.note && (
                <p style={{ fontSize: 11, color: '#94a3b8', margin: '8px 0 0', fontStyle: 'italic' }}>{item.note}</p>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          {/* Status legend */}
          <Card title="Status Legend">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
              {(defs.status_legend || []).map(s => (
                <div key={s.status} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map(g => (
                  <tr key={g.term}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical notes */}
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', lineHeight: 1.6, marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          {/* References */}
          <Card title="References">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Source</th>
                  <th style={thStyle}>Note</th>
                </tr>
              </thead>
              <tbody>
                {(defs.references || []).map(r => (
                  <tr key={r.label}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.label}</td>
                    <td style={tdStyle}>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  )
}
