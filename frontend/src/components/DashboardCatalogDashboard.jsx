import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308']
const STATUS_COLORS = { Built: '#22c55e', Partial: '#f97316', Planned: '#94a3b8', Unknown: '#cbd5e1' }

const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 13, fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155' }

function Card({ title, children, span }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)', gridColumn: span ? `span ${span}` : undefined }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const s = (status || 'unknown').charAt(0).toUpperCase() + (status || 'unknown').slice(1)
  const color = STATUS_COLORS[s] || STATUS_COLORS.Unknown
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{s}</span>
}

export default function DashboardCatalogDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [phaseFilter, setPhaseFilter] = useState('all')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/dashboard-catalog/overview`),
      axios.get(`${API}/dashboard-catalog/breakdown`),
      axios.get(`${API}/dashboard-catalog/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Dashboard Catalog...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#f97316' }}>{overview.note || 'Not available'}</div>

  const tabs = ['overview', 'by-phase', 'all-dashboards', 'viz-vocabulary', 'definitions']
  const k = overview.kpis || {}
  const allDashboards = overview.dashboards_summary || []
  const bdPhases = (breakdown && breakdown.phases) || []
  const bdExtra = (breakdown && breakdown.additional) || []

  const filteredDashboards = phaseFilter === 'all'
    ? allDashboards
    : allDashboards.filter(d => String(d.phase) === phaseFilter || d.phase_name === phaseFilter)

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{overview.title || 'Dashboard Catalog'}</h2>
      <p style={{ margin: '0 0 18px', fontSize: 13, color: '#64748b' }}>Enterprise AI dashboard coverage map — 5 phases, all dashboards, status tracking</p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 500, background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569', transition: 'all .15s' }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Dashboards" value={k.total_dashboards} />
              <KPI label="Built" value={k.built} sub={`${k.total_dashboards ? Math.round(k.built / k.total_dashboards * 100) : 0}%`} />
              <KPI label="Partial" value={k.partial} />
              <KPI label="Planned" value={k.planned} />
              <KPI label="Phases" value={k.total_phases} />
              <KPI label="Viz Types" value={k.visualization_types} />
            </div>
          </Card>

          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.status_distribution || []).map((e, i) => <Cell key={i} fill={STATUS_COLORS[e.name] || COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Dashboards per Phase">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.phase_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {(overview.phase_distribution || []).map((e, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Dashboards Summary" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Dashboard</th>
                  <th style={thStyle}>Phase</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Maps To</th>
                </tr></thead>
                <tbody>
                  {allDashboards.slice(0, 40).map((d, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{d.name}</td>
                      <td style={tdStyle}>{d.phase_name || `Phase ${d.phase}`}</td>
                      <td style={tdStyle}><StatusBadge status={d.status} /></td>
                      <td style={{ ...tdStyle, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.maps_to || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {allDashboards.length > 40 && <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing 40 of {allDashboards.length} — see All Dashboards tab for full list</p>}
            </div>
          </Card>
        </div>
      )}

      {tab === 'by-phase' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {bdPhases.map(ph => (
            <Card key={ph.phase} title={`Phase ${ph.phase}: ${ph.name}`}>
              <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 13, color: '#64748b' }}>{ph.dashboards.length} dashboards</span>
                <span style={{ fontSize: 13, color: '#22c55e', fontWeight: 600 }}>{ph.built} built</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
                {ph.dashboards.map((d, i) => (
                  <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <strong style={{ fontSize: 13, color: '#1e293b' }}>{d.name}</strong>
                      <StatusBadge status={d.status} />
                    </div>
                    {d.maps_to && <div style={{ fontSize: 11, color: '#64748b', wordBreak: 'break-all' }}>{d.maps_to}</div>}
                    {d.component && <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 4 }}>{d.component}</div>}
                    {d.note && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{d.note}</div>}
                  </div>
                ))}
              </div>
            </Card>
          ))}
          {bdExtra.length > 0 && (
            <Card title="Additional Dashboards">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
                {bdExtra.map((d, i) => (
                  <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <strong style={{ fontSize: 13, color: '#1e293b' }}>{d.name}</strong>
                      <StatusBadge status={d.status} />
                    </div>
                    {d.category && <span style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 9999, fontSize: 10, background: '#ede9fe', color: '#7c3aed', marginBottom: 4 }}>{d.category}</span>}
                    {d.description && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{d.description}</div>}
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === 'all-dashboards' && (
        <Card title={`All Dashboards (${filteredDashboards.length})`}>
          <div style={{ marginBottom: 12 }}>
            <select value={phaseFilter} onChange={e => setPhaseFilter(e.target.value)} style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 13 }}>
              <option value="all">All Phases</option>
              {(overview.phase_distribution || []).map((p, i) => <option key={i} value={String(i + 1)}>{p.name}</option>)}
              <option value="0">Additional</option>
            </select>
          </div>
          <div style={{ maxHeight: 600, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>Dashboard</th>
                <th style={thStyle}>Phase</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Maps To</th>
              </tr></thead>
              <tbody>
                {filteredDashboards.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}>{i + 1}</td>
                    <td style={tdStyle}>{d.name}</td>
                    <td style={tdStyle}>{d.phase_name || `Phase ${d.phase}`}</td>
                    <td style={tdStyle}><StatusBadge status={d.status} /></td>
                    <td style={tdStyle}>{d.category || '-'}</td>
                    <td style={{ ...tdStyle, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.maps_to || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'viz-vocabulary' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title={`Visualization Types (${(overview.visualization_vocabulary || []).length})`} span={2}>
            <p style={{ fontSize: 13, color: '#64748b', marginBottom: 12 }}>Chart and visualization types available across all dashboards</p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {(overview.visualization_vocabulary || []).map((v, i) => (
                <span key={i} style={{ display: 'inline-block', padding: '6px 14px', borderRadius: 8, fontSize: 13, fontWeight: 500, background: COLORS[i % COLORS.length] + '18', color: COLORS[i % COLORS.length], border: `1px solid ${COLORS[i % COLORS.length]}30` }}>{v}</span>
              ))}
            </div>
          </Card>
          {overview.category_distribution && overview.category_distribution.length > 0 && (
            <Card title="Additional Dashboards by Category">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.category_distribution} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                    {(overview.category_distribution || []).map((e, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          <Card title="Status Legend">
            {(defs.status_legend || []).map((l, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{ width: 14, height: 14, borderRadius: 4, background: l.color, flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{l.label}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>— {l.description}</span>
              </div>
            ))}
          </Card>
          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 6 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: 8, borderBottom: '1px solid #f1f5f9' }}>
                  <strong style={{ fontSize: 13, color: '#1e293b' }}>{g.term}</strong>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6 }}>{n}</li>)}
            </ul>
          </Card>
          <Card title="References">
            {(defs.references || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <strong style={{ fontSize: 13, color: '#1e293b' }}>{r.ref}</strong>
                <div style={{ fontSize: 12, color: '#64748b' }}>{r.detail}</div>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
