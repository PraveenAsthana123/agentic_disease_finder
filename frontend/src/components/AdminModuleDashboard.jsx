import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#94a3b8', 'n/a': '#cbd5e1' }

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

export default function AdminModuleDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/admin-module/overview`),
      axios.get(`${API_URL}/admin-module/breakdown`),
      axios.get(`${API_URL}/admin-module/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Admin Module...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#94a3b8' }}>{overview.note}</div>

  const tabs = ['overview', 'roles', 'ops', 'access', 'integrations', 'definitions']
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
            {t === 'overview' ? 'Overview' : t === 'roles' ? 'Team Roles' : t === 'ops' ? 'Ops Dashboards' :
             t === 'access' ? 'Access Control' : t === 'integrations' ? 'Integrations' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Team Roles" value={kpis.total_roles} sub={`${kpis.built_roles} built`} />
              <KPI label="Ops Dashboards" value={kpis.total_ops_dashboards} sub={`${kpis.built_ops} built`} />
              <KPI label="Access Control" value={kpis.total_access_control} sub={`${kpis.built_acl} built`} />
              <KPI label="Integrations" value={kpis.total_integrations} sub={`${kpis.planned_integrations} planned`} />
            </div>
          </Card>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
            <Card title="Role Status Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.role_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.role_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Responsibilities per Role">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={charts.responsibilities_per_role || []} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Integration Status">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.integration_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.integration_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Summary table */}
          <Card title="All Team Roles">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Icon</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Owns</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Maps To</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.summary_table || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{r.icon}</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{r.role}</td>
                      <td style={tdStyle}>
                        {(r.owns || []).map((o, j) => (
                          <span key={j} style={{ background: '#eff6ff', color: '#3b82f6', borderRadius: 4, padding: '1px 6px', fontSize: 10, marginRight: 4, display: 'inline-block', marginBottom: 2 }}>{o}</span>
                        ))}
                      </td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                      <td style={{ ...tdStyle, fontSize: 10, color: '#94a3b8', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.maps_to}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── ROLES TAB ── */}
      {tab === 'roles' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {(breakdown.team_roles || []).map((r, i) => (
              <Card key={i} title={`${r.icon || ''} ${r.role}`}>
                <div style={{ marginBottom: 8 }}><StatusBadge status={r.status} /></div>
                <div style={{ fontSize: 12, color: '#334155', marginBottom: 8 }}>
                  <strong>Owns:</strong>
                  <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(r.owns || []).map((o, j) => (
                      <span key={j} style={{ background: '#eff6ff', color: '#3b82f6', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{o}</span>
                    ))}
                  </div>
                </div>
                {r.maps_to && (
                  <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 8, wordBreak: 'break-all' }}>
                    <strong>Maps to:</strong> {r.maps_to}
                  </div>
                )}
              </Card>
            ))}
          </div>
        </>
      )}

      {/* ── OPS DASHBOARDS TAB ── */}
      {tab === 'ops' && breakdown && (
        <>
          <Card title="Ops Dashboard Status">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={charts.ops_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                  {(charts.ops_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
            {(breakdown.ops_dashboards || []).map((d, i) => (
              <Card key={i} title={`${d.label || d.id}`}>
                <div style={{ marginBottom: 8 }}><StatusBadge status={d.status} /></div>
                <p style={{ fontSize: 12, color: '#64748b', margin: '8px 0' }}>{d.purpose}</p>
                {d.maps_to && (
                  <div style={{ fontSize: 10, color: '#94a3b8', wordBreak: 'break-all' }}>
                    <strong>Maps to:</strong> {d.maps_to}
                  </div>
                )}
              </Card>
            ))}
          </div>
        </>
      )}

      {/* ── ACCESS CONTROL TAB ── */}
      {tab === 'access' && breakdown && (
        <>
          <Card title="Access Control Status">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={charts.access_control_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                  {(charts.access_control_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
            {(breakdown.access_control || []).map((a, i) => (
              <Card key={i} title={`${a.label || a.id}`}>
                <div style={{ marginBottom: 8 }}><StatusBadge status={a.status} /></div>
                <p style={{ fontSize: 12, color: '#64748b', margin: '8px 0' }}>{a.purpose}</p>
                {a.note && <p style={{ fontSize: 11, color: '#f97316', fontStyle: 'italic', margin: '4px 0' }}>{a.note}</p>}
                {a.maps_to && (
                  <div style={{ fontSize: 10, color: '#94a3b8', wordBreak: 'break-all' }}>
                    <strong>Maps to:</strong> {a.maps_to}
                  </div>
                )}
              </Card>
            ))}
          </div>
        </>
      )}

      {/* ── INTEGRATIONS TAB ── */}
      {tab === 'integrations' && breakdown && (
        <>
          {breakdown.integration_note && (
            <Card>
              <p style={{ fontSize: 12, color: '#64748b', margin: 0, fontStyle: 'italic' }}>{breakdown.integration_note}</p>
            </Card>
          )}

          <Card title="All Integrations">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Integration</th>
                    <th style={thStyle}>Via</th>
                    <th style={thStyle}>Purpose</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.integrations || []).map((ig, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{ig.label}</td>
                      <td style={tdStyle}>{ig.via}</td>
                      <td style={tdStyle}>{ig.purpose}</td>
                      <td style={tdStyle}><StatusBadge status={ig.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ fontSize: 11, color: '#64748b' }}>{s.meaning}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ overflowX: 'auto' }}>
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
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>{r}</li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  )
}
