import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308', '#06b6d4', '#f43f5e', '#84cc16', '#a855f7', '#fb923c']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#8b5cf6', unknown: '#94a3b8' }

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

export default function RoleDashboardsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [roleFilter, setRoleFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/role-dashboards/overview`),
      axios.get(`${API_URL}/role-dashboards/breakdown`),
      axios.get(`${API_URL}/role-dashboards/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Role Dashboards...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-role', 'reports', 'cadences', 'definitions']
  const kpis = overview.kpis || {}

  const roleNames = breakdown && breakdown.per_role ? breakdown.per_role.map(r => r.role) : []

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>
          Per-Role Dashboards & Reports
        </h2>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          {overview.note} &middot; Updated {overview.updated_at}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: 600, textTransform: 'capitalize',
            background: tab === t ? '#3b82f6' : '#e2e8f0',
            color: tab === t ? '#fff' : '#475569'
          }}>
            {t.replace(/-/g, ' ')}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={kpis.total_roles} />
              <KPI label="Total KPIs" value={kpis.total_kpis} />
              <KPI label="KPIs Built" value={kpis.kpis_built} />
              <KPI label="Total Reports" value={kpis.total_reports} />
              <KPI label="Reports Built" value={kpis.reports_built} />
            </div>
          </Card>

          <Card title="KPI Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.kpi_status_distribution || []} dataKey="count" nameKey="status"
                  cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                  {(overview.kpi_status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Report Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.report_status_distribution || []} dataKey="count" nameKey="status"
                  cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                  {(overview.report_status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="KPIs per Role" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.kpis_per_role || []} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="role" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Reports per Role" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.reports_per_role || []} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="role" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Summary Table */}
          <Card title="All Roles Summary" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>KPIs</th>
                    <th style={thStyle}>Reports</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Endpoints</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.per_role || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={tdStyle}>{r.icon} {r.role}</td>
                      <td style={tdStyle}>{(r.kpis || []).length}</td>
                      <td style={tdStyle}>{(r.reports || []).length}</td>
                      <td style={tdStyle}>
                        {r.dashboard_component ? <StatusBadge status="built" /> : <span style={{ color: '#94a3b8', fontSize: 11 }}>—</span>}
                      </td>
                      <td style={tdStyle}>{(r.api_endpoints || []).length || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── By Role Tab ── */}
      {tab === 'by-role' && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)}
              style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #cbd5e1', fontSize: 12 }}>
              <option value="all">All Roles</option>
              {roleNames.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: 16 }}>
            {(breakdown?.per_role || [])
              .filter(r => roleFilter === 'all' || r.role === roleFilter)
              .map((role, ri) => (
                <Card key={ri} title={`${role.icon} ${role.role}`}>
                  {/* Dashboard component + endpoints */}
                  <div style={{ marginBottom: 10, display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                    {role.dashboard_component && (
                      <span style={{
                        background: '#3b82f622', color: '#3b82f6', border: '1px solid #3b82f655',
                        borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 600
                      }}>
                        {role.dashboard_component}
                      </span>
                    )}
                    {(role.api_endpoints || []).map((ep, j) => (
                      <span key={j} style={{
                        background: '#f1f5f9', color: '#475569', borderRadius: 4,
                        padding: '2px 6px', fontSize: 10, fontFamily: 'monospace'
                      }}>
                        {ep}
                      </span>
                    ))}
                  </div>

                  {/* KPIs */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>
                      KPIs ({(role.kpis || []).length})
                    </div>
                    {(role.kpis || []).map((kpi, ki) => (
                      <div key={ki} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '4px 8px', background: ki % 2 === 0 ? '#f8fafc' : '#fff',
                        borderRadius: 4, marginBottom: 2
                      }}>
                        <div>
                          <div style={{ fontSize: 12, color: '#1e293b' }}>{kpi.label}</div>
                          <div style={{ fontSize: 10, color: '#94a3b8' }}>{kpi.source}</div>
                        </div>
                        <StatusBadge status={kpi.status} />
                      </div>
                    ))}
                  </div>

                  {/* Reports */}
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 6 }}>
                      Reports ({(role.reports || []).length})
                    </div>
                    {(role.reports || []).map((rpt, ri2) => (
                      <div key={ri2} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '4px 8px', background: ri2 % 2 === 0 ? '#f8fafc' : '#fff',
                        borderRadius: 4, marginBottom: 2
                      }}>
                        <div>
                          <div style={{ fontSize: 12, color: '#1e293b' }}>{rpt.name}</div>
                          <div style={{ fontSize: 10, color: '#94a3b8' }}>{rpt.cadence} &middot; {rpt.format}</div>
                        </div>
                        <StatusBadge status={rpt.status} />
                      </div>
                    ))}
                  </div>
                </Card>
              ))}
          </div>
        </div>
      )}

      {/* ── Reports Tab ── */}
      {tab === 'reports' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Report Format Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.format_distribution || []} dataKey="count" nameKey="format"
                  cx="50%" cy="50%" outerRadius={80} label={({ format, count }) => `${format}: ${count}`}>
                  {(overview.format_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Report Cadence Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.cadence_distribution || []} dataKey="count" nameKey="cadence"
                  cx="50%" cy="50%" outerRadius={80} label={({ cadence, count }) => `${cadence}: ${count}`}>
                  {(overview.cadence_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Reports Inventory" span={2}>
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Report Name</th>
                    <th style={thStyle}>Cadence</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    let idx = 0
                    return (breakdown?.per_role || []).flatMap(role =>
                      (role.reports || []).map(rpt => {
                        idx++
                        return (
                          <tr key={`${role.role}-${rpt.name}`}>
                            <td style={tdStyle}>{idx}</td>
                            <td style={tdStyle}>{role.icon} {role.role}</td>
                            <td style={tdStyle}>{rpt.name}</td>
                            <td style={tdStyle}>{rpt.cadence}</td>
                            <td style={tdStyle}>{rpt.format}</td>
                            <td style={tdStyle}><StatusBadge status={rpt.status} /></td>
                          </tr>
                        )
                      })
                    )
                  })()}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Cadences Tab ── */}
      {tab === 'cadences' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Cadence Definitions" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Cadence</th>
                    <th style={thStyle}>Meaning</th>
                    <th style={thStyle}>Report Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.cadence_legend || []).map((cl, i) => {
                    const count = (overview.cadence_distribution || []).find(c => c.cadence === cl.cadence)?.count || 0
                    return (
                      <tr key={i}>
                        <td style={tdStyle}>
                          <span style={{
                            background: '#3b82f622', color: '#3b82f6', borderRadius: 4,
                            padding: '2px 8px', fontSize: 11, fontWeight: 600
                          }}>
                            {cl.cadence}
                          </span>
                        </td>
                        <td style={tdStyle}>{cl.meaning}</td>
                        <td style={tdStyle}>{count}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Cadence Distribution Chart" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.cadence_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cadence" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card title="Status Legend">
            {(defs.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                <StatusBadge status={s.status} />
                <span style={{ fontSize: 12, color: '#475569' }}>{s.meaning}</span>
              </div>
            ))}
          </Card>

          <Card title="Glossary">
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
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                      <td style={tdStyle}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Reference</th>
                    <th style={thStyle}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.references || []).map((r, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{r.ref}</td>
                      <td style={tdStyle}>{r.detail}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
