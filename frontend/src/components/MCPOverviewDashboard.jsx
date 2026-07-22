import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function MCPOverviewDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/mcp-overview/overview`),
          axios.get(`${API_URL}/api/mcp-overview/breakdown`),
          axios.get(`${API_URL}/api/mcp-overview/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load MCP overview data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading MCP overview data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'MCP overview data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const compHealth = overview.component_health || []
  const actionCatalog = overview.action_catalog || []
  const actorSummary = overview.actor_summary || []
  const dailyActivity = overview.daily_activity || []
  const hourlyHeatmap = overview.hourly_heatmap || []
  const compliance = overview.protocol_compliance || {}

  const compActionMatrix = breakdown?.component_action_matrix || []
  const convRoles = breakdown?.conversation_roles || []
  const convComps = breakdown?.conversation_components || []
  const patientCov = breakdown?.patient_coverage || {}
  const recentEvents = breakdown?.recent_events || []
  const secAudit = breakdown?.security_audit_log || []
  const compInterconn = breakdown?.component_interconnections || []
  const definitions = defs?.metrics || []

  const cardStyle = { background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', marginBottom: 18 }
  const kpiStyle = { background: '#f8fafc', borderRadius: 10, padding: '14px 18px', minWidth: 140, textAlign: 'center' }
  const sectionTitle = { fontSize: 15, fontWeight: 700, color: '#1e293b', marginBottom: 12 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: 'none', fontSize: 13, marginRight: 4
  })
  const thStyle = { padding: '8px 12px', textAlign: 'left', fontSize: 12, color: '#64748b', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '8px 12px', fontSize: 13, color: '#334155', borderBottom: '1px solid #f1f5f9' }

  const kpiItems = [
    { label: 'Components', value: fmt(s.total_components) },
    { label: 'Transactions', value: fmt(s.total_transactions) },
    { label: 'Conversations', value: fmt(s.total_conversations) },
    { label: 'Analyses', value: fmt(s.total_analyses) },
    { label: 'Actors', value: fmt(s.total_actors) },
    { label: 'Actions', value: fmt(s.total_actions) },
    { label: 'Compliance Rate', value: `${fmt(s.compliance_rate)}%` },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Components & Actions' },
    { id: 'security', label: 'Security & Compliance' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>MCP Overview Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>real clinical.db MCP system analytics</span>
      </div>

      {/* Tab bar */}
      <div style={{ marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* === OVERVIEW TAB === */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((k, i) => (
              <div key={i} style={kpiStyle}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Charts row: Daily Activity + Hourly Heatmap */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Daily Activity (last 30 days)</h4>
              {dailyActivity.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={dailyActivity}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Line type="monotone" dataKey="transactions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Transactions" />
                    <Line type="monotone" dataKey="conversations" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Conversations" />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No daily data</div>}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Hourly Activity Pattern</h4>
              {hourlyHeatmap.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={hourlyHeatmap}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" fontSize={11} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="transactions" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No hourly data</div>}
            </div>
          </div>

          {/* Component Health Table */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Component Health</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Transactions</th>
                    <th style={thStyle}>Last Active</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {compHealth.map((c, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={tdStyle}><strong>{c.component}</strong></td>
                      <td style={tdStyle}>{fmt(c.transactions)}</td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{c.last_active || '--'}</td>
                      <td style={tdStyle}>
                        <span style={{
                          padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                          background: c.status === 'active' ? '#dcfce7' : '#fef3c7',
                          color: c.status === 'active' ? '#166534' : '#92400e'
                        }}>
                          {c.status || 'unknown'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Actor Summary */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Actor Summary</h4>
            {actorSummary.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={actorSummary}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="actor" fontSize={10} angle={-20} textAnchor="end" height={50} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="transactions" fill="#10b981" radius={[4, 4, 0, 0]}>
                    {actorSummary.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
          </div>
        </>
      )}

      {/* === COMPONENTS & ACTIONS TAB === */}
      {tab === 'components' && (
        <>
          {/* Action Catalog */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Action Catalog</h4>
            {actionCatalog.length > 0 ? (
              <ResponsiveContainer width="100%" height={Math.max(280, actionCatalog.length * 22)}>
                <BarChart data={actionCatalog} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={11} />
                  <YAxis dataKey="action" type="category" fontSize={10} width={130} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                    {actionCatalog.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
          </div>

          {/* Component x Action Matrix */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Component x Action Matrix</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {compActionMatrix.map((v, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={tdStyle}><strong>{v.component}</strong></td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>
                        {Object.entries(v.actions || {}).map(([a, c]) => `${a}(${c})`).join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Component Interconnections */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Component Interconnections</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Unique Patients</th>
                    <th style={thStyle}>Unique Actors</th>
                    <th style={thStyle}>Unique Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {compInterconn.map((c, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={tdStyle}><strong>{c.component}</strong></td>
                      <td style={tdStyle}>{fmt(c.unique_patients)}</td>
                      <td style={tdStyle}>{fmt(c.unique_actors)}</td>
                      <td style={tdStyle}>{fmt(c.unique_actions)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Conversation Roles + Components */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Conversation Roles</h4>
              {convRoles.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={convRoles} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                      {convRoles.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Conversation by Component</h4>
              {convComps.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={convComps}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="component" fontSize={10} angle={-20} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]}>
                      {convComps.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
            </div>
          </div>
        </>
      )}

      {/* === SECURITY & COMPLIANCE TAB === */}
      {tab === 'security' && (
        <>
          {/* Protocol Compliance KPIs */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {[
              { label: 'Total Transactions', value: fmt(compliance.total_transactions) },
              { label: 'Guardrail Events', value: fmt(compliance.guardrail_events) },
              { label: 'Security Events', value: fmt(compliance.security_events) },
              { label: 'Audit Events', value: fmt(compliance.audit_events) },
              { label: 'Compliance Rate', value: `${fmt(compliance.compliance_rate_pct)}%` },
            ].map((k, i) => (
              <div key={i} style={kpiStyle}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Patient Coverage */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Patient Coverage</h4>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {[
                { label: 'Total Patients', value: fmt(patientCov.total_patients) },
                { label: 'With Conversations', value: fmt(patientCov.patients_with_conversations) },
                { label: 'With Analyses', value: fmt(patientCov.patients_with_analyses) },
                { label: 'With Reviews', value: fmt(patientCov.patients_with_reviews) },
              ].map((k, i) => (
                <div key={i} style={{ ...kpiStyle, flex: 1 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                  <div style={{ fontSize: 20, fontWeight: 700, color: '#1e293b' }}>{k.value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Security Audit Log */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Security & Guardrail Audit Log</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Time</th>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Action</th>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {secAudit.length > 0 ? secAudit.map((e, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{e.created_at || '--'}</td>
                      <td style={tdStyle}>{e.patient_id || '--'}</td>
                      <td style={tdStyle}>{e.component}</td>
                      <td style={tdStyle}>
                        <span style={{ padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600, background: '#fef3c7', color: '#92400e' }}>
                          {e.action}
                        </span>
                      </td>
                      <td style={tdStyle}>{e.actor}</td>
                      <td style={{ ...tdStyle, fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.details || '--'}</td>
                    </tr>
                  )) : (
                    <tr><td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: '#94a3b8' }}>No security/guardrail events found</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent Events */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Recent System Events</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Time</th>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Action</th>
                    <th style={thStyle}>Actor</th>
                  </tr>
                </thead>
                <tbody>
                  {recentEvents.map((e, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{e.created_at || '--'}</td>
                      <td style={tdStyle}>{e.patient_id || '--'}</td>
                      <td style={tdStyle}>{e.component}</td>
                      <td style={tdStyle}>{e.action}</td>
                      <td style={tdStyle}>{e.actor}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* === DEFINITIONS TAB === */}
      {tab === 'definitions' && (
        <div style={cardStyle}>
          <h4 style={sectionTitle}>Metric Definitions</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Metric</th>
                <th style={thStyle}>Definition</th>
                <th style={thStyle}>Source</th>
              </tr>
            </thead>
            <tbody>
              {definitions.map((d, i) => (
                <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{d.name}</td>
                  <td style={tdStyle}>{d.definition}</td>
                  <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{d.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
