import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function MCPFederationDashboard() {
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
          axios.get(`${API_URL}/mcp-federation/overview`),
          axios.get(`${API_URL}/mcp-federation/breakdown`),
          axios.get(`${API_URL}/mcp-federation/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load MCP federation data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading MCP federation data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'MCP federation data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const nodeDetails = overview.node_details || []
  const nodeClassDist = overview.node_class_distribution || []
  const actorDist = overview.actor_distribution || []
  const actionDist = overview.action_distribution || []
  const dailyTrend = overview.daily_trend || []
  const hourlyPattern = overview.hourly_pattern || []
  const topEdges = overview.top_federation_edges || []

  const patientFed = breakdown?.patient_federation || []
  const meshLinks = breakdown?.mesh_links || []
  const verbMatrix = breakdown?.verb_matrix || []
  const componentPairs = breakdown?.component_pairs || []
  const actorComp = breakdown?.actor_component_crosstab || []
  const recentEvents = breakdown?.recent_events || []
  const diseaseFed = breakdown?.disease_federation || []
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
    { label: 'Federation Nodes', value: fmt(s.total_nodes) },
    { label: 'Federation Edges', value: fmt(s.total_edges) },
    { label: 'Transactions', value: fmt(s.total_transactions) },
    { label: 'Cross-Component Rate', value: `${fmt(s.cross_component_rate_pct)}%` },
    { label: 'Oversight Rate', value: `${fmt(s.oversight_rate_pct)}%` },
    { label: 'Actors', value: fmt(s.total_actors) },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'topology', label: 'Topology & Mesh' },
    { id: 'events', label: 'Events & Protocols' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>MCP Federation Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>real clinical.db cross-component federation analytics</span>
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

          {/* Charts row: Node Class Pie + Daily Trend Line */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Node Class Distribution</h4>
              {nodeClassDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={nodeClassDist} dataKey="count" nameKey="class" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                      {nodeClassDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Daily Federation Volume</h4>
              {dailyTrend.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={dailyTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Line type="monotone" dataKey="transactions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No daily data</div>}
            </div>
          </div>

          {/* Charts row: Node Volume Bar + Actor Distribution Bar */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Node Transaction Volume</h4>
              {nodeDetails.length > 0 ? (
                <ResponsiveContainer width="100%" height={Math.max(260, nodeDetails.length * 22)}>
                  <BarChart data={nodeDetails} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" fontSize={11} />
                    <YAxis dataKey="node" type="category" fontSize={10} width={130} />
                    <Tooltip />
                    <Bar dataKey="transactions" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                      {nodeDetails.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Actor Distribution</h4>
              {actorDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={actorDist}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="actor" fontSize={10} angle={-20} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="transactions" fill="#10b981" radius={[4, 4, 0, 0]}>
                      {actorDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
            </div>
          </div>

          {/* Hourly Pattern */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Hourly Federation Pattern</h4>
            {hourlyPattern.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={hourlyPattern}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No hourly data</div>}
          </div>
        </>
      )}

      {/* === TOPOLOGY & MESH TAB === */}
      {tab === 'topology' && (
        <>
          {/* Top federation edges (service mesh links) */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Service Mesh — Top Federation Links</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Source</th>
                  <th style={thStyle}>Target</th>
                  <th style={thStyle}>Shared Patients</th>
                  <th style={thStyle}>Federation Type</th>
                </tr>
              </thead>
              <tbody>
                {componentPairs.map((p, i) => (
                  <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                    <td style={tdStyle}>{p.source}</td>
                    <td style={tdStyle}>{p.target}</td>
                    <td style={tdStyle}>{p.shared_patients}</td>
                    <td style={tdStyle}><span style={{ fontSize: 11, color: '#6366f1' }}>{p.federation_type}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-patient federation profile */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Patient Federation Depth</h4>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Components</th>
                    <th style={thStyle}>Transactions</th>
                    <th style={thStyle}>Actors</th>
                    <th style={thStyle}>Component List</th>
                  </tr>
                </thead>
                <tbody>
                  {patientFed.map((p, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={tdStyle}>{p.name}</td>
                      <td style={tdStyle}>{p.disease || '--'}</td>
                      <td style={tdStyle}><strong>{p.components_touched}</strong></td>
                      <td style={tdStyle}>{p.total_transactions}</td>
                      <td style={tdStyle}>{(p.actors_involved || []).length}</td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{(p.component_list || []).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Disease federation depth */}
          {diseaseFed.length > 0 && (
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Disease Federation Depth</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Patients</th>
                    <th style={thStyle}>Unique Components</th>
                    <th style={thStyle}>Total Transactions</th>
                    <th style={thStyle}>Avg Components/Patient</th>
                  </tr>
                </thead>
                <tbody>
                  {diseaseFed.map((d, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{d.disease}</td>
                      <td style={tdStyle}>{d.patients}</td>
                      <td style={tdStyle}>{d.unique_components}</td>
                      <td style={tdStyle}>{d.total_transactions}</td>
                      <td style={tdStyle}>{d.avg_components_per_patient}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Mesh adjacency (full list) */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Full Mesh Adjacency</h4>
            {meshLinks.length > 0 ? (
              <ResponsiveContainer width="100%" height={Math.max(280, meshLinks.slice(0, 20).length * 22)}>
                <BarChart data={meshLinks.slice(0, 20)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={11} />
                  <YAxis dataKey={(d) => `${d.source} — ${d.target}`} type="category" fontSize={9} width={220} />
                  <Tooltip />
                  <Bar dataKey="weight" fill="#06b6d4" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No mesh data</div>}
          </div>
        </>
      )}

      {/* === EVENTS & PROTOCOLS TAB === */}
      {tab === 'events' && (
        <>
          {/* Action (protocol verb) distribution */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Protocol Verb Distribution</h4>
            {actionDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={Math.max(260, actionDist.length * 22)}>
                <BarChart data={actionDist} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={11} />
                  <YAxis dataKey="action" type="category" fontSize={11} width={120} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]}>
                    {actionDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No data</div>}
          </div>

          {/* Verb matrix: component × action */}
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
                  {verbMatrix.map((v, i) => (
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

          {/* Actor × component cross-tab */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Actor x Component Cross-tab</h4>
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Components</th>
                  </tr>
                </thead>
                <tbody>
                  {actorComp.map((a, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={tdStyle}><strong>{a.actor}</strong></td>
                      <td style={{ ...tdStyle, fontSize: 11 }}>
                        {Object.entries(a.components || {}).map(([c, n]) => `${c}(${n})`).join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent federation events */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Recent Federation Events</h4>
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
                  {recentEvents.map((e, i) => (
                    <tr key={i} style={i % 2 === 0 ? {} : { background: '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontSize: 11 }}>{e.timestamp || '--'}</td>
                      <td style={tdStyle}>{e.patient_id || '--'}</td>
                      <td style={tdStyle}>{e.component}</td>
                      <td style={tdStyle}>{e.action}</td>
                      <td style={tdStyle}>{e.actor}</td>
                      <td style={{ ...tdStyle, fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || '--'}</td>
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
