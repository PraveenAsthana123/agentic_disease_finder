import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STAGE_COLORS = {
  created: '#3b82f6',
  approved: '#10b981',
  published: '#8b5cf6',
  expired: '#f59e0b',
  archived: '#64748b',
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function KnowledgeManagementDashboard() {
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
          axios.get(`${API_URL}/api/knowledge-mgmt/overview`),
          axios.get(`${API_URL}/api/knowledge-mgmt/breakdown`),
          axios.get(`${API_URL}/api/knowledge-mgmt/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load knowledge management data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128218;</div>
      Loading Knowledge Management data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Knowledge management data not available.'}
    </div>
  )

  const stageDist = overview.stage_distribution || []
  const typeDist = overview.type_distribution || []
  const sourceBkdn = overview.source_breakdown || []
  const activityTrend = overview.activity_trend || []
  const register = breakdown?.knowledge_register || []
  const patients = breakdown?.patient_profiles || []
  const lifecycle = breakdown?.lifecycle_events || []
  const stageFlow = breakdown?.stage_flow || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#4338ca' : '#f1f5f9', color: active ? '#fff' : '#64748b',
    border: 'none', fontSize: 13, marginRight: 4
  })

  const kpiItems = [
    { label: 'Total Items', value: overview.total_knowledge_items, color: '#3b82f6' },
    { label: 'Published', value: overview.published_count, color: '#8b5cf6' },
    { label: 'Approved', value: overview.approved_count, color: '#10b981' },
    { label: 'Created', value: overview.created_count, color: '#3b82f6' },
    { label: 'Publish Rate', value: `${overview.publish_rate_pct}%`, color: overview.publish_rate_pct >= 60 ? '#10b981' : '#f59e0b' },
    { label: 'Approval Rate', value: `${overview.approval_rate_pct}%`, color: overview.approval_rate_pct >= 75 ? '#10b981' : '#f59e0b' },
    { label: 'Patients', value: overview.patients_with_knowledge, color: '#06b6d4' },
    { label: 'Avg Confidence', value: overview.avg_confidence, color: overview.avg_confidence >= 0.7 ? '#10b981' : '#f59e0b' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 110
  })

  const stageBadge = (stage) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12,
    fontSize: 11, fontWeight: 600, color: '#fff',
    background: STAGE_COLORS[stage] || '#64748b',
  })

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Knowledge Management Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 18 }}>
        Knowledge lifecycle tracking: create, approve, publish, expiry, archive &mdash; from real clinical.db
      </p>

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 12, marginBottom: 20 }}>
        {kpiItems.map((kpi, i) => (
          <div key={i} style={kpiStyle(kpi.color)}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: kpi.color }}>{fmt(kpi.value)}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', marginBottom: 0 }}>
        {['overview', 'knowledge register', 'patient profiles', 'lifecycle log', 'definitions'].map(t => (
          <button key={t} onClick={() => setTab(t)} style={tabStyle(tab === t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <div>
          {/* Stage Distribution + Type Distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginTop: 18 }}>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Stage Distribution</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={stageDist} dataKey="count" nameKey="stage" cx="50%" cy="50%"
                    outerRadius={90} label={({ stage, count }) => `${stage}: ${count}`}>
                    {stageDist.map((d, i) => (
                      <Cell key={i} fill={STAGE_COLORS[d.stage] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Knowledge Type Distribution</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={typeDist} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="type" type="category" width={140} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#4338ca" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Activity Trend + Source Breakdown */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 18, marginTop: 18 }}>
            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Daily Lifecycle Activity</h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={activityTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="events" stroke="#4338ca" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Source Breakdown</h3>
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={sourceBkdn} dataKey="count" nameKey="source" cx="50%" cy="50%"
                    outerRadius={80} label={({ source, count }) => `${source}: ${count}`}>
                    {sourceBkdn.map((d, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Stage Flow */}
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Stage Flow (Action &rarr; Stage)</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stageFlow.slice(0, 15)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="transition" tick={{ fontSize: 9, angle: -30 }} interval={0} height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {tab === 'knowledge register' && (
        <div style={{ ...cardStyle, marginTop: 18 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
            Knowledge Register ({breakdown?.total_register_items || 0} items)
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Title</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>Stage</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Confidence</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Created</th>
                </tr>
              </thead>
              <tbody>
                {register.map((item, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{item.id}</td>
                    <td style={{ padding: 8 }}>{item.type}</td>
                    <td style={{ padding: 8, maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.title}
                    </td>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{item.patient_id || '--'}</td>
                    <td style={{ padding: 8, textAlign: 'center' }}>
                      <span style={stageBadge(item.stage)}>{item.stage}</span>
                    </td>
                    <td style={{ padding: 8, textAlign: 'right' }}>
                      {item.confidence != null ? item.confidence.toFixed(2) : '--'}
                    </td>
                    <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{(item.created_at || '--').slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'patient profiles' && (
        <div style={{ ...cardStyle, marginTop: 18 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
            Patient Knowledge Profiles ({patients.length} patients)
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Patient ID</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Total Items</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Knowledge Types</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Stages Present</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{p.patient_id}</td>
                    <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{p.total_items}</td>
                    <td style={{ padding: 8 }}>
                      {(p.types || []).map((t, j) => (
                        <span key={j} style={{
                          display: 'inline-block', padding: '1px 8px', borderRadius: 10,
                          fontSize: 10, background: '#e0e7ff', color: '#3730a3', margin: '1px 2px',
                        }}>{t}</span>
                      ))}
                    </td>
                    <td style={{ padding: 8 }}>
                      {(p.stages || []).map((s, j) => (
                        <span key={j} style={stageBadge(s)}>{s}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'lifecycle log' && (
        <div style={{ ...cardStyle, marginTop: 18 }}>
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>
            Lifecycle Events ({breakdown?.total_lifecycle_events || 0} total, showing latest 50)
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Event</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Component</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Action</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>Stage</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Actor</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Detail</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {lifecycle.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>#{e.event_id}</td>
                    <td style={{ padding: 8 }}>{e.component}</td>
                    <td style={{ padding: 8 }}>{e.action}</td>
                    <td style={{ padding: 8, textAlign: 'center' }}>
                      <span style={stageBadge(e.stage)}>{e.stage}</span>
                    </td>
                    <td style={{ padding: 8, fontSize: 11 }}>{e.actor || '--'}</td>
                    <td style={{ padding: 8, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 11 }}>
                      {e.detail || '--'}
                    </td>
                    <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{(e.timestamp || '--').slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ marginTop: 18 }}>
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Knowledge Management Concepts</h3>
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12 }}>
                <strong style={{ color: '#1e293b' }}>{c.term}</strong>
                <p style={{ color: '#64748b', fontSize: 12, margin: '4px 0 0' }}>{c.definition}</p>
              </div>
            ))}
          </div>
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Metrics</h3>
            {(defs.metrics || []).map((m, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <strong style={{ color: '#1e293b' }}>{m.name}</strong>
                <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>{m.description}</span>
              </div>
            ))}
          </div>
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Compliance References</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
              {(defs.compliance || []).map((c, i) => (
                <li key={i} style={{ marginBottom: 6 }}>{c}</li>
              ))}
            </ul>
          </div>
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Remediation Strategies</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 12, color: '#475569' }}>
              {(defs.remediation || []).map((r, i) => (
                <li key={i} style={{ marginBottom: 6 }}>{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
