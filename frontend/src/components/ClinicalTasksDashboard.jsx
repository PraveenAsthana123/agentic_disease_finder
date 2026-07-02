import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const s = (status || '').toLowerCase()
  let bg = '#e2e8f0', fg = '#475569'
  if (s === 'addressed' || s === 'completed' || s === 'done') { bg = '#dcfce7'; fg = '#166534' }
  if (s === 'open') { bg = '#fef3c7'; fg = '#92400e' }
  if (s === 'pending') { bg = '#dbeafe'; fg = '#1e40af' }
  if (s === 'logged') { bg = '#f1f5f9'; fg = '#475569' }
  if (s === 'rejected' || s === 'not-implemented') { bg = '#fee2e2'; fg = '#991b1b' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: bg, color: fg
    }}>{status}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'categories', label: 'Category Breakdown' },
  { id: 'workflow', label: 'Workflow Activity' },
  { id: 'recent', label: 'Recent Requests' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ClinicalTasksDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/clinical-tasks/overview`),
      axios.get(`${API_URL}/clinical-tasks/breakdown`),
      axios.get(`${API_URL}/clinical-tasks/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Tasks...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40 }}>Task data not available</div>

  const k = overview.kpis || {}

  // Chart data
  const statusData = Object.entries(overview.status_distribution || {}).map(([name, value], i) => ({
    name, value, fill: COLORS[i % COLORS.length]
  }))
  const categoryData = Object.entries(overview.category_distribution || {}).map(([name, value], i) => ({
    name, value, fill: COLORS[i % COLORS.length]
  }))
  const componentData = (overview.top_components || []).slice(0, 8)
  const actionData = (overview.top_actions || []).slice(0, 8)

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Clinical Tasks Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Role-operational task tracking: operator requests, form assignments, workflow events from clinical.db
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}
          >{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Requests" value={k.total_requests} color="#3b82f6" /></Card>
            <Card><KPI label="Done" value={k.done} color="#10b981" /></Card>
            <Card><KPI label="Open" value={k.open} color="#f59e0b" /></Card>
            <Card><KPI label="Rejected" value={k.rejected} color="#ef4444" /></Card>
            <Card><KPI label="Completion Rate" value={`${k.completion_rate}%`} color="#8b5cf6" /></Card>
            <Card><KPI label="Form Assignments" value={k.total_form_assignments} sub={`${k.form_completion_rate}% complete`} /></Card>
            <Card><KPI label="Workflow Events" value={k.total_workflow_events} color="#06b6d4" /></Card>
            <Card><KPI label="Implemented" value={breakdown?.implementation_stats?.implemented || 0} sub={`${breakdown?.implementation_stats?.tested || 0} tested`} /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Request Status Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {statusData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Category Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={categoryData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {categoryData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Top components bar */}
          <Card title="Top Workflow Components (by event count)">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={componentData} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" angle={-30} textAnchor="end" height={80} fontSize={11} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* ── Categories Tab ── */}
      {tab === 'categories' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16, marginBottom: 24 }}>
            {(breakdown.category_details || []).map(cat => (
              <Card key={cat.category} title={`${cat.category} (${cat.total})`}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 12 }}>
                  <KPI label="Done" value={cat.done} color="#10b981" />
                  <KPI label="Open" value={cat.open} color="#f59e0b" />
                  <KPI label="Rate" value={`${cat.completion_rate}%`} color="#8b5cf6" />
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  {Object.entries(cat.statuses || {}).map(([s, c]) => (
                    <span key={s} style={{ marginRight: 10 }}><StatusBadge status={s} /> {c}</span>
                  ))}
                </div>
              </Card>
            ))}
          </div>

          {/* Component-Action cross-tab */}
          <Card title="Component-Action Activity">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(breakdown.component_actions || []).slice(0, 15)} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" angle={-30} textAnchor="end" height={80} fontSize={11} />
                <YAxis />
                <Tooltip content={({ active, payload }) => {
                  if (!active || !payload?.length) return null
                  const d = payload[0].payload
                  return (
                    <div style={{ background: '#fff', padding: 8, border: '1px solid #e2e8f0', borderRadius: 6, fontSize: 12 }}>
                      <div><strong>{d.component}</strong> / {d.action}</div>
                      <div>Count: {d.count}</div>
                    </div>
                  )
                }} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Form assignments */}
          {(breakdown.form_assignments || []).length > 0 && (
            <Card title="Form Assignments" span={2}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Patient', 'Instrument', 'Assigned By', 'Status', 'Created', 'Completed'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.form_assignments.map(f => (
                      <tr key={f.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px' }}>{f.patient_id}</td>
                        <td>{f.instrument}</td>
                        <td>{f.assigned_by}</td>
                        <td><StatusBadge status={f.status} /></td>
                        <td style={{ fontSize: 11, color: '#94a3b8' }}>{(f.created_at || '').slice(0, 16)}</td>
                        <td style={{ fontSize: 11, color: '#94a3b8' }}>{(f.completed_at || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      {/* ── Workflow Activity Tab ── */}
      {tab === 'workflow' && breakdown && (
        <>
          {/* Daily activity trend */}
          <Card title="Daily Workflow Activity (last 30 days)">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={[...(breakdown.daily_activity || [])].reverse()} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={11} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Top actions and actors side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <Card title="Top Actions">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={actionData} layout="vertical" margin={{ left: 60, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="action" fontSize={11} width={60} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Top Actors">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={(overview.top_actors || []).slice(0, 8)} layout="vertical" margin={{ left: 60, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="actor" fontSize={11} width={60} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Recent Requests Tab ── */}
      {tab === 'recent' && (
        <Card title="Recent Operator Requests (last 20)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['#', 'Request', 'Category', 'Status', 'Source', 'Module', 'Tested', 'Time'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.recent_requests || []).map(r => (
                  <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{r.id}</td>
                    <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.request}</td>
                    <td>{r.category}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td>{r.source}</td>
                    <td style={{ fontSize: 11, color: r.module ? '#166534' : '#94a3b8' }}>{r.module || '--'}</td>
                    <td style={{ fontSize: 11 }}>{r.tested || '--'}</td>
                    <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {Object.entries(defs.definitions).map(([section, items]) => (
            <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                {Object.entries(items).map(([term, desc]) => (
                  <div key={term} style={{ marginBottom: 8 }}>
                    <strong style={{ color: '#1e293b' }}>{term}:</strong>{' '}
                    <span style={{ color: '#475569' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
