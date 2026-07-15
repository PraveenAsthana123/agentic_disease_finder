import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']
const STATUS_COLORS = { active: '#3b82f6', completed: '#10b981', failed: '#ef4444', paused: '#f59e0b', pending: '#94a3b8' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Workflow Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PaperclipDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/paperclip/overview`),
      axios.get(`${API_URL}/api/paperclip/breakdown`),
      axios.get(`${API_URL}/api/paperclip/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Paperclip Orchestration data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Paperclip Business Orchestration Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Clinical workflow orchestration — intake, referrals, reports, scheduling, compliance, exports
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function StatusBadge({ status }) {
  const bg = { active: '#dbeafe', completed: '#dcfce7', failed: '#fee2e2', paused: '#fef3c7', pending: '#f1f5f9' }
  const fg = { active: '#2563eb', completed: '#16a34a', failed: '#dc2626', paused: '#d97706', pending: '#64748b' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
      background: bg[status] || '#f1f5f9', color: fg[status] || '#64748b',
    }}>{status}</span>
  )
}

function ProgressBar({ completed, total }) {
  const pct = total > 0 ? Math.round(completed / total * 100) : 0
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{ width: 50, height: 6, borderRadius: 3, background: '#e2e8f0', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: pct === 100 ? '#10b981' : '#3b82f6', borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 11, color: '#64748b' }}>{completed}/{total}</span>
    </div>
  )
}

function OverviewTab({ data }) {
  const statusData = (data.status_distribution || []).map(s => ({
    name: s.status, value: s.count, fill: STATUS_COLORS[s.status] || '#94a3b8'
  }))

  const catChart = (data.category_distribution || []).map((c, i) => ({
    name: c.category, count: c.count, fill: COLORS[i % COLORS.length]
  }))

  const triggerChart = (data.trigger_distribution || []).map((t, i) => ({
    name: t.trigger_type, count: t.count, fill: COLORS[(i + 2) % COLORS.length]
  }))

  const dailyData = (data.daily_volume || []).map(d => ({
    date: d.date.slice(5),
    count: d.count
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Workflows" value={data.total_workflows} />
      </Card>
      <Card>
        <KPI label="Active" value={data.active} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Completed" value={data.completed} sub={`${data.completion_rate}% rate`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Failed" value={data.failed} color={data.failed > 0 ? '#ef4444' : '#10b981'} />
      </Card>

      <Card title="Status Distribution" span={2}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <ResponsiveContainer width="55%" height={200}>
            <PieChart>
              <Pie data={statusData} dataKey="value" cx="50%" cy="50%" outerRadius={70} innerRadius={35}
                label={({ name, value }) => `${name}: ${value}`}>
                {statusData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div>
            {statusData.map((s, i) => (
              <div key={i} style={{ fontSize: 12, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: s.fill, display: 'inline-block' }} />
                {s.name}: <strong>{s.value}</strong>
              </div>
            ))}
            <div style={{ fontSize: 12, marginTop: 8, color: '#64748b' }}>
              Avg steps: <strong>{data.avg_steps_completion_pct}%</strong>
            </div>
          </div>
        </div>
      </Card>

      <Card title="Category Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={catChart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6">
              {catChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Types" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={triggerChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={90} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6">
              {triggerChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Workflow Volume (14 days)" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={dailyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Priority Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={(data.priority_distribution || []).map((p, i) => ({ name: p.priority, value: p.count }))}
              dataKey="value" cx="50%" cy="50%" outerRadius={70} innerRadius={35}
              label={({ name, value }) => `${name}: ${value}`}>
              {(data.priority_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {data.top_failing && data.top_failing.length > 0 && (
        <Card title="Top Failing Workflows" span={2}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Workflow</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Failures</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Last Error</th>
              </tr>
            </thead>
            <tbody>
              {data.top_failing.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.workflow_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px', color: '#ef4444', fontWeight: 600 }}>{f.fail_count}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{f.error_message || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

function BreakdownTab({ data }) {
  const { per_category, recent_workflows, owner_workload, stalled_workflows, per_workflow_type } = data || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Recent Workflows (last 20)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Workflow</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Category</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Priority</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Progress</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Trigger</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Duration</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {(recent_workflows || []).map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{w.workflow_id}</td>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{w.workflow_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}><StatusBadge status={w.status} /></td>
                  <td style={{ padding: '5px 8px' }}>
                    <span style={{ padding: '1px 6px', borderRadius: 6, fontSize: 10, background: '#eff6ff', color: '#3b82f6' }}>{w.category}</span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      padding: '1px 6px', borderRadius: 6, fontSize: 10,
                      background: w.priority === 'critical' ? '#fee2e2' : w.priority === 'high' ? '#fef3c7' : '#f1f5f9',
                      color: w.priority === 'critical' ? '#dc2626' : w.priority === 'high' ? '#d97706' : '#64748b',
                    }}>{w.priority}</span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <ProgressBar completed={w.steps_completed} total={w.steps_total} />
                  </td>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{w.trigger_type}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{w.owner}</td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#64748b' }}>
                    {w.duration_seconds != null ? `${Math.round(w.duration_seconds / 60)}m` : '-'}
                  </td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{(w.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Owner Workload">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Active</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Completed</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Failed</th>
            </tr>
          </thead>
          <tbody>
            {(owner_workload || []).map((o, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '5px 8px', fontWeight: 500 }}>{o.owner}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: '#3b82f6', fontWeight: 600 }}>{o.active_count}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: '#10b981', fontWeight: 600 }}>{o.completed_count}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: o.failed_count > 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>{o.failed_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {stalled_workflows && stalled_workflows.length > 0 && (
        <Card title={`Stalled Workflows (${stalled_workflows.length} — active, <50% steps, >3 days old)`}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Workflow</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Progress</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Owner</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {stalled_workflows.map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{w.workflow_id}</td>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{w.workflow_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <ProgressBar completed={w.steps_completed} total={w.steps_total} />
                  </td>
                  <td style={{ padding: '5px 8px' }}>{w.owner}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{(w.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {per_workflow_type && per_workflow_type.length > 0 && (
        <Card title="Workflow Type Summary">
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Workflow Type</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Count</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Success Rate</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Avg Duration</th>
              </tr>
            </thead>
            <tbody>
              {per_workflow_type.map((wt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{wt.workflow_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>{wt.count}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      fontWeight: 600,
                      color: (wt.success_rate || 0) >= 80 ? '#10b981' : (wt.success_rate || 0) >= 50 ? '#f59e0b' : '#ef4444',
                    }}>{wt.success_rate ?? 0}%</span>
                  </td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#64748b' }}>
                    {wt.avg_duration != null ? `${Math.round(wt.avg_duration / 60)}m` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {per_category && Object.keys(per_category).length > 0 && (
        <Card title="Workflows by Category">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12 }}>
            {Object.entries(per_category).map(([cat, workflows]) => (
              <div key={cat} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 12 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>{cat} ({Array.isArray(workflows) ? workflows.length : 0})</h4>
                {(Array.isArray(workflows) ? workflows : []).slice(0, 8).map((w, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0', fontSize: 12 }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: '50%',
                      background: STATUS_COLORS[w.status] || '#94a3b8',
                    }} />
                    <span style={{ flex: 1 }}>{w.workflow_name}</span>
                    <StatusBadge status={w.status} />
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Workflow Statuses', items: data.workflow_statuses || [] },
    { title: 'Categories', items: data.categories || [] },
    { title: 'Trigger Types', items: data.trigger_types || [] },
    { title: 'Priority Levels', items: data.priority_levels || [] },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((s, si) => (
        <Card key={si} title={s.title}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <tbody>
              {(Array.isArray(s.items) ? s.items : Object.entries(s.items).map(([k, v]) => ({ term: k, definition: v }))).map((item, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155' }}>
                    {item.term || item.name || item.status || item.level || Object.keys(item)[0]}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>
                    {item.definition || item.description || item.desc || Object.values(item)[1]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}

      {data.glossary && data.glossary.length > 0 && (
        <Card title="Orchestration Glossary" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 0 }}>
            {data.glossary.map((g, i) => (
              <div key={i} style={{ padding: '6px 8px', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                <span style={{ fontWeight: 600, color: '#334155' }}>{g.term}</span>
                <span style={{ color: '#64748b', marginLeft: 8 }}>{g.definition}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
