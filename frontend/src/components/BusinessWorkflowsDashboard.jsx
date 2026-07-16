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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']
const STATUS_COLORS = { active: '#3b82f6', completed: '#10b981', failed: '#ef4444', paused: '#f59e0b', pending: '#94a3b8' }
const PRIORITY_COLORS = { critical: '#ef4444', high: '#f59e0b', medium: '#3b82f6', low: '#94a3b8' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
      fontWeight: 600, background: (color || '#94a3b8') + '22', color: color || '#94a3b8'
    }}>{text}</span>
  )
}

export default function BusinessWorkflowsDashboard() {
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
      axios.get(`${API_URL}/api/business-workflows/overview`),
      axios.get(`${API_URL}/api/business-workflows/breakdown`),
      axios.get(`${API_URL}/api/business-workflows/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Business Workflows data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Business Workflows Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Clinical &amp; administrative workflow automation — {overview?.total_workflows} workflows,
        {' '}{overview?.total_workflow_types} types, {overview?.total_patients} patients
      </p>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const statusPie = data.status_distribution.map((s, i) => ({ ...s, fill: STATUS_COLORS[s.status] || COLORS[i] }))
  const categoryPie = data.category_distribution.map((c, i) => ({ ...c, fill: COLORS[i] }))
  const priorityPie = data.priority_distribution.map((p, i) => ({ ...p, fill: PRIORITY_COLORS[p.priority] || COLORS[i] }))
  const triggerPie = data.trigger_distribution.map((t, i) => ({ ...t, fill: COLORS[i + 2] }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Workflows"><KPI label="Workflows" value={data.total_workflows} /></Card>
      <Card title="Active"><KPI label="Running now" value={data.active_workflows} color="#3b82f6" /></Card>
      <Card title="Completion Rate"><KPI label="% completed" value={`${data.completion_rate}%`} color="#10b981" /></Card>
      <Card title="Failure Rate"><KPI label="% failed" value={`${data.failure_rate}%`} color="#ef4444" /></Card>

      <Card title="Avg Duration (completed)" span={2}>
        <KPI label="minutes" value={data.avg_duration_min} sub={`min ${data.min_duration_min} / max ${data.max_duration_min} min`} />
      </Card>
      <Card title="Step Completion"><KPI label="avg % steps done" value={`${data.avg_step_completion_pct}%`} color="#8b5cf6" /></Card>
      <Card title="Retries"><KPI label="total retries" value={data.total_retries} sub={`${data.workflows_with_retries} workflows retried`} color="#f59e0b" /></Card>

      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={statusPie} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`}>
            {statusPie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Category Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={categoryPie} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={80} label={({ category, count }) => `${category} (${count})`}>
            {categoryPie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Priority Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={priorityPie}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="priority" /><YAxis /><Tooltip />
            <Bar dataKey="count" fill="#3b82f6">{priorityPie.map((e, i) => <Cell key={i} fill={e.fill} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Trigger Types" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart><Pie data={triggerPie} dataKey="count" nameKey="trigger_type" cx="50%" cy="50%" outerRadius={80} label={({ trigger_type, count }) => `${trigger_type} (${count})`}>
            {triggerPie.map((e, i) => <Cell key={i} fill={e.fill} />)}
          </Pie><Tooltip /><Legend /></PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Workflow Types" span={2}>
        <ResponsiveContainer width="100%" height={Math.max(200, data.workflow_distribution.length * 30)}>
          <BarChart data={data.workflow_distribution} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="workflow_name" width={160} tick={{ fontSize: 11 }} /><Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card title="Monthly Trend" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="month" /><YAxis /><Tooltip />
            <Line type="monotone" dataKey="total" stroke="#3b82f6" name="Total" strokeWidth={2} />
            <Line type="monotone" dataKey="completed" stroke="#10b981" name="Completed" strokeWidth={2} />
            <Line type="monotone" dataKey="failed" stroke="#ef4444" name="Failed" strokeWidth={2} />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Owner Workload" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.owner_workload}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="owner" /><YAxis /><Tooltip />
            <Bar dataKey="completed" stackId="a" fill="#10b981" name="Completed" />
            <Bar dataKey="active" stackId="a" fill="#3b82f6" name="Active" />
            <Bar dataKey="failed" stackId="a" fill="#ef4444" name="Failed" />
            <Legend />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category × Status" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.category_status_crosstab}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="category" /><YAxis /><Tooltip />
            <Bar dataKey="completed" stackId="a" fill="#10b981" name="Completed" />
            <Bar dataKey="active" stackId="a" fill="#3b82f6" name="Active" />
            <Bar dataKey="failed" stackId="a" fill="#ef4444" name="Failed" />
            <Bar dataKey="paused" stackId="a" fill="#f59e0b" name="Paused" />
            <Bar dataKey="pending" stackId="a" fill="#94a3b8" name="Pending" />
            <Legend />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Workflow Type Performance" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Workflow</th><th style={th}>Total</th><th style={th}>Completed</th><th style={th}>Failed</th><th style={th}>Avg Step %</th><th style={th}>Avg Duration (min)</th>
            </tr></thead>
            <tbody>{data.workflow_stats.map((w, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{w.workflow_name}</td><td style={td}>{w.total}</td>
                <td style={td}><Badge text={w.completed} color="#10b981" /></td>
                <td style={td}>{w.failed > 0 ? <Badge text={w.failed} color="#ef4444" /> : '0'}</td>
                <td style={td}>{w.avg_step_pct}%</td><td style={td}>{w.avg_duration_min ?? '—'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      {data.active_workflows.length > 0 && (
        <Card title={`Active Workflows (${data.active_workflows.length})`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr style={{ background: '#f8fafc' }}>
                <th style={th}>ID</th><th style={th}>Workflow</th><th style={th}>Category</th><th style={th}>Priority</th><th style={th}>Owner</th><th style={th}>Patient</th><th style={th}>Progress</th><th style={th}>Created</th>
              </tr></thead>
              <tbody>{data.active_workflows.map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{w.workflow_id}</td><td style={td}>{w.workflow_name}</td>
                  <td style={td}><Badge text={w.category} color={COLORS[['Administrative','Clinical','Compliance','Technical'].indexOf(w.category)]} /></td>
                  <td style={td}><Badge text={w.priority} color={PRIORITY_COLORS[w.priority]} /></td>
                  <td style={td}>{w.owner}</td><td style={td}>{w.patient_id || '—'}</td>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3 }}>
                        <div style={{ width: `${(w.steps_completed / w.steps_total * 100)}%`, height: '100%', background: '#3b82f6', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{w.steps_completed}/{w.steps_total}</span>
                    </div>
                  </td>
                  <td style={td}>{w.created_at?.slice(0, 10)}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
      )}

      {data.failed_workflows.length > 0 && (
        <Card title={`Failed Workflows (${data.failed_workflows.length})`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead><tr style={{ background: '#fef2f2' }}>
                <th style={th}>ID</th><th style={th}>Workflow</th><th style={th}>Priority</th><th style={th}>Owner</th><th style={th}>Error</th><th style={th}>Retries</th>
              </tr></thead>
              <tbody>{data.failed_workflows.map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={td}>{w.workflow_id}</td><td style={td}>{w.workflow_name}</td>
                  <td style={td}><Badge text={w.priority} color={PRIORITY_COLORS[w.priority]} /></td>
                  <td style={td}>{w.owner}</td>
                  <td style={{ ...td, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{w.error_message || '—'}</td>
                  <td style={td}>{w.retry_count}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Per-Patient Workflow Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>Patient</th><th style={th}>Total</th><th style={th}>Completed</th><th style={th}>Failed</th>
            </tr></thead>
            <tbody>{data.per_patient.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{p.patient_id}</td><td style={td}>{p.total}</td>
                <td style={td}><Badge text={p.completed} color="#10b981" /></td>
                <td style={td}>{p.failed > 0 ? <Badge text={p.failed} color="#ef4444" /> : '0'}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent Workflows" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={th}>ID</th><th style={th}>Workflow</th><th style={th}>Category</th><th style={th}>Status</th><th style={th}>Priority</th><th style={th}>Trigger</th><th style={th}>Owner</th><th style={th}>Progress</th><th style={th}>Created</th>
            </tr></thead>
            <tbody>{data.recent_workflows.map((w, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={td}>{w.workflow_id}</td><td style={td}>{w.workflow_name}</td>
                <td style={td}>{w.category}</td>
                <td style={td}><Badge text={w.status} color={STATUS_COLORS[w.status]} /></td>
                <td style={td}><Badge text={w.priority} color={PRIORITY_COLORS[w.priority]} /></td>
                <td style={td}>{w.trigger_type}</td><td style={td}>{w.owner}</td>
                <td style={td}>{w.steps_completed}/{w.steps_total}</td>
                <td style={td}>{w.created_at?.slice(0, 10)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

const th = { textAlign: 'left', padding: '8px 10px', fontSize: 11, color: '#64748b', fontWeight: 600 }
const td = { padding: '8px 10px' }

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Field Definitions', items: data.fields },
    { title: 'Status Definitions', items: data.statuses },
    { title: 'Category Descriptions', items: data.categories },
    { title: 'Priority Levels', items: data.priorities },
    { title: 'Trigger Types', items: data.trigger_types },
    { title: 'Workflow Types', items: data.workflow_types },
    { title: 'Clinical Glossary', items: data.glossary },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((sec, i) => (
        <Card key={i} title={sec.title} span={sec.items && Object.keys(sec.items).length > 6 ? 2 : 1}>
          {sec.items && Object.entries(sec.items).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 12, color: '#334155' }}>{k}: </span>
              <span style={{ fontSize: 12, color: '#64748b' }}>{v}</span>
            </div>
          ))}
        </Card>
      ))}
    </div>
  )
}
