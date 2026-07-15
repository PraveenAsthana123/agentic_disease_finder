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
const STATUS_COLORS = { completed: '#10b981', running: '#3b82f6', failed: '#ef4444', queued: '#94a3b8', cancelled: '#f59e0b' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Execution Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function OpenClawDashboard() {
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
      axios.get(`${API_URL}/api/openclaw/overview`),
      axios.get(`${API_URL}/api/openclaw/breakdown`),
      axios.get(`${API_URL}/api/openclaw/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading OpenClaw Execution data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>OpenClaw Execution Orchestration Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Agent execution orchestration — autonomous/supervised/manual runs, token usage, pipeline DAGs
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
  const bg = { completed: '#dcfce7', running: '#dbeafe', failed: '#fee2e2', queued: '#f1f5f9', cancelled: '#fef3c7' }
  const fg = { completed: '#16a34a', running: '#2563eb', failed: '#dc2626', queued: '#64748b', cancelled: '#d97706' }
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

  const modeChart = (data.mode_distribution || []).map((m, i) => ({
    name: m.execution_mode, count: m.count, fill: COLORS[i % COLORS.length]
  }))

  const triggerChart = (data.trigger_distribution || []).map((t, i) => ({
    name: t.triggered_by, count: t.count, fill: COLORS[(i + 2) % COLORS.length]
  }))

  const dailyData = (data.daily_volume || []).map(d => ({
    date: d.date.slice(5),
    count: d.count
  }))

  const agentChart = (data.agent_distribution || []).slice(0, 10).map((a, i) => ({
    name: a.agent_name.length > 18 ? a.agent_name.slice(0, 16) + '..' : a.agent_name,
    count: a.count, fill: COLORS[i % COLORS.length]
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Executions" value={data.total_executions} />
      </Card>
      <Card>
        <KPI label="Running" value={data.running} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Completed" value={data.completed} sub={`${data.completion_rate}% rate`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Failed" value={data.failed} color={data.failed > 0 ? '#ef4444' : '#10b981'} />
      </Card>

      <Card>
        <KPI label="Queued" value={data.queued} color="#64748b" />
      </Card>
      <Card>
        <KPI label="Cancelled" value={data.cancelled} color="#d97706" />
      </Card>
      <Card>
        <KPI label="Total Tokens" value={data.total_tokens ? data.total_tokens.toLocaleString() : '0'} color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Avg Duration" value={data.avg_duration_seconds ? `${Math.round(data.avg_duration_seconds)}s` : '--'} color="#06b6d4" />
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
          </div>
        </div>
      </Card>

      <Card title="Execution Mode Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={modeChart}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6">
              {modeChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Top Agents by Executions" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={agentChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={130} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6">
              {agentChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Trigger Types" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={triggerChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={80} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4">
              {triggerChart.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Execution Volume (21 days)" span={2}>
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

      {data.top_failing_agents && data.top_failing_agents.length > 0 && (
        <Card title="Top Failing Agents" span={2}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Failures</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Last Error</th>
              </tr>
            </thead>
            <tbody>
              {data.top_failing_agents.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.agent_name}</td>
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
  const { recent_executions, agent_workload, chained_executions, failed_executions, per_agent_stats } = data || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Recent Executions (last 25)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Exec ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Mode</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Priority</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Progress</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Trigger</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Tokens</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Duration</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {(recent_executions || []).map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{(e.execution_id || '').slice(0, 12)}</td>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{e.agent_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}><StatusBadge status={e.status} /></td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      padding: '1px 6px', borderRadius: 6, fontSize: 10,
                      background: e.execution_mode === 'autonomous' ? '#dcfce7' : e.execution_mode === 'supervised' ? '#dbeafe' : '#f1f5f9',
                      color: e.execution_mode === 'autonomous' ? '#16a34a' : e.execution_mode === 'supervised' ? '#2563eb' : '#64748b',
                    }}>{e.execution_mode}</span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      padding: '1px 6px', borderRadius: 6, fontSize: 10,
                      background: e.priority === 'critical' ? '#fee2e2' : e.priority === 'high' ? '#fef3c7' : '#f1f5f9',
                      color: e.priority === 'critical' ? '#dc2626' : e.priority === 'high' ? '#d97706' : '#64748b',
                    }}>{e.priority}</span>
                  </td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <ProgressBar completed={e.steps_completed} total={e.steps_total} />
                  </td>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{e.triggered_by}</td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#8b5cf6' }}>
                    {((e.input_tokens || 0) + (e.output_tokens || 0)).toLocaleString()}
                  </td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#64748b' }}>
                    {e.duration_seconds != null ? `${Math.round(e.duration_seconds)}s` : '-'}
                  </td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{(e.created_at || '').slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Agent Workload">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Completed</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Running</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Failed</th>
              <th style={{ textAlign: 'right', padding: '6px 8px' }}>Avg Duration</th>
            </tr>
          </thead>
          <tbody>
            {(agent_workload || []).map((a, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '5px 8px', fontWeight: 500 }}>{a.agent_name}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: '#10b981', fontWeight: 600 }}>{a.completed_count}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: '#3b82f6', fontWeight: 600 }}>{a.running_count}</td>
                <td style={{ textAlign: 'center', padding: '5px 8px', color: a.failed_count > 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>{a.failed_count}</td>
                <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#64748b' }}>
                  {a.avg_duration != null ? `${Math.round(a.avg_duration)}s` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {chained_executions && chained_executions.length > 0 && (
        <Card title={`Chained Executions (${chained_executions.length} — triggered by parent execution)`}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Exec ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Parent Exec</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {chained_executions.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{(c.execution_id || '').slice(0, 12)}</td>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{c.agent_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}><StatusBadge status={c.status} /></td>
                  <td style={{ padding: '5px 8px', fontFamily: 'monospace', fontSize: 11 }}>{(c.parent_execution_id || '').slice(0, 12)}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{(c.created_at || '').slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {failed_executions && failed_executions.length > 0 && (
        <Card title={`Failed Executions (${failed_executions.length})`}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Task</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Error</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Retries</th>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {failed_executions.slice(0, 20).map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fff5f5' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{f.agent_name}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{(f.task_description || '').slice(0, 50)}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#ef4444' }}>{f.error_message || '-'}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px', fontWeight: 600 }}>{f.retry_count}</td>
                  <td style={{ padding: '5px 8px', fontSize: 11, color: '#64748b' }}>{(f.created_at || '').slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {per_agent_stats && per_agent_stats.length > 0 && (
        <Card title="Per-Agent Performance Summary">
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#64748b' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px' }}>Agent</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Total</th>
                <th style={{ textAlign: 'center', padding: '6px 8px' }}>Success Rate</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Avg Duration</th>
                <th style={{ textAlign: 'right', padding: '6px 8px' }}>Avg Tokens</th>
              </tr>
            </thead>
            <tbody>
              {per_agent_stats.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '5px 8px', fontWeight: 500 }}>{a.agent_name}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>{a.total_count}</td>
                  <td style={{ textAlign: 'center', padding: '5px 8px' }}>
                    <span style={{
                      fontWeight: 600,
                      color: (a.success_rate || 0) >= 80 ? '#10b981' : (a.success_rate || 0) >= 50 ? '#f59e0b' : '#ef4444',
                    }}>{a.success_rate ?? 0}%</span>
                  </td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#64748b' }}>
                    {a.avg_duration != null ? `${Math.round(a.avg_duration)}s` : '-'}
                  </td>
                  <td style={{ textAlign: 'right', padding: '5px 8px', fontSize: 11, color: '#8b5cf6' }}>
                    {a.avg_tokens != null ? Math.round(a.avg_tokens).toLocaleString() : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sections = [
    { title: 'Execution Statuses', items: data.execution_statuses || {} },
    { title: 'Execution Modes', items: data.execution_modes || {} },
    { title: 'Trigger Types', items: data.trigger_types || {} },
    { title: 'Priority Levels', items: data.priority_levels || {} },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map((s, si) => (
        <Card key={si} title={s.title}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <tbody>
              {Object.entries(s.items).map(([term, definition], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155' }}>
                    {term}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>
                    {definition}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}

      {data.glossary && Object.keys(data.glossary).length > 0 && (
        <Card title="Execution Orchestration Glossary" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 0 }}>
            {Object.entries(data.glossary).map(([term, definition], i) => (
              <div key={i} style={{ padding: '6px 8px', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                <span style={{ fontWeight: 600, color: '#334155' }}>{term}</span>
                <span style={{ color: '#64748b', marginLeft: 8 }}>{definition}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
