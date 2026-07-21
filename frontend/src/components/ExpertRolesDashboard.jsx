import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
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

export default function ExpertRolesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [roleFilter, setRoleFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/expert-roles/overview`),
      axios.get(`${API_URL}/expert-roles/breakdown`),
      axios.get(`${API_URL}/expert-roles/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Expert Roles...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-role', 'tasks', 'endpoints', 'definitions']
  const s = overview.summary || {}

  // Build flat tasks list from breakdown for Tasks + Endpoints tabs
  const allTasks = []
  const allEndpointTasks = []
  if (breakdown && breakdown.per_role) {
    breakdown.per_role.forEach(role => {
      (role.tasks || []).forEach(task => {
        const row = {
          role: role.role,
          role_icon: role.icon,
          task: task.name,
          ai_feature: task.ai_feature,
          status: task.status,
          steps: task.step_count ?? (task.steps || []).length,
          challenges: task.challenge_count ?? (task.challenges || []).length,
          endpoints: task.endpoints || []
        }
        allTasks.push(row)
        if (row.endpoints.length > 0) allEndpointTasks.push(row)
      })
    })
  }

  // Unique role names for filter
  const roleNames = breakdown
    ? ['all', ...(breakdown.per_role || []).map(r => r.role)]
    : ['all']

  const filteredTasks = roleFilter === 'all'
    ? allTasks
    : allTasks.filter(t => t.role === roleFilter)

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1280, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>
        {overview.title || 'Multidisciplinary Expert Roles — Epilepsy AI Platform'}
      </h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note}</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t ? '2px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t ? '#eff6ff' : '#fff',
            color: tab === t ? '#2563eb' : '#64748b'
          }}>
            {t.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={s.total_roles} />
              <KPI label="Total Tasks" value={s.total_tasks} />
              <KPI label="Built Tasks" value={s.built_tasks} sub="all live" />
              <KPI label="Total Steps" value={s.total_steps} />
              <KPI label="Total Challenges" value={s.total_challenges} />
              <KPI label="Total Endpoints" value={s.total_endpoints} />
              <KPI label="Dashboards" value={s.roles_with_dashboards} sub="with components" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Task status distribution pie */}
            <Card title="Task Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={overview.status_distribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%" cy="50%"
                    outerRadius={80}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {(overview.status_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Dashboard status distribution pie */}
            <Card title="Dashboard Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={overview.dashboard_status_distribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%" cy="50%"
                    outerRadius={80}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {(overview.dashboard_status_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Tasks per role bar chart (horizontal) */}
          <Card title="Tasks per Role">
            <ResponsiveContainer width="100%" height={Math.max(300, (overview.tasks_per_role || []).length * 36)}>
              <BarChart
                data={overview.tasks_per_role}
                layout="vertical"
                margin={{ left: 220, right: 30 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={210} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Tasks" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Roles summary table */}
          <Card title="All Roles Summary">
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Icon</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Mission</th>
                    <th style={thStyle}>Tasks</th>
                    <th style={thStyle}>Built</th>
                    <th style={thStyle}>Dashboard</th>
                    <th style={thStyle}>Endpoints</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.roles_table || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontSize: 18 }}>{r.icon}</td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{r.role}</td>
                      <td style={{ ...tdStyle, color: '#64748b', maxWidth: 260 }}>{r.mission_short}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>{r.task_count}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>{r.built_count}</td>
                      <td style={tdStyle}><StatusBadge status={r.dashboard_status || 'unknown'} /></td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>
                        {r.has_endpoints ? (
                          <span style={{ color: '#22c55e', fontWeight: 700 }}>Yes</span>
                        ) : (
                          <span style={{ color: '#94a3b8' }}>—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* BY ROLE TAB */}
      {tab === 'by-role' && breakdown && (
        <>
          {(breakdown.per_role || []).map((r, i) => (
            <Card key={i} title={`${r.icon || ''} ${r.role}`}>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8, alignItems: 'center' }}>
                <StatusBadge status={r.dashboard_status || 'unknown'} />
                <span style={{ fontSize: 12, color: '#64748b' }}>Tasks: {(r.tasks || []).length}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>Built: {(r.tasks || []).filter(t => t.status === 'built').length}</span>
              </div>

              {r.mission && (
                <p style={{ fontSize: 12, color: '#475569', margin: '0 0 8px', fontStyle: 'italic' }}>
                  {r.mission}
                </p>
              )}

              {r.data_hook && (
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 10, background: '#f8fafc', padding: '4px 8px', borderRadius: 4 }}>
                  Data: {r.data_hook}
                </div>
              )}

              {/* Task list */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {(r.tasks || []).map((task, j) => (
                  <div key={j} style={{
                    background: '#f8fafc', borderRadius: 6, padding: '10px 12px',
                    border: '1px solid #e2e8f0'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
                      <span style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{task.name}</span>
                      <StatusBadge status={task.status} />
                    </div>
                    <div style={{ fontSize: 11, color: '#3b82f6', marginBottom: 6 }}>
                      AI: {task.ai_feature}
                    </div>
                    <div style={{ display: 'flex', gap: 16 }}>
                      <span style={{ fontSize: 11, color: '#64748b' }}>
                        Steps: <strong>{(task.steps || []).length}</strong>
                      </span>
                      <span style={{ fontSize: 11, color: '#64748b' }}>
                        Challenges: <strong>{(task.challenges || []).length}</strong>
                      </span>
                    </div>
                    {task.endpoints && task.endpoints.length > 0 && (
                      <div style={{ marginTop: 6, fontSize: 11, color: '#64748b' }}>
                        Endpoints: {task.endpoints.map((ep, k) => (
                          <code key={k} style={{
                            background: '#e0f2fe', color: '#0369a1',
                            padding: '1px 5px', borderRadius: 3, marginRight: 4, fontSize: 10
                          }}>
                            {ep}
                          </code>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {r.dashboard_component && (
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 10 }}>
                  Component: <code style={{ background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{r.dashboard_component}</code>
                </div>
              )}
            </Card>
          ))}
        </>
      )}

      {/* TASKS TAB */}
      {tab === 'tasks' && (
        <>
          {/* Role filter */}
          <Card>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: '#64748b', marginRight: 4 }}>Filter by role:</span>
              {roleNames.map((name, i) => (
                <button key={i} onClick={() => setRoleFilter(name)} style={{
                  padding: '4px 10px', borderRadius: 5, fontSize: 11, fontWeight: 600, cursor: 'pointer',
                  border: roleFilter === name ? '2px solid #3b82f6' : '1px solid #e2e8f0',
                  background: roleFilter === name ? '#eff6ff' : '#fff',
                  color: roleFilter === name ? '#2563eb' : '#64748b'
                }}>
                  {name === 'all' ? 'All Roles' : name}
                </button>
              ))}
            </div>
          </Card>

          <Card title={`All Tasks (${filteredTasks.length})`}>
            <div style={{ maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Task Name</th>
                    <th style={thStyle}>AI Feature</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Steps</th>
                    <th style={thStyle}>Challenges</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTasks.map((t, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                        <span style={{ marginRight: 4 }}>{t.role_icon}</span>
                        <span style={{ fontSize: 11, color: '#64748b' }}>{t.role}</span>
                      </td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{t.task}</td>
                      <td style={{ ...tdStyle, color: '#3b82f6', fontSize: 11 }}>{t.ai_feature}</td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>{t.steps}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>{t.challenges}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ENDPOINTS TAB */}
      {tab === 'endpoints' && (
        <>
          <Card>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <KPI label="Tasks with Endpoints" value={allEndpointTasks.length} />
              <KPI label="Total Endpoint Refs" value={allEndpointTasks.reduce((acc, t) => acc + t.endpoints.length, 0)} />
            </div>
          </Card>

          <Card title="Tasks with Live Endpoints">
            <div style={{ maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Task</th>
                    <th style={thStyle}>Endpoints</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {allEndpointTasks.map((t, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                        <span style={{ marginRight: 4 }}>{t.role_icon}</span>
                        <span style={{ fontSize: 11, color: '#64748b' }}>{t.role}</span>
                      </td>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{t.task}</td>
                      <td style={tdStyle}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {t.endpoints.map((ep, j) => (
                            <code key={j} style={{
                              background: '#e0f2fe', color: '#0369a1',
                              padding: '2px 6px', borderRadius: 3, fontSize: 11
                            }}>
                              {ep}
                            </code>
                          ))}
                        </div>
                      </td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Endpoints per role bar */}
          {(() => {
            const epByRole = {}
            allEndpointTasks.forEach(t => {
              epByRole[t.role] = (epByRole[t.role] || 0) + t.endpoints.length
            })
            const epChartData = Object.entries(epByRole).map(([name, value]) => ({ name, value }))
            return (
              <Card title="Endpoints per Role">
                <ResponsiveContainer width="100%" height={Math.max(260, epChartData.length * 36)}>
                  <BarChart data={epChartData} layout="vertical" margin={{ left: 220, right: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" allowDecimals={false} />
                    <YAxis type="category" dataKey="name" width={210} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#14b8a6" radius={[0, 4, 4, 0]} name="Endpoints" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )
          })()}
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            {(defs.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <StatusBadge status={s.status} />
                <span style={{ fontSize: 12, color: '#475569' }}>{s.description}</span>
              </div>
            ))}
          </Card>

          <Card title="Glossary">
            <div style={{ maxHeight: 420, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Term</th>
                    <th style={thStyle}>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                      <td style={tdStyle}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
