import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'

const STATUS_COLORS = {
  active: '#22c55e',
  inactive: '#ef4444',
  suspended: '#f59e0b',
  healthy: '#22c55e',
  degraded: '#f59e0b',
  down: '#ef4444',
}

const PIE_COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span, accent }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined,
      borderLeft: accent ? `4px solid ${accent}` : undefined,
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color,
      textTransform: 'capitalize'
    }}>{status ? status.replace('_', ' ') : 'Unknown'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'users', label: 'Users' },
  { id: 'feature_flags', label: 'Feature Flags' },
  { id: 'system_health', label: 'System Health' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AdminDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedDef, setExpandedDef] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/admin-panel/overview`),
      axios.get(`${API_URL}/api/admin-panel/breakdown`),
      axios.get(`${API_URL}/api/admin-panel/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefs(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Admin Panel data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const kpis = ov.kpis || {}
  const usersByRole = ov.users_by_role || []
  const usersByStatus = ov.users_by_status || []
  const flagsByCategory = ov.flags_by_category || []
  const healthTrend = ov.health_trend || []
  const users = bd.users || []
  const featureFlags = bd.feature_flags || []
  const healthChecks = bd.health_checks || []
  const componentHealth = bd.component_health || []

  const thStyle = {
    textAlign: 'left', padding: '8px 10px', fontSize: 12,
    color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600
  }
  const tdStyle = {
    padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9'
  }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Admin Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        System administration: user management, feature flags, system health monitoring, and configuration
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13,
            fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b',
            background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Total Users"><KPI label="Total Users" value={fmt(kpis.total_users)} /></Card>
          <Card title="Active Users"><KPI label="Active Users" value={fmt(kpis.active_users)} color="#22c55e" /></Card>
          <Card title="MFA Adoption"><KPI label="MFA Adoption %" value={fmt(kpis.mfa_adoption_pct)} sub="%" color="#3b82f6" /></Card>
          <Card title="Flags Enabled"><KPI label="Flags Enabled" value={fmt(kpis.flags_enabled)} color="#22c55e" /></Card>
          <Card title="Flags Disabled"><KPI label="Flags Disabled" value={fmt(kpis.flags_disabled)} color="#ef4444" /></Card>
          <Card title="Avg Response Time"><KPI label="Avg Response (ms)" value={fmt(kpis.avg_response_time_ms)} sub="ms" color="#f59e0b" /></Card>
          <Card title="System Uptime"><KPI label="System Uptime %" value={fmt(kpis.system_uptime_pct)} sub="%" color="#22c55e" /></Card>
          <Card title="Config Entries"><KPI label="Config Entries" value={fmt(kpis.config_entries)} /></Card>

          <Card title="Users by Role" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={usersByRole}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Users by Status" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={usersByStatus} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {usersByStatus.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Flags by Category" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={flagsByCategory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Response Time Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={healthTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="response_time_ms" stroke="#f59e0b" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Users Tab */}
      {tab === 'users' && (
        <Card title="User Management">
          <div style={{ overflowX: 'auto', maxHeight: 520, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Username</th>
                  <th style={thStyle}>Full Name</th>
                  <th style={thStyle}>Role</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Department</th>
                  <th style={thStyle}>MFA Enabled</th>
                  <th style={thStyle}>Last Login</th>
                  <th style={thStyle}>Login Count</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{u.username}</td>
                    <td style={tdStyle}>{u.full_name}</td>
                    <td style={tdStyle}>{u.role}</td>
                    <td style={tdStyle}><StatusBadge status={u.status} /></td>
                    <td style={tdStyle}>{u.department}</td>
                    <td style={tdStyle}>
                      <Badge text={u.mfa_enabled ? 'Yes' : 'No'} color={u.mfa_enabled ? '#22c55e' : '#ef4444'} />
                    </td>
                    <td style={tdStyle}>{u.last_login}</td>
                    <td style={tdStyle}>{fmt(u.login_count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Feature Flags Tab */}
      {tab === 'feature_flags' && (
        <Card title="Feature Flags">
          <div style={{ overflowX: 'auto', maxHeight: 520, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Category</th>
                  <th style={thStyle}>Enabled</th>
                  <th style={thStyle}>Rollout %</th>
                  <th style={thStyle}>Owner</th>
                  <th style={thStyle}>Updated At</th>
                </tr>
              </thead>
              <tbody>
                {featureFlags.map((f, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{f.name}</td>
                    <td style={tdStyle}>{f.category}</td>
                    <td style={tdStyle}>
                      <Badge text={f.enabled ? 'Enabled' : 'Disabled'} color={f.enabled ? '#22c55e' : '#ef4444'} />
                    </td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, maxWidth: 100 }}>
                          <div style={{ width: `${f.rollout_percentage || 0}%`, height: '100%', background: '#3b82f6', borderRadius: 3 }} />
                        </div>
                        <span style={{ fontSize: 12, color: '#64748b' }}>{f.rollout_percentage || 0}%</span>
                      </div>
                    </td>
                    <td style={tdStyle}>{f.owner}</td>
                    <td style={tdStyle}>{f.updated_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* System Health Tab */}
      {tab === 'system_health' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 16 }}>
          {componentHealth.map((c, i) => (
            <Card key={i} title={c.component} accent={STATUS_COLORS[c.status] || '#94a3b8'}>
              <StatusBadge status={c.status} />
              <div style={{ marginTop: 10, fontSize: 13, color: '#475569' }}>
                <div>Avg Response: <strong>{fmt(c.avg_response_time_ms)} ms</strong></div>
                <div>Error Rate: <strong>{fmt(c.error_rate)}%</strong></div>
              </div>
            </Card>
          ))}

          <Card title="Health Checks" span={4}>
            <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Timestamp</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Response (ms)</th>
                    <th style={thStyle}>CPU %</th>
                    <th style={thStyle}>Memory %</th>
                    <th style={thStyle}>Disk %</th>
                    <th style={thStyle}>Error Count</th>
                  </tr>
                </thead>
                <tbody>
                  {healthChecks.map((h, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{h.timestamp}</td>
                      <td style={tdStyle}>{h.component}</td>
                      <td style={tdStyle}><StatusBadge status={h.status} /></td>
                      <td style={tdStyle}>{fmt(h.response_time_ms)}</td>
                      <td style={tdStyle}>{fmt(h.cpu_pct)}</td>
                      <td style={tdStyle}>{fmt(h.memory_pct)}</td>
                      <td style={tdStyle}>{fmt(h.disk_pct)}</td>
                      <td style={tdStyle}>{fmt(h.error_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <Card title="Definitions">
          <div style={{ display: 'grid', gap: 10 }}>
            {(defs || []).map((d, i) => (
              <div key={i} style={{
                padding: 14, borderRadius: 8, background: '#f8fafc',
                border: '1px solid #e2e8f0', cursor: 'pointer'
              }} onClick={() => setExpandedDef(expandedDef === i ? null : i)}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{d.title}</div>
                {expandedDef === i && (
                  <div style={{ marginTop: 8, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>
                    {d.description}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
