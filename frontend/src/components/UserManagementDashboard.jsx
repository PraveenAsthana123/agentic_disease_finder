import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, BarChart, Bar
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const STATUS_COLORS = {
  Active: '#10b981',
  Inactive: '#f59e0b',
  Suspended: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'users', label: 'Users' },
  { id: 'roles', label: 'Roles & Permissions' },
  { id: 'activity', label: 'Activity' },
  { id: 'definitions', label: 'Definitions' },
]

export default function UserManagementDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState('All')
  const [statusFilter, setStatusFilter] = useState('All')
  const [expandedTerm, setExpandedTerm] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/user-management/overview`),
      axios.get(`${API_URL}/api/user-management/breakdown`),
      axios.get(`${API_URL}/api/user-management/definitions`),
    ])
      .then(([ovRes, bdRes, dfRes]) => {
        setOverview(ovRes.data)
        setBreakdown(bdRes.data)
        setDefinitions(dfRes.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading User Management data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}

  /* ── Filtered user list ── */
  const allUsers = breakdown?.users || []
  const filteredUsers = allUsers.filter(u => {
    if (roleFilter !== 'All' && u.role !== roleFilter) return false
    if (statusFilter !== 'All' && u.status !== statusFilter) return false
    if (searchTerm && !u.name?.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !u.user_id?.toLowerCase().includes(searchTerm.toLowerCase())) return false
    return true
  })

  /* ── Tab renderers ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
          <KPI label="Total Users" value={kpis.total_users} color="#3b82f6" />
          <KPI label="Active Users" value={kpis.active_users} color="#10b981" />
          <KPI label="Inactive" value={kpis.inactive_users} color="#f59e0b" />
          <KPI label="Suspended" value={kpis.suspended_users} color="#ef4444" />
          <KPI label="Roles" value={kpis.roles_count} color="#8b5cf6" />
          <KPI label="Avg Sessions/User" value={kpis.avg_sessions_per_day} color="#06b6d4" />
        </div>
      </Card>

      <Card title="Users by Role">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={overview?.users_by_role || []} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(overview?.users_by_role || []).map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Users by Status">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={overview?.users_by_status || []} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(overview?.users_by_status || []).map((entry, i) => (
                <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Login Trend (30 Days)" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={overview?.login_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }}
                   tickFormatter={d => d?.slice(5)} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="logins" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Users by Department" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overview?.users_by_department || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )

  const renderUsers = () => (
    <div>
      <Card title={`User Directory (${filteredUsers.length} of ${allUsers.length})`}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
          <input
            type="text" placeholder="Search by name or ID..."
            value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, minWidth: 200 }}
          />
          <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)}
                  style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
            <option value="All">All Roles</option>
            {['Clinician', 'Researcher', 'Technician', 'Admin', 'Patient'].map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}
                  style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
            <option value="All">All Statuses</option>
            {['Active', 'Inactive', 'Suspended'].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['ID', 'Name', 'Role', 'Department', 'Status', 'Last Login', 'Sessions (30d)'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map(u => (
                <tr key={u.user_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{u.user_id}</td>
                  <td style={{ padding: '8px 10px' }}>{u.name}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <Badge text={u.role} color={COLORS[['Clinician','Researcher','Technician','Admin','Patient'].indexOf(u.role) % COLORS.length]} />
                  </td>
                  <td style={{ padding: '8px 10px', color: '#64748b' }}>{u.department}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <Badge text={u.status} color={STATUS_COLORS[u.status] || '#64748b'} />
                  </td>
                  <td style={{ padding: '8px 10px', color: '#64748b' }}>{u.last_login}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600 }}>{u.sessions_30d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderRoles = () => {
    const perms = overview?.role_permissions || []
    const permKeys = perms.length > 0
      ? Object.keys(perms[0]).filter(k => k !== 'role')
      : []

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Role Permissions Matrix" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Role</th>
                  {permKeys.map(k => (
                    <th key={k} style={{ padding: '8px 10px', textAlign: 'center', color: '#475569', fontWeight: 600, fontSize: 11 }}>
                      {k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {perms.map(row => (
                  <tr key={row.role} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>
                      <Badge text={row.role} color={COLORS[['Clinician','Researcher','Technician','Admin','Patient'].indexOf(row.role) % COLORS.length]} />
                    </td>
                    {permKeys.map(k => (
                      <td key={k} style={{ padding: '8px 10px', textAlign: 'center', fontSize: 16 }}>
                        {row[k] ? '\u2705' : '\u274C'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Role Summary">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {Object.entries(breakdown?.role_summary || {}).map(([role, info]) => (
              <div key={role} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{role}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Total: {info.total} | Active: {info.active} | Avg Sessions: {info.avg_sessions}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const renderActivity = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Login Trend (30 Days)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={overview?.login_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d?.slice(5)} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="logins" stroke="#3b82f6" strokeWidth={2} name="Daily Logins" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Sessions by Role">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={Object.entries(breakdown?.role_summary || {}).map(([role, info]) => ({
            role, avg_sessions: info.avg_sessions, total: info.total
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="role" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="avg_sessions" fill="#06b6d4" name="Avg Sessions (30d)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Active vs Inactive by Role">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={Object.entries(breakdown?.role_summary || {}).map(([role, info]) => ({
            role, active: info.active, inactive: info.total - info.active
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="role" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="active" stackId="a" fill="#10b981" name="Active" />
            <Bar dataKey="inactive" stackId="a" fill="#f59e0b" name="Inactive/Suspended" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recent Activity" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Name', 'Role', 'Last Login', 'Sessions'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...allUsers].sort((a, b) => b.last_login?.localeCompare(a.last_login)).slice(0, 15).map(u => (
                <tr key={u.user_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px' }}>{u.name}</td>
                  <td style={{ padding: '8px 10px' }}><Badge text={u.role} color={COLORS[['Clinician','Researcher','Technician','Admin','Patient'].indexOf(u.role) % COLORS.length]} /></td>
                  <td style={{ padding: '8px 10px', color: '#64748b' }}>{u.last_login}</td>
                  <td style={{ padding: '8px 10px', fontWeight: 600, textAlign: 'center' }}>{u.sessions_30d}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  const renderDefinitions = () => {
    const terms = definitions?.terms || []
    return (
      <Card title="User Management Glossary">
        <div style={{ display: 'grid', gap: 8 }}>
          {terms.map((t, i) => (
            <div key={i}
                 onClick={() => setExpandedTerm(expandedTerm === i ? null : i)}
                 style={{
                   padding: '10px 14px', background: expandedTerm === i ? '#f0f9ff' : '#f8fafc',
                   borderRadius: 8, cursor: 'pointer', border: '1px solid',
                   borderColor: expandedTerm === i ? '#bfdbfe' : '#f1f5f9',
                   transition: 'all 0.15s'
                 }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>
                {expandedTerm === i ? '\u25BC' : '\u25B6'} {t.term}
              </div>
              {expandedTerm === i && (
                <div style={{ marginTop: 6, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>
                  {t.definition}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    )
  }

  const renderTab = () => {
    switch (tab) {
      case 'overview': return renderOverview()
      case 'users': return renderUsers()
      case 'roles': return renderRoles()
      case 'activity': return renderActivity()
      case 'definitions': return renderDefinitions()
      default: return renderOverview()
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>User Management</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Platform user accounts, roles, permissions, and activity monitoring
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
              background: tab === t.id ? '#3b82f6' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {renderTab()}
    </div>
  )
}
