import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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
  const s = String(status || '').toLowerCase()
  const color = s === 'active' ? '#10b981' : s === 'inactive' ? '#ef4444'
    : s === 'healthy' ? '#10b981' : s === 'degraded' ? '#f59e0b'
    : s === 'down' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{status || '--'}</span>
  )
}

function RoleBadge({ role }) {
  const r = String(role || '')
  const color = r === 'Admin' ? '#ef4444' : r === 'Neurologist' ? '#3b82f6'
    : r === 'EEG Tech' ? '#06b6d4' : r === 'Researcher' ? '#8b5cf6'
    : r === 'Nurse' ? '#10b981' : r === 'Data Scientist' ? '#f59e0b'
    : r === 'IT Support' ? '#64748b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{role || '--'}</span>
  )
}

function ToggleBadge({ enabled }) {
  const on = enabled === 1 || enabled === true
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: on ? '#10b98122' : '#ef444422', color: on ? '#10b981' : '#ef4444',
      fontWeight: 600, fontSize: 12
    }}>{on ? 'Enabled' : 'Disabled'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'users', label: 'Users' },
  { id: 'flags', label: 'Feature Flags' },
  { id: 'health', label: 'System Health' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AdminPanelDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/admin-panel/overview`),
      axios.get(`${API_URL}/api/admin-panel/breakdown`),
      axios.get(`${API_URL}/api/admin-panel/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading admin panel data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Admin Panel Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        System administration — {fmt(kpis.total_users)} users, {fmt(kpis.flags_enabled + kpis.flags_disabled)} feature flags, {fmt(kpis.system_uptime_pct)}% uptime, {fmt(kpis.config_entries)} config entries
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'users' && <UsersTab users={breakdown?.users || []} />}
      {tab === 'flags' && <FlagsTab flags={breakdown?.flags || []} />}
      {tab === 'health' && <HealthTab checks={breakdown?.health_checks || []} configs={breakdown?.configs || []} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

/* ─── Overview Tab ─────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return null
  const kpis = data.kpis || {}
  const roleData = data.users_by_role || []
  const statusData = data.users_by_status || []
  const flagCats = data.flags_by_category || []
  const healthTrend = (data.health_trend || []).map(h => ({
    ...h,
    date: h.timestamp ? h.timestamp.slice(5, 10) : ''
  }))
  const compHealth = data.component_health || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Total Users" value={kpis.total_users} color="#3b82f6" />
          <KPI label="Active Users" value={kpis.active_users} color="#10b981" />
          <KPI label="MFA Adoption" value={kpis.mfa_adoption_pct} sub="%" color="#8b5cf6" />
          <KPI label="System Uptime" value={kpis.system_uptime_pct} sub="%" color="#10b981" />
        </div>
      </Card>

      <Card span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Flags Enabled" value={kpis.flags_enabled} color="#10b981" />
          <KPI label="Flags Disabled" value={kpis.flags_disabled} color="#ef4444" />
          <KPI label="Avg Response Time" value={kpis.avg_response_time_ms} sub="ms" color="#f59e0b" />
          <KPI label="Config Entries" value={kpis.config_entries} color="#64748b" />
        </div>
      </Card>

      <Card title="Users by Role" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={roleData} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={90} label={({ role, count }) => `${role}: ${count}`}>
              {roleData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Users by Status" span={1}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={statusData} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
              {statusData.map((_, i) => <Cell key={i} fill={i === 0 ? '#10b981' : '#ef4444'} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Feature Flags by Category" span={1}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={flagCats} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="category" width={100} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="System Health Trend (Response Time + CPU + Memory)" span={4}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={healthTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="ms" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="pct" orientation="right" tick={{ fontSize: 11 }} domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Line yAxisId="ms" dataKey="response_time_ms" stroke="#3b82f6" name="Response (ms)" dot={false} />
            <Line yAxisId="pct" dataKey="cpu_pct" stroke="#f59e0b" name="CPU %" dot={false} />
            <Line yAxisId="pct" dataKey="memory_pct" stroke="#ef4444" name="Memory %" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Component Health (Avg Response Time & Error Rate)" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={compHealth}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="component" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="ms" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="pct" orientation="right" tick={{ fontSize: 11 }} domain={[0, 10]} />
            <Tooltip />
            <Legend />
            <Bar yAxisId="ms" dataKey="avg_response_time" fill="#3b82f6" name="Avg Response (ms)" radius={[6, 6, 0, 0]} />
            <Bar yAxisId="pct" dataKey="error_rate" fill="#ef4444" name="Error Rate %" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ─── Users Tab ────────────────────────────────────────────────── */
function UsersTab({ users }) {
  const [sort, setSort] = useState('user_id')
  const [dir, setDir] = useState(1)
  const [filter, setFilter] = useState('')

  const toggle = col => { if (sort === col) setDir(-dir); else { setSort(col); setDir(1) } }
  const arrow = col => sort === col ? (dir === 1 ? ' ▲' : ' ▼') : ''

  const filtered = users.filter(u => {
    const q = filter.toLowerCase()
    return !q || u.full_name?.toLowerCase().includes(q) || u.role?.toLowerCase().includes(q)
      || u.department?.toLowerCase().includes(q) || u.status?.toLowerCase().includes(q)
  })
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sort], bv = b[sort]
    if (av == null) return 1; if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir
  })

  const th = { padding: '8px 10px', textAlign: 'left', cursor: 'pointer', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap', userSelect: 'none' }
  const td = { padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <Card>
      <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter by name, role, department, status..."
        style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, marginBottom: 12 }} />
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={th} onClick={() => toggle('user_id')}>ID{arrow('user_id')}</th>
              <th style={th} onClick={() => toggle('full_name')}>Name{arrow('full_name')}</th>
              <th style={th} onClick={() => toggle('role')}>Role{arrow('role')}</th>
              <th style={th} onClick={() => toggle('department')}>Department{arrow('department')}</th>
              <th style={th} onClick={() => toggle('status')}>Status{arrow('status')}</th>
              <th style={th} onClick={() => toggle('login_count')}>Logins{arrow('login_count')}</th>
              <th style={th} onClick={() => toggle('mfa_enabled')}>MFA{arrow('mfa_enabled')}</th>
              <th style={th} onClick={() => toggle('last_login')}>Last Login{arrow('last_login')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(u => (
              <tr key={u.user_id}>
                <td style={td}>{u.user_id}</td>
                <td style={td}>{u.full_name}</td>
                <td style={td}><RoleBadge role={u.role} /></td>
                <td style={td}>{u.department}</td>
                <td style={td}><StatusBadge status={u.status} /></td>
                <td style={{ ...td, textAlign: 'right' }}>{fmt(u.login_count)}</td>
                <td style={td}><ToggleBadge enabled={u.mfa_enabled} /></td>
                <td style={td}>{u.last_login ? u.last_login.slice(0, 16).replace('T', ' ') : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ─── Feature Flags Tab ────────────────────────────────────────── */
function FlagsTab({ flags }) {
  const [sort, setSort] = useState('flag_id')
  const [dir, setDir] = useState(1)
  const [filter, setFilter] = useState('')

  const toggle = col => { if (sort === col) setDir(-dir); else { setSort(col); setDir(1) } }
  const arrow = col => sort === col ? (dir === 1 ? ' ▲' : ' ▼') : ''

  const filtered = flags.filter(f => {
    const q = filter.toLowerCase()
    return !q || f.name?.toLowerCase().includes(q) || f.category?.toLowerCase().includes(q)
      || f.owner?.toLowerCase().includes(q)
  })
  const sorted = [...filtered].sort((a, b) => {
    const av = a[sort], bv = b[sort]
    if (av == null) return 1; if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir
  })

  const th = { padding: '8px 10px', textAlign: 'left', cursor: 'pointer', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap', userSelect: 'none' }
  const td = { padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <Card>
      <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter by name, category, owner..."
        style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, marginBottom: 12 }} />
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={th} onClick={() => toggle('flag_id')}>ID{arrow('flag_id')}</th>
              <th style={th} onClick={() => toggle('name')}>Name{arrow('name')}</th>
              <th style={th} onClick={() => toggle('category')}>Category{arrow('category')}</th>
              <th style={th} onClick={() => toggle('enabled')}>Status{arrow('enabled')}</th>
              <th style={th} onClick={() => toggle('rollout_percentage')}>Rollout %{arrow('rollout_percentage')}</th>
              <th style={th} onClick={() => toggle('owner')}>Owner{arrow('owner')}</th>
              <th style={th} onClick={() => toggle('updated_at')}>Updated{arrow('updated_at')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(f => (
              <tr key={f.flag_id}>
                <td style={td}>{f.flag_id}</td>
                <td style={td}>{f.name}</td>
                <td style={td}>{f.category}</td>
                <td style={td}><ToggleBadge enabled={f.enabled} /></td>
                <td style={{ ...td, textAlign: 'right' }}>{fmt(f.rollout_percentage)}%</td>
                <td style={td}>{f.owner}</td>
                <td style={td}>{f.updated_at ? f.updated_at.slice(0, 16).replace('T', ' ') : '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

/* ─── System Health Tab ────────────────────────────────────────── */
function HealthTab({ checks, configs }) {
  const [sort, setSort] = useState('check_id')
  const [dir, setDir] = useState(-1)

  const toggle = col => { if (sort === col) setDir(-dir); else { setSort(col); setDir(1) } }
  const arrow = col => sort === col ? (dir === 1 ? ' ▲' : ' ▼') : ''

  const sorted = [...checks].sort((a, b) => {
    const av = a[sort], bv = b[sort]
    if (av == null) return 1; if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir
  })

  const th = { padding: '8px 10px', textAlign: 'left', cursor: 'pointer', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap', userSelect: 'none' }
  const td = { padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <>
      <Card title="Health Checks">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={th} onClick={() => toggle('check_id')}>ID{arrow('check_id')}</th>
                <th style={th} onClick={() => toggle('timestamp')}>Timestamp{arrow('timestamp')}</th>
                <th style={th} onClick={() => toggle('component')}>Component{arrow('component')}</th>
                <th style={th} onClick={() => toggle('status')}>Status{arrow('status')}</th>
                <th style={th} onClick={() => toggle('response_time_ms')}>Response (ms){arrow('response_time_ms')}</th>
                <th style={th} onClick={() => toggle('cpu_pct')}>CPU %{arrow('cpu_pct')}</th>
                <th style={th} onClick={() => toggle('memory_pct')}>Memory %{arrow('memory_pct')}</th>
                <th style={th} onClick={() => toggle('disk_pct')}>Disk %{arrow('disk_pct')}</th>
                <th style={th} onClick={() => toggle('error_count')}>Errors{arrow('error_count')}</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(c => (
                <tr key={c.check_id}>
                  <td style={td}>{c.check_id}</td>
                  <td style={td}>{c.timestamp ? c.timestamp.slice(0, 16).replace('T', ' ') : '--'}</td>
                  <td style={td}>{c.component}</td>
                  <td style={td}><StatusBadge status={c.status} /></td>
                  <td style={{ ...td, textAlign: 'right' }}>{fmt(c.response_time_ms)}</td>
                  <td style={{ ...td, textAlign: 'right' }}>{fmt(c.cpu_pct)}</td>
                  <td style={{ ...td, textAlign: 'right' }}>{fmt(c.memory_pct)}</td>
                  <td style={{ ...td, textAlign: 'right' }}>{fmt(c.disk_pct)}</td>
                  <td style={{ ...td, textAlign: 'right' }}>{fmt(c.error_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div style={{ marginTop: 16 }}>
        <Card title="Configuration Entries">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ ...th, cursor: 'default' }}>Key</th>
                  <th style={{ ...th, cursor: 'default' }}>Value</th>
                  <th style={{ ...th, cursor: 'default' }}>Category</th>
                  <th style={{ ...th, cursor: 'default' }}>Description</th>
                  <th style={{ ...th, cursor: 'default' }}>Updated</th>
                  <th style={{ ...th, cursor: 'default' }}>Updated By</th>
                </tr>
              </thead>
              <tbody>
                {configs.map(c => (
                  <tr key={c.config_id}>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: 12 }}>{c.key}</td>
                    <td style={{ ...td, fontFamily: 'monospace', fontSize: 12 }}>{c.value}</td>
                    <td style={td}>{c.category}</td>
                    <td style={{ ...td, fontSize: 12, color: '#64748b' }}>{c.description}</td>
                    <td style={td}>{c.updated_at ? c.updated_at.slice(0, 16).replace('T', ' ') : '--'}</td>
                    <td style={td}>{c.updated_by}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  )
}

/* ─── Definitions Tab ──────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data || !Array.isArray(data)) return null
  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {data.map((d, i) => (
        <Card key={i}>
          <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>{d.title}</h4>
          <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{d.description}</p>
        </Card>
      ))}
    </div>
  )
}
