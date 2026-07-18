import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
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

function RoleBadge({ role }) {
  const colors = {
    Neurologist: '#3b82f6', 'EEG Tech': '#8b5cf6', Nurse: '#10b981',
    Researcher: '#f59e0b', Admin: '#ef4444', 'Data Scientist': '#06b6d4'
  }
  const c = colors[role] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: c + '18', color: c
    }}>{role}</span>
  )
}

function StatusBadge({ status }) {
  const c = status === 'active' ? '#10b981' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: c + '18', color: c
    }}>{status}</span>
  )
}

function MFABadge({ enabled }) {
  const c = enabled ? '#10b981' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: c + '18', color: c
    }}>{enabled ? '✓ MFA' : '✗ No MFA'}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'roles', label: 'Roles' },
  { id: 'users', label: 'Users' },
  { id: 'security', label: 'Security' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RBACDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/rbac/overview`),
      axios.get(`${API_URL}/api/rbac/breakdown`),
      axios.get(`${API_URL}/api/rbac/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading RBAC data…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No RBAC data available.</div>

  const k = overview.kpis

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>🔑 RBAC Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Role-Based Access Control — user roles, permissions, MFA compliance, access audit
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* KPIs */}
          <Card title="Key Metrics" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
              <KPI label="Total Users" value={k.total_users} />
              <KPI label="Active" value={k.active_users} color="#10b981" />
              <KPI label="Inactive" value={k.inactive_users} color="#ef4444" />
              <KPI label="Roles" value={k.roles} color="#3b82f6" />
              <KPI label="Departments" value={k.departments} color="#8b5cf6" />
              <KPI label="MFA Enabled" value={k.mfa_enabled} sub={`${k.mfa_rate}%`} color="#10b981" />
              <KPI label="Transactions" value={k.total_transactions.toLocaleString()} color="#f59e0b" />
            </div>
          </Card>

          {/* Role distribution pie */}
          <Card title="Role Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.role_distribution} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={80} label={({ role, count }) => `${role}: ${count}`}>
                  {overview.role_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Department distribution bar */}
          <Card title="Department Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.dept_distribution} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="department" width={80} style={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Login leaders */}
          <Card title="Top Login Activity" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.login_leaders} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="user" style={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="logins" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Component access */}
          <Card title="Component Access Frequency">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.component_access} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="component" width={100} style={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="accesses" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Actions by actor */}
          <Card title="Actions by Actor" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.action_by_actor} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="actor" style={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="actions" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'roles' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Role Summaries" span={3}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Role', 'Total', 'Active', 'Inactive', 'Avg Logins', 'MFA Rate', 'Departments'].map(h => (
                    <th key={h} style={{ padding: '10px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.role_summaries.map(r => (
                  <tr key={r.role} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px' }}><RoleBadge role={r.role} /></td>
                    <td style={{ padding: '8px' }}>{r.total}</td>
                    <td style={{ padding: '8px', color: '#10b981' }}>{r.active}</td>
                    <td style={{ padding: '8px', color: r.inactive > 0 ? '#ef4444' : '#64748b' }}>{r.inactive}</td>
                    <td style={{ padding: '8px' }}>{r.avg_logins}</td>
                    <td style={{ padding: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ flex: 1, height: 6, background: '#f1f5f9', borderRadius: 3 }}>
                          <div style={{ width: `${r.mfa_rate}%`, height: '100%', background: r.mfa_rate === 100 ? '#10b981' : '#f59e0b', borderRadius: 3 }} />
                        </div>
                        <span style={{ fontSize: 11 }}>{r.mfa_rate}%</span>
                      </div>
                    </td>
                    <td style={{ padding: '8px', fontSize: 11, color: '#64748b' }}>{r.departments.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Access Matrix from definitions */}
          {definitions && (
            <Card title="Access Matrix (R=Read, W=Write, —=None, R*=De-identified)" span={3}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Resource</th>
                    {['Neurologist', 'EEG Tech', 'Nurse', 'Researcher', 'Admin', 'Data Scientist'].map(r => (
                      <th key={r} style={{ padding: '8px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', color: '#475569', fontSize: 11 }}>{r}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {definitions.access_matrix.map(row => (
                    <tr key={row.resource} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px', fontWeight: 600 }}>{row.resource}</td>
                      {['Neurologist', 'EEG Tech', 'Nurse', 'Researcher', 'Admin', 'Data Scientist'].map(r => (
                        <td key={r} style={{
                          padding: '8px', textAlign: 'center', fontWeight: 600,
                          color: row[r] === 'RW' ? '#10b981' : row[r] === 'R' ? '#3b82f6' : row[r] === 'R*' ? '#f59e0b' : '#cbd5e1'
                        }}>{row[r]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}

      {tab === 'users' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`All Users (${breakdown.user_list.length})`}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Name', 'Username', 'Role', 'Department', 'Status', 'MFA', 'Logins', 'Last Login'].map(h => (
                    <th key={h} style={{ padding: '10px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.user_list.map(u => (
                  <tr key={u.user_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 500 }}>{u.full_name}</td>
                    <td style={{ padding: '8px', color: '#64748b', fontFamily: 'monospace', fontSize: 11 }}>{u.username}</td>
                    <td style={{ padding: '8px' }}><RoleBadge role={u.role} /></td>
                    <td style={{ padding: '8px', fontSize: 11 }}>{u.department}</td>
                    <td style={{ padding: '8px' }}><StatusBadge status={u.status} /></td>
                    <td style={{ padding: '8px' }}><MFABadge enabled={u.mfa_enabled} /></td>
                    <td style={{ padding: '8px' }}>{u.login_count}</td>
                    <td style={{ padding: '8px', fontSize: 11, color: '#64748b' }}>{u.last_login?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {tab === 'security' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* Inactive users alert */}
          <Card title="⚠️ Inactive Accounts" span={2}>
            {breakdown.inactive_users.length === 0 ? (
              <p style={{ color: '#10b981', fontWeight: 600 }}>No inactive accounts — all clear.</p>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Name', 'Role', 'Department', 'Last Login', 'MFA'].map(h => (
                      <th key={h} style={{ padding: '8px', textAlign: 'left', borderBottom: '2px solid #fecaca', color: '#991b1b', fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.inactive_users.map(u => (
                    <tr key={u.user_id} style={{ borderBottom: '1px solid #fef2f2' }}>
                      <td style={{ padding: '8px' }}>{u.full_name}</td>
                      <td style={{ padding: '8px' }}><RoleBadge role={u.role} /></td>
                      <td style={{ padding: '8px' }}>{u.department}</td>
                      <td style={{ padding: '8px', fontSize: 11 }}>{u.last_login?.slice(0, 16)}</td>
                      <td style={{ padding: '8px' }}><MFABadge enabled={u.mfa_enabled} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {/* No MFA alert */}
          <Card title="🔓 Users Without MFA" span={2}>
            {breakdown.no_mfa_users.length === 0 ? (
              <p style={{ color: '#10b981', fontWeight: 600 }}>All users have MFA enabled — 100% compliance.</p>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fffbeb' }}>
                    {['Name', 'Role', 'Department', 'Status', 'Logins'].map(h => (
                      <th key={h} style={{ padding: '8px', textAlign: 'left', borderBottom: '2px solid #fde68a', color: '#92400e', fontSize: 11 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.no_mfa_users.map(u => (
                    <tr key={u.user_id} style={{ borderBottom: '1px solid #fffbeb' }}>
                      <td style={{ padding: '8px' }}>{u.full_name}</td>
                      <td style={{ padding: '8px' }}><RoleBadge role={u.role} /></td>
                      <td style={{ padding: '8px' }}>{u.department}</td>
                      <td style={{ padding: '8px' }}><StatusBadge status={u.status} /></td>
                      <td style={{ padding: '8px' }}>{u.login_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {/* Recent transactions */}
          <Card title="Recent Access Log" span={3}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Actor', 'Action', 'Component', 'Timestamp (UTC)'].map(h => (
                    <th key={h} style={{ padding: '8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.recent_transactions.slice(0, 20).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 500 }}>{t.actor}</td>
                    <td style={{ padding: '8px' }}>{t.action}</td>
                    <td style={{ padding: '8px', fontFamily: 'monospace', fontSize: 11 }}>{t.component}</td>
                    <td style={{ padding: '8px', fontSize: 11, color: '#64748b' }}>{t.ts_utc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Security policies */}
          {definitions && (
            <Card title="Security Policies" span={2}>
              <div style={{ display: 'grid', gap: 8 }}>
                {definitions.security_policies.map(p => (
                  <div key={p.policy} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.policy}</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{p.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Role Descriptions" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {definitions.roles.map(r => (
                <div key={r.role} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}><RoleBadge role={r.role} /></div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 6 }}>{r.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Permission Levels">
            <div style={{ display: 'grid', gap: 8 }}>
              {definitions.permissions.map(p => (
                <div key={p.level} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.level}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{p.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {definitions.glossary.map(g => (
                <div key={g.term} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
              {definitions.clinical_notes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
