import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TYPE_COLORS = {
  clinical: '#3b82f6',
  admin: '#8b5cf6',
  'cross-functional': '#10b981',
}

const STATUS_COLORS = {
  Active: '#10b981',
  Archived: '#94a3b8',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'groups', label: 'Groups' },
  { id: 'permissions', label: 'Permissions' },
  { id: 'membership', label: 'Membership' },
  { id: 'definitions', label: 'Definitions' },
]

export default function GroupsTeamsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState('All')
  const [expandedGroup, setExpandedGroup] = useState(null)
  const [expandedTerm, setExpandedTerm] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/groups-teams/overview`),
      axios.get(`${API_URL}/api/groups-teams/breakdown`),
      axios.get(`${API_URL}/api/groups-teams/definitions`),
    ])
      .then(([ovRes, bdRes, dfRes]) => {
        setOverview(ovRes.data)
        setBreakdown(bdRes.data)
        setDefinitions(dfRes.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Groups & Teams data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}

  /* -- Filtered groups -- */
  const allGroups = breakdown?.groups || []
  const filteredGroups = allGroups.filter(g => {
    if (typeFilter !== 'All' && g.type !== typeFilter) return false
    if (searchTerm && !g.name?.toLowerCase().includes(searchTerm.toLowerCase()) &&
        !g.group_id?.toLowerCase().includes(searchTerm.toLowerCase())) return false
    return true
  })

  /* -- Tab renderers -- */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
          <KPI label="Total Groups" value={kpis.total_groups} color="#3b82f6" />
          <KPI label="Active Groups" value={kpis.active_groups} color="#10b981" />
          <KPI label="Archived" value={kpis.archived_groups} color="#94a3b8" />
          <KPI label="Total Memberships" value={kpis.total_memberships} color="#8b5cf6" />
          <KPI label="Avg Members/Group" value={kpis.avg_members_per_group} color="#06b6d4" />
          <KPI label="Cross-Functional" value={kpis.cross_functional_count} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Groups by Type">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={overview?.groups_by_type || []} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {(overview?.groups_by_type || []).map((entry, i) => (
                <Cell key={i} fill={TYPE_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Group Sizes">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overview?.groups_by_size || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="members" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Membership Trend (30 Days)" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={overview?.membership_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d?.slice(5)} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="memberships" stroke="#8b5cf6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )

  const renderGroups = () => (
    <div>
      <Card title={`Group Directory (${filteredGroups.length} of ${allGroups.length})`}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
          <input
            type="text" placeholder="Search groups..."
            value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, minWidth: 200 }}
          />
          <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)}
                  style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
            <option value="All">All Types</option>
            {['clinical', 'admin', 'cross-functional'].map(t => (
              <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
            ))}
          </select>
        </div>

        <div style={{ display: 'grid', gap: 12 }}>
          {filteredGroups.map(g => (
            <div key={g.group_id}
                 onClick={() => setExpandedGroup(expandedGroup === g.group_id ? null : g.group_id)}
                 style={{
                   padding: 16, background: expandedGroup === g.group_id ? '#f0f9ff' : '#f8fafc',
                   borderRadius: 10, cursor: 'pointer', border: '1px solid',
                   borderColor: expandedGroup === g.group_id ? '#bfdbfe' : '#f1f5f9',
                   transition: 'all 0.15s'
                 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <span style={{ fontWeight: 600, fontSize: 15, color: '#1e293b' }}>
                    {expandedGroup === g.group_id ? '\u25BC' : '\u25B6'} {g.name}
                  </span>
                  <span style={{ marginLeft: 10 }}>
                    <Badge text={g.type} color={TYPE_COLORS[g.type] || '#64748b'} />
                  </span>
                  <span style={{ marginLeft: 8 }}>
                    <Badge text={g.status} color={STATUS_COLORS[g.status] || '#64748b'} />
                  </span>
                </div>
                <div style={{ fontSize: 13, color: '#64748b' }}>
                  {g.member_count} members | Lead: {g.lead}
                </div>
              </div>
              {g.description && (
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{g.description}</div>
              )}
              {expandedGroup === g.group_id && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 6 }}>Members</div>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                          {['ID', 'Name', 'Role'].map(h => (
                            <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(g.members || []).map(m => (
                          <tr key={m.user_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{m.user_id}</td>
                            <td style={{ padding: '6px 8px' }}>{m.name}</td>
                            <td style={{ padding: '6px 8px' }}>
                              <Badge text={m.role} color={COLORS[['Clinician','Researcher','Technician','Admin','Patient'].indexOf(m.role) % COLORS.length]} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginTop: 12, marginBottom: 6 }}>Permissions</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {(g.permissions || []).map(p => (
                      <Badge key={p} text={p.replace(/_/g, ' ')} color="#3b82f6" />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    </div>
  )

  const renderPermissions = () => {
    const matrix = overview?.permission_matrix || []
    const permKeys = matrix.length > 0
      ? Object.keys(matrix[0]).filter(k => k !== 'group')
      : []

    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Group Permissions Matrix" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Group</th>
                  {permKeys.map(k => (
                    <th key={k} style={{ padding: '8px 10px', textAlign: 'center', color: '#475569', fontWeight: 600, fontSize: 11 }}>
                      {k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map(row => (
                  <tr key={row.group} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{row.group}</td>
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

        <Card title="Type Summary">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
            {Object.entries(breakdown?.type_summary || {}).map(([type, info]) => (
              <div key={type} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>
                  <Badge text={type} color={TYPE_COLORS[type] || '#64748b'} />
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  Total: {info.total} | Active: {info.active} | Avg Members: {info.avg_members}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const renderMembership = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Membership Trend (30 Days)" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={overview?.membership_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d?.slice(5)} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="memberships" stroke="#8b5cf6" strokeWidth={2} name="Active Memberships" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Members per Group">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={allGroups.map(g => ({ name: g.name, members: g.member_count }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="members" fill="#06b6d4" name="Members" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Groups per Type">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overview?.groups_by_type || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#f59e0b" name="Groups" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Memberships" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Group', 'Type', 'Status', 'Lead', 'Members', 'Created'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {allGroups.map(g => (
                <tr key={g.group_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{g.name}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <Badge text={g.type} color={TYPE_COLORS[g.type] || '#64748b'} />
                  </td>
                  <td style={{ padding: '8px 10px' }}>
                    <Badge text={g.status} color={STATUS_COLORS[g.status] || '#64748b'} />
                  </td>
                  <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.lead}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600 }}>{g.member_count}</td>
                  <td style={{ padding: '8px 10px', color: '#64748b' }}>{g.created}</td>
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
      <Card title="Groups & Teams Glossary">
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
      case 'groups': return renderGroups()
      case 'permissions': return renderPermissions()
      case 'membership': return renderMembership()
      case 'definitions': return renderDefinitions()
      default: return renderOverview()
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Groups & Teams</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Group membership, team composition, and group-level permissions
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
