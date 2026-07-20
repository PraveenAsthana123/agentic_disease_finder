import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b', '#795548']
const STATUS_COLORS = { built: '#4caf50', partial: '#ff9800', planned: '#f44336' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
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

export default function RoleChallengesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/role-challenges/overview`),
      axios.get(`${API_URL}/role-challenges/breakdown`),
      axios.get(`${API_URL}/role-challenges/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Role Challenges...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Role Challenges data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'all', label: 'All Roles' },
    { id: 'matrix', label: 'Challenge Matrix' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Per-Role Challenges &amp; AI Mitigation</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
        {s.total_roles} clinical roles, {s.total_challenges} workflow challenges, {s.built_pct}% AI-mitigated
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
              fontSize: 13, fontWeight: 500,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview tab */}
      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={s.total_roles} />
              <KPI label="Total Challenges" value={s.total_challenges} />
              <KPI label="Built" value={s.built} sub={`${s.built_pct}%`} />
              <KPI label="Partial" value={s.partial} />
              <KPI label="Planned" value={s.planned} />
              <KPI label="Roles Fully Built" value={s.roles_fully_built} sub={`of ${s.total_roles}`} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={overview.status_distribution}
                    cx="50%" cy="50%" outerRadius={80}
                    dataKey="value" nameKey="name"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {(overview.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[overview.status_distribution[i]?.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Challenges per Role (by status)">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.challenges_per_role} layout="vertical" margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="name" width={95} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="built" stackId="a" fill={STATUS_COLORS.built} name="Built" />
                  <Bar dataKey="partial" stackId="a" fill={STATUS_COLORS.partial} name="Partial" />
                  <Bar dataKey="planned" stackId="a" fill={STATUS_COLORS.planned} name="Planned" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Role Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['', 'Role', 'Total', 'Built', 'Partial', 'Planned'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.role_summary || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontSize: 18 }}>{r.icon}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{r.role}</td>
                      <td style={{ padding: '6px 10px' }}>{r.total}</td>
                      <td style={{ padding: '6px 10px', color: STATUS_COLORS.built }}>{r.built}</td>
                      <td style={{ padding: '6px 10px', color: STATUS_COLORS.partial }}>{r.partial || '-'}</td>
                      <td style={{ padding: '6px 10px', color: STATUS_COLORS.planned }}>{r.planned || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* All Roles tab */}
      {tab === 'all' && breakdown?.roles && (
        <>
          {breakdown.roles.map((role, ri) => (
            <Card key={ri} title={`${role.icon} ${role.role}`}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['#', 'Challenge', 'AI Mitigation', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {role.items.map((item, ii) => (
                    <tr key={ii} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{ii + 1}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 300 }}>{item.challenge}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 350, color: '#475569' }}>
                        {item.ai}
                        {item.dashboard && <span style={{ display: 'block', fontSize: 10, color: '#1e88e5', marginTop: 2 }}>{item.dashboard}</span>}
                      </td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={item.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </>
      )}

      {/* Challenge Matrix tab */}
      {tab === 'matrix' && breakdown?.roles && (
        <Card title="Full Challenge Matrix">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Role', 'Challenge', 'AI Mitigation', 'Status', 'Dashboard / Note'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.roles.flatMap((role) =>
                  role.items.map((item, ii) => (
                    <tr key={`${role.role}-${ii}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      {ii === 0 ? (
                        <td style={{ padding: '6px 10px', fontWeight: 500, verticalAlign: 'top' }} rowSpan={role.items.length}>
                          {role.icon} {role.role}
                        </td>
                      ) : null}
                      <td style={{ padding: '6px 10px', maxWidth: 250 }}>{item.challenge}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 280, color: '#475569' }}>{item.ai}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={item.status} /></td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>
                        {item.dashboard || item.solution || item.note || '-'}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs?.available && (
        <>
          <Card title="Role Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Role', 'Description'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.role_descriptions || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 500, whiteSpace: 'nowrap' }}>{r.role}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Status Legend">
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ flex: '1 1 200px', padding: 12, background: '#f8fafc', borderRadius: 6 }}>
                  <StatusBadge status={s.status} />
                  <p style={{ fontSize: 12, color: '#475569', marginTop: 6, marginBottom: 0 }}>{s.description}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Term', 'Definition'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {(defs.clinical_notes || []).length > 0 && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.clinical_notes.map((n, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
                ))}
              </ul>
            </Card>
          )}

          {(defs.references || []).length > 0 && (
            <Card title="References">
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {defs.references.map((r, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{r}</li>
                ))}
              </ol>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
