import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const STATUS_COLORS = { pass: '#4caf50', built: '#4caf50', partial: '#ff9800', planned: '#f44336', 'N/A': '#e0e0e0' }

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

export default function RoleTestsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/role-tests/overview`),
      axios.get(`${API_URL}/api/role-tests/breakdown`),
      axios.get(`${API_URL}/api/role-tests/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Role Tests...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Role Tests data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'roles', label: 'By Role' },
    { id: 'matrix', label: 'Test Matrix' },
    { id: 'all', label: 'All Tests' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Per-Role Testing Matrix</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>
        {s.total_roles} roles, {s.total_tests} test cases, {s.total_dims} dimensions, {s.pass_pct}% passing
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569',
            fontSize: 12, fontWeight: 600,
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={s.total_roles} />
              <KPI label="Total Tests" value={s.total_tests} />
              <KPI label="Dimensions" value={s.total_dims} />
              <KPI label="Passed" value={s.passed} sub={`${s.pass_pct}%`} />
              <KPI label="Partial" value={s.partial} />
              <KPI label="Planned" value={s.planned} />
              <KPI label="Roles All-Pass" value={s.roles_all_pass} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Test Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((d, i) => (
                      <Cell key={i} fill={STATUS_COLORS[d.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Tests per Role (by status)">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.tests_per_role} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="pass" stackId="a" fill="#4caf50" name="Pass" />
                  <Bar dataKey="partial" stackId="a" fill="#ff9800" name="Partial" />
                  <Bar dataKey="planned" stackId="a" fill="#f44336" name="Planned" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Tests per Dimension (by status)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.tests_per_dim}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="pass" stackId="a" fill="#4caf50" name="Pass" />
                <Bar dataKey="partial" stackId="a" fill="#ff9800" name="Partial" />
                <Bar dataKey="planned" stackId="a" fill="#f44336" name="Planned" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Role Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Role</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#4caf50' }}>Pass</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#ff9800' }}>Partial</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#f44336' }}>Planned</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.role_summary || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.role}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px' }}>{r.total}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px', color: '#4caf50', fontWeight: 600 }}>{r.pass}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px', color: '#ff9800', fontWeight: 600 }}>{r.partial}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px', color: '#f44336', fontWeight: 600 }}>{r.planned}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* By Role Tab */}
      {tab === 'roles' && breakdown?.roles && (
        <>
          {breakdown.roles.map((role, ri) => (
            <Card key={ri} title={role.role}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Dimension</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Test Case</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Maps To</th>
                  </tr>
                </thead>
                <tbody>
                  {role.tests.map((t, ti) => (
                    <tr key={ti} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{t.dim}</td>
                      <td style={{ padding: '6px 8px' }}>{t.case}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px' }}><StatusBadge status={t.status} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{t.maps_to || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </>
      )}

      {/* Test Matrix Tab */}
      {tab === 'matrix' && breakdown?.matrix && (
        <Card title="Role x Dimension Matrix">
          <p style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>
            Cross-reference of {s.total_roles} roles against {s.total_dims} test dimensions — cell shows status.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', minWidth: 130 }}>Role</th>
                  {(breakdown.dimensions || []).map(dim => (
                    <th key={dim} style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', minWidth: 70 }}>{dim}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.matrix.map((row, ri) => (
                  <tr key={ri} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.role}</td>
                    {(breakdown.dimensions || []).map(dim => (
                      <td key={dim} style={{ textAlign: 'center', padding: '6px 8px' }}>
                        {row[dim] === 'N/A'
                          ? <span style={{ color: '#d1d5db', fontSize: 11 }}>—</span>
                          : <StatusBadge status={row[dim]} />}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* All Tests Tab */}
      {tab === 'all' && breakdown?.roles && (
        <Card title={`All ${s.total_tests} Test Cases`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Role</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Dim</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Test Case</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Maps To</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.roles.flatMap((role, ri) =>
                  role.tests.map((t, ti) => (
                    <tr key={`${ri}-${ti}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: ti === 0 ? 600 : 400 }}>{ti === 0 ? role.role : ''}</td>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{t.dim}</td>
                      <td style={{ padding: '6px 8px' }}>{t.case}</td>
                      <td style={{ textAlign: 'center', padding: '6px 8px' }}><StatusBadge status={t.status} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{t.maps_to || '—'}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs?.available && (
        <>
          <Card title="Test Dimensions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Dimension</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.dimension_descriptions || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{d.dim}</td>
                    <td style={{ padding: '6px 8px' }}>{d.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Status</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.status_legend || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px' }}><StatusBadge status={s.status} /></td>
                    <td style={{ padding: '6px 8px' }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Term</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{g.term}</td>
                    <td style={{ padding: '6px 8px' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ marginBottom: 6, fontSize: 12, color: '#475569' }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ marginBottom: 4, fontSize: 12, color: '#475569' }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
