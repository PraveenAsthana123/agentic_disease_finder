import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function RoleBadge({ role }) {
  const roleColors = {
    'Neurologist': '#1e88e5',
    'EEG Technician': '#4caf50',
    'Clinical Neurophysiologist': '#7c4dff',
    'Patient': '#ff9800',
    'AI Governance': '#f44336',
    'Pharmacist': '#00bcd4',
  }
  const bg = roleColors[role] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {role}
    </span>
  )
}

export default function RoleProcessFlowsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/role-process-flows/overview`),
      axios.get(`${API_URL}/api/role-process-flows/breakdown`),
      axios.get(`${API_URL}/api/role-process-flows/definitions`),
    ])
      .then(([o, b, d]) => { setOverview(o.data); setBreakdown(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Role Process Flows...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#94a3b8' }}>Role Process Flows data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'roles', label: 'All Roles' },
    { id: 'steps', label: 'All Steps' },
    { id: 'definitions', label: 'Definitions' },
  ]
  const s = overview.summary || {}

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>
        Role Process Flows Dashboard
      </h2>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#2563eb' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ────── OVERVIEW TAB ────── */}
      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={fmt(s.total_roles)} />
              <KPI label="Total Steps" value={fmt(s.total_steps)} />
              <KPI label="Avg Steps/Role" value={fmt(s.avg_steps_per_role)} />
              <KPI label="Most Steps" value={fmt(s.max_steps_count)} sub={s.max_steps_role} />
              <KPI label="Fewest Steps" value={fmt(s.min_steps_count)} sub={s.min_steps_role} />
              <KPI label="With Flowchart" value={fmt(s.with_mermaid)} />
              <KPI label="Default Steps" value={fmt(s.default_steps)} />
            </div>
          </Card>

          {/* Steps per role bar chart */}
          <Card title="Steps per Role">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={overview.steps_distribution || []} layout="vertical" margin={{ left: 120, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="role" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="steps" fill="#2563eb" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Role summary table */}
          <Card title="Role Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Role', 'Steps', 'First Step', 'Last Step', 'Flowchart'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.role_table || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.num_steps}</td>
                      <td style={{ padding: '6px 10px', color: '#64748b' }}>{r.first_step}</td>
                      <td style={{ padding: '6px 10px', color: '#64748b' }}>{r.last_step}</td>
                      <td style={{ padding: '6px 10px' }}>{r.has_mermaid ? '✓' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ────── ALL ROLES TAB ────── */}
      {tab === 'roles' && breakdown?.available && (
        <>
          {/* Default flow */}
          {breakdown.default_flow && breakdown.default_flow.steps?.length > 0 && (
            <Card title="Default Flow (applied to roles without specific process)">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                {breakdown.default_flow.steps.map((s, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      background: '#e0e7ff', color: '#3730a3', borderRadius: 6,
                      padding: '4px 10px', fontSize: 12, fontWeight: 600
                    }}>
                      {s.n}. {s.step}
                    </span>
                    {i < breakdown.default_flow.steps.length - 1 && (
                      <span style={{ color: '#94a3b8', fontSize: 16 }}>→</span>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Per-role cards */}
          {(breakdown.role_details || []).map((rd, ri) => (
            <Card key={ri} title={rd.role}>
              <div style={{ marginBottom: 8, fontSize: 12, color: '#64748b' }}>
                {rd.num_steps} steps
              </div>

              {/* Step flow visualization */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
                {(rd.steps || []).map((s, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{
                      background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0',
                      borderRadius: 6, padding: '4px 10px', fontSize: 12, fontWeight: 500
                    }}>
                      {s.n}. {s.step}
                    </span>
                    {i < rd.steps.length - 1 && (
                      <span style={{ color: '#94a3b8', fontSize: 16 }}>→</span>
                    )}
                  </div>
                ))}
              </div>

              {/* Step table */}
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, width: 40 }}>#</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>Step</th>
                  </tr>
                </thead>
                <tbody>
                  {(rd.steps || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 10px', color: '#94a3b8', fontWeight: 600 }}>{s.n}</td>
                      <td style={{ padding: '5px 10px' }}>{s.step}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </>
      )}

      {/* ────── ALL STEPS TAB ────── */}
      {tab === 'steps' && breakdown?.available && (
        <Card title={`All Steps (${(breakdown.all_steps || []).length} total)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Role', 'Step #', 'Step'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.all_steps || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><RoleBadge role={s.role} /></td>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#94a3b8' }}>{s.step_num}</td>
                    <td style={{ padding: '6px 10px' }}>{s.step}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ────── DEFINITIONS TAB ────── */}
      {tab === 'definitions' && defs?.available && (
        <>
          <Card title="Role Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Role', 'Description'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.role_descriptions || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}><RoleBadge role={r.role} /></td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Term', 'Definition'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569' }}>
              {(defs.clinical_notes || []).map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569' }}>
              {(defs.references || []).map((r, i) => <li key={i} style={{ marginBottom: 6 }}>{r}</li>)}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
