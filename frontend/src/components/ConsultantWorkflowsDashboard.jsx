import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
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

function TierBadge({ tier }) {
  const bg = tier === 1 ? '#4caf50' : '#1e88e5'
  const label = tier === 1 ? 'Tier 1' : 'Tier 2'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}

function SignoffBadge({ text }) {
  return (
    <span style={{
      background: '#f0fdf4', color: '#16a34a', border: '1px solid #bbf7d0',
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500,
      display: 'inline-block', margin: '2px 4px 2px 0'
    }}>
      {text}
    </span>
  )
}

export default function ConsultantWorkflowsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/consultant-workflows/overview`),
          axios.get(`${API_URL}/api/consultant-workflows/breakdown`),
          axios.get(`${API_URL}/api/consultant-workflows/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Consultant Workflows data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Consultant Workflows...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#94a3b8' }}>Consultant Workflows data not available.</div>

  const s = overview.summary || {}
  const tabs = ['overview', 'roles', 'steps', 'definitions']
  const tabLabels = { overview: 'Overview', roles: 'All Roles', steps: 'Workflow Steps', definitions: 'Definitions' }

  const phaseData = overview.phase_distribution || []
  const signoffData = overview.signoff_distribution || []

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Consultant Workflows Dashboard</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>Human clinical oversight — per-role workflow phases, steps, and sign-off gates</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid #e2e8f0', paddingBottom: 8 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t ? '#1e293b' : 'transparent', color: tab === t ? '#fff' : '#64748b'
          }}>
            {tabLabels[t]}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI Grid */}
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
              <KPI label="Total Roles" value={fmt(s.total_roles)} />
              <KPI label="Total Phases" value={fmt(s.total_phases)} />
              <KPI label="Total Steps" value={fmt(s.total_steps)} />
              <KPI label="Total Sign-offs" value={fmt(s.total_signoffs)} />
              <KPI label="Avg Phases/Role" value={fmt(s.avg_phases_per_role)} />
              <KPI label="Avg Steps/Phase" value={fmt(s.avg_steps_per_phase)} />
            </div>
          </Card>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Phases per Role">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={phaseData} layout="vertical" margin={{ left: 140 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={140} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {phaseData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Sign-offs per Role">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={signoffData} layout="vertical" margin={{ left: 140 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={140} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {signoffData.map((_, i) => (
                      <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Role Summary Table */}
          <Card title="Role Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Role</th>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600, textAlign: 'center' }}>Phases</th>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600, textAlign: 'center' }}>Steps</th>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600, textAlign: 'center' }}>Sign-offs</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.role_summary || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 500 }}>{r.name}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.phases}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.steps}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.signoffs}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── All Roles Tab ── */}
      {tab === 'roles' && (
        <>
          {(breakdown?.roles || []).map((role, ri) => (
            <Card key={ri} title={role.name}>
              <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>{role.summary}</p>

              {/* Phases table */}
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Phase</th>
                    <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600, textAlign: 'center' }}>Steps</th>
                  </tr>
                </thead>
                <tbody>
                  {(role.phases || []).map((phase, pi) => (
                    <tr key={pi} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 6px' }}>{phase.name}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{phase.step_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Sign-off badges */}
              <div style={{ marginTop: 4 }}>
                <span style={{ fontSize: 12, color: '#475569', fontWeight: 600, marginRight: 8 }}>Sign-offs:</span>
                {(role.signoffs || []).map((so, si) => (
                  <SignoffBadge key={si} text={so} />
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ── Workflow Steps Tab ── */}
      {tab === 'steps' && (
        <Card title="All Workflow Steps">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Role</th>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Phase</th>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Step</th>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Input</th>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Task</th>
                  <th style={{ padding: '8px 6px', color: '#475569', fontWeight: 600 }}>Output</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.all_steps || []).map((st, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 500, whiteSpace: 'nowrap' }}>{st.role_name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', whiteSpace: 'nowrap' }}>{st.phase_name}</td>
                    <td style={{ padding: '8px 6px', fontWeight: 500 }}>{st.step}</td>
                    <td style={{ padding: '8px 6px', color: '#64748b' }}>{st.input}</td>
                    <td style={{ padding: '8px 6px', color: '#64748b', maxWidth: 300 }}>{st.task}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{st.output}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <>
          {/* Role descriptions */}
          <Card title="Role Descriptions">
            {(defs?.roles || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 10, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{r.name}</div>
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{r.summary}</div>
              </div>
            ))}
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            {(defs?.glossary || []).map((g, i) => (
              <div key={i} style={{ marginBottom: 10, padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{g.term}</div>
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{g.definition}</div>
              </div>
            ))}
          </Card>

          {/* Clinical Notes */}
          <Card title="Clinical Notes">
            {(defs?.clinical_notes || []).map((note, i) => (
              <div key={i} style={{ marginBottom: 8, padding: '8px 12px', background: '#fffbeb', borderRadius: 6, fontSize: 12, color: '#92400e', lineHeight: 1.5 }}>
                {note}
              </div>
            ))}
          </Card>

          {/* References */}
          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs?.references || []).map((ref, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', lineHeight: 1.6, marginBottom: 4 }}>{ref}</li>
              ))}
            </ol>
          </Card>
        </>
      )}

      {/* Meta */}
      {breakdown?.meta && (
        <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8', textAlign: 'right' }}>
          Updated: {breakdown.meta.updated_at || '--'}
        </div>
      )}
    </div>
  )
}
