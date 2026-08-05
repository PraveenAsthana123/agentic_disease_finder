import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color }) {
  const colorMap = { blue: '#3b82f6', green: '#10b981', yellow: '#f59e0b', gray: '#94a3b8', red: '#ef4444' }
  return (
    <div style={{ textAlign: 'center', padding: '8px 4px' }}>
      <div style={{ fontSize: 30, fontWeight: 700, color: colorMap[color] || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 3 }}>{label}</div>
    </div>
  )
}

const PHASE_COLORS = { input: '#3b82f6', process: '#8b5cf6', output: '#10b981' }
const STATUS_COLORS = { built: '#10b981', partial: '#f59e0b', planned: '#94a3b8' }
const ROLE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Pipelines' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RoleIPODashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedRole, setSelectedRole] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/r-ipo/overview`),
      axios.get(`${API_URL}/api/r-ipo/breakdown`),
      axios.get(`${API_URL}/api/r-ipo/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
      if (o.data?.roles?.length) setSelectedRole(o.data.roles[0].name)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Role IPO data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const roles = overview?.roles || []
  const kpis = overview?.kpis || []
  const statusDist = overview?.status_distribution || []
  const pipelines = breakdown?.pipelines || {}
  const crossMatrix = breakdown?.cross_matrix || {}
  const phases = definitions?.phases || []
  const qualityGates = definitions?.quality_gates || []
  const statusLegend = definitions?.status_legend || []

  // Steps-per-role bar chart data
  const stepsBarData = roles.map((r, i) => ({
    role: r.name.replace('Clinical ', 'Clin. ').replace('Neurophysiologist', 'Neurophysiol.'),
    steps: r.step_count,
    fill: ROLE_COLORS[i % ROLE_COLORS.length]
  }))

  // Pipeline stages for selected role
  const roleData = selectedRole && pipelines[selectedRole] ? pipelines[selectedRole] : null
  const roleStages = roleData?.stages || []
  const phaseCounts = roleStages.reduce((acc, s) => {
    acc[s.phase] = (acc[s.phase] || 0) + 1
    return acc
  }, {})
  const phaseChartData = Object.entries(phaseCounts).map(([phase, count]) => ({
    name: phase.charAt(0).toUpperCase() + phase.slice(1),
    count,
    fill: PHASE_COLORS[phase] || '#94a3b8'
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Role IPO Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Input-Process-Output pipeline coverage per clinical role — neurologist, technician, pharmacist, and more
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '6px 6px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${kpis.length}, 1fr)`, gap: 8 }}>
              {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
            </div>
          </Card>

          {/* Steps per Role Bar Chart */}
          <Card title="Pipeline Steps per Role" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={stepsBarData} margin={{ top: 4, right: 8, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Steps']} />
                <Bar dataKey="steps" radius={[4, 4, 0, 0]}>
                  {stepsBarData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Status Distribution */}
          <Card title="Build Status">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={statusDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`} labelLine={false}>
                  {statusDist.map((entry, i) => (
                    <Cell key={i} fill={STATUS_COLORS[entry.name] || '#94a3b8'} />
                  ))}
                </Pie>
                <Legend />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Role Cards */}
          {roles.map((role, i) => (
            <Card key={role.name} title={role.name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <span style={{
                  padding: '2px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                  background: STATUS_COLORS[role.status] + '20', color: STATUS_COLORS[role.status]
                }}>{role.status}</span>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{role.step_count} steps</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {(role.sections || []).map((s, j) => (
                  <span key={j} style={{
                    padding: '2px 8px', background: '#f1f5f9', borderRadius: 4, fontSize: 11, color: '#475569'
                  }}>{s}</span>
                ))}
              </div>
              {role.priority && (
                <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>Priority: {role.priority}</div>
              )}
            </Card>
          ))}

          {/* Honest Note */}
          {overview?.honest_note && (
            <Card span={3}>
              <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>
                {overview.honest_note}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Breakdown Tab */}
      {tab === 'breakdown' && (
        <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 16 }}>
          {/* Role Selector */}
          <Card title="Select Role">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {roles.map((role, i) => (
                <button key={role.name} onClick={() => setSelectedRole(role.name)} style={{
                  padding: '8px 12px', border: '1px solid',
                  borderColor: selectedRole === role.name ? '#3b82f6' : '#e2e8f0',
                  borderRadius: 6, background: selectedRole === role.name ? '#eff6ff' : '#fff',
                  color: selectedRole === role.name ? '#1d4ed8' : '#475569',
                  fontSize: 13, cursor: 'pointer', textAlign: 'left', fontWeight: selectedRole === role.name ? 600 : 400
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{role.name}</span>
                    <span style={{ fontSize: 11, color: '#94a3b8' }}>{role.step_count}s</span>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {/* Pipeline Detail */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {roleData ? (
              <>
                {/* Phase distribution for this role */}
                <Card title={`${selectedRole} — Phase Distribution`}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                    <ResponsiveContainer width="100%" height={160}>
                      <PieChart>
                        <Pie data={phaseChartData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={60} label={({ name, count }) => `${name}: ${count}`} labelLine={false}>
                          {phaseChartData.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 8 }}>
                      {phaseChartData.map((p, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: 12, height: 12, borderRadius: 2, background: p.fill, flexShrink: 0 }} />
                          <span style={{ fontSize: 13 }}>{p.name}: <strong>{p.count}</strong> steps</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>

                {/* Pipeline Stages */}
                <Card title={`${selectedRole} — Pipeline Stages (${roleStages.length} steps)`}>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
                    {roleStages.map((stage, i) => (
                      <React.Fragment key={i}>
                        <div style={{
                          padding: '6px 14px', borderRadius: 20,
                          background: (PHASE_COLORS[stage.phase] || '#94a3b8') + '20',
                          border: `1px solid ${PHASE_COLORS[stage.phase] || '#94a3b8'}40`,
                          fontSize: 12, color: '#1e293b'
                        }}>
                          <span style={{ fontSize: 10, color: PHASE_COLORS[stage.phase] || '#94a3b8', fontWeight: 600, marginRight: 4 }}>
                            {stage.phase?.toUpperCase()}
                          </span>
                          {stage.label}
                        </div>
                        {i < roleStages.length - 1 && (
                          <span style={{ color: '#cbd5e1', fontSize: 18 }}>›</span>
                        )}
                      </React.Fragment>
                    ))}
                  </div>
                </Card>

                {/* Cross Matrix — which step labels overlap across roles */}
                {crossMatrix?.step_labels && (
                  <Card title="Cross-Role Step Coverage Matrix">
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ borderCollapse: 'collapse', fontSize: 11, minWidth: 600 }}>
                        <thead>
                          <tr>
                            <th style={{ padding: '6px 10px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0', whiteSpace: 'nowrap' }}>Step</th>
                            {(crossMatrix.rows || []).map(row => (
                              <th key={row.role} style={{ padding: '6px 8px', textAlign: 'center', color: '#475569', borderBottom: '1px solid #e2e8f0', whiteSpace: 'nowrap', fontSize: 10 }}>
                                {row.role.replace('Clinical ', '').replace('Neurophysiologist', 'Neurophysiol.')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(crossMatrix.step_labels || []).slice(0, 20).map((step, si) => (
                            <tr key={si} style={{ background: si % 2 === 0 ? '#f8fafc' : '#fff' }}>
                              <td style={{ padding: '4px 10px', color: '#334155', whiteSpace: 'nowrap' }}>{step}</td>
                              {(crossMatrix.rows || []).map(row => (
                                <td key={row.role} style={{ padding: '4px 8px', textAlign: 'center' }}>
                                  {(row.steps || []).includes(step)
                                    ? <span style={{ color: '#10b981', fontSize: 14 }}>✓</span>
                                    : <span style={{ color: '#e2e8f0', fontSize: 14 }}>·</span>
                                  }
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {crossMatrix.step_labels?.length > 20 && (
                        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 6 }}>
                          Showing 20 of {crossMatrix.step_labels.length} steps
                        </div>
                      )}
                    </div>
                  </Card>
                )}
              </>
            ) : (
              <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>Select a role to view its pipeline</div></Card>
            )}
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* IPO Phases */}
          <Card title="IPO Phases" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              {phases.map((p, i) => (
                <div key={i} style={{
                  padding: 16, borderRadius: 8, border: `2px solid ${Object.values(PHASE_COLORS)[i] || '#94a3b8'}40`,
                  background: (Object.values(PHASE_COLORS)[i] || '#94a3b8') + '10'
                }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: Object.values(PHASE_COLORS)[i] || '#1e293b', marginBottom: 6 }}>
                    {p.name}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{p.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Quality Gates */}
          {qualityGates.length > 0 && (
            <Card title="Quality Gates" span={2}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {qualityGates.map((g, i) => (
                  <div key={i} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 3 }}>{g.gate}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{g.description || g.desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Status Legend */}
          {statusLegend.length > 0 && (
            <Card title="Status Legend">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {statusLegend.map((s, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <span style={{
                      padding: '2px 10px', borderRadius: 20, fontSize: 11, fontWeight: 600, flexShrink: 0,
                      background: (STATUS_COLORS[s.status] || '#94a3b8') + '20',
                      color: STATUS_COLORS[s.status] || '#94a3b8'
                    }}>{s.status}</span>
                    <span style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{s.description || s.desc || s.meaning}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
