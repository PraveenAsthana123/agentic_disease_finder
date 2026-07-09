import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Legend
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'stakeholders', label: 'Stakeholders' },
  { id: 'business_case', label: 'Business Case' },
  { id: 'roadmap', label: 'Roadmap' },
  { id: 'definitions', label: 'Definitions' },
]

const STAKEHOLDER_ICONS = {
  'Neurologist': '\u{1F9E0}',
  'EEG Technician': '\u{1F50C}',
  'IT Administrator': '\u{1F4BB}',
  'Hospital Administration': '\u{1F3E5}',
  'Patient': '\u{1F9D1}',
  'Nurse': '\u{1FA7A}',
  'Data Scientist': '\u{1F4CA}',
  'Regulatory Officer': '\u{1F4DC}',
}

export default function NeuroLabReadinessDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/neurolab-readiness/overview`),
      axios.get(`${API_URL}/api/neurolab-readiness/breakdown`),
      axios.get(`${API_URL}/api/neurolab-readiness/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading NeuroLab Readiness Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>NeuroLab Readiness Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Deployment readiness assessment for AI-powered neurology lab platform
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Stakeholders" value={fmt(ov?.kpis?.stakeholders_count)} sub="Identified roles" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Processes" value={ov?.kpis?.processes_built != null && ov?.kpis?.processes_total != null ? `${ov.kpis.processes_built}/${ov.kpis.processes_total}` : '--'} sub="Built / Total" color="#10b981" />
          </Card>
          <Card>
            <KPI label="Functionality" value={ov?.kpis?.functionality_built != null && ov?.kpis?.functionality_total != null ? `${ov.kpis.functionality_built}/${ov.kpis.functionality_total}` : '--'} sub="Built / Total" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Overall Readiness" value={pct(ov?.kpis?.overall_readiness_pct)} sub="Deployment score" color="#f59e0b" />
          </Card>
          <Card>
            <KPI label="Phase Completed" value={fmt(ov?.kpis?.phase_completed)} sub="Current phase" color="#06b6d4" />
          </Card>

          {/* Radar Chart — Readiness across dimensions */}
          <Card title="Readiness Dimensions" span={3}>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={[
                { dimension: 'Stakeholders', value: ov?.radar?.stakeholders || 0 },
                { dimension: 'Processes', value: ov?.radar?.processes || 0 },
                { dimension: 'Functionality', value: ov?.radar?.functionality || 0 },
                { dimension: 'Phases', value: ov?.radar?.phases || 0 },
              ]}>
                <PolarGrid />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar name="Readiness %" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.35} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pie Chart — Process statuses */}
          <Card title="Process Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={(ov?.process_status_distribution || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="status"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.process_status_distribution || []).filter(d => d.value > 0).map((_, i) => {
                    const statusColors = { built: '#10b981', partial: '#f59e0b', missing: '#ef4444' }
                    const item = (ov?.process_status_distribution || [])[i]
                    return <Cell key={i} fill={statusColors[item?.status?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  })}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── STAKEHOLDERS TAB ─── */}
      {tab === 'stakeholders' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(bd?.stakeholder_detail || []).map((sh, i) => (
            <Card key={i}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <span style={{ fontSize: 28 }}>{STAKEHOLDER_ICONS[sh.role] || '\u{1F464}'}</span>
                <h3 style={{ margin: 0, fontSize: 16, color: '#1e293b' }}>{sh.role || 'Unknown Role'}</h3>
              </div>

              {/* Horizontal bar chart: built vs missing */}
              <ResponsiveContainer width="100%" height={60}>
                <BarChart layout="vertical" data={[{
                  role: sh.role,
                  built: (sh.built || []).length,
                  missing: (sh.missing || []).length
                }]}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 10 }} />
                  <YAxis type="category" dataKey="role" hide />
                  <Tooltip />
                  <Bar dataKey="built" name="Built" fill="#10b981" stackId="a" radius={[4, 0, 0, 4]} />
                  <Bar dataKey="missing" name="Missing" fill="#ef4444" stackId="a" radius={[0, 4, 4, 0]} />
                  <Legend />
                </BarChart>
              </ResponsiveContainer>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 12 }}>
                {/* Built items */}
                <div>
                  <h4 style={{ margin: '0 0 6px', fontSize: 13, color: '#10b981' }}>Built</h4>
                  <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#334155', lineHeight: 1.8 }}>
                    {(sh.built || []).map((item, j) => (
                      <li key={j} style={{ color: '#166534' }}>{item}</li>
                    ))}
                    {(sh.built || []).length === 0 && (
                      <li style={{ color: '#94a3b8', listStyle: 'none', marginLeft: -18 }}>None yet</li>
                    )}
                  </ul>
                </div>
                {/* Missing items */}
                <div>
                  <h4 style={{ margin: '0 0 6px', fontSize: 13, color: '#ef4444' }}>Missing</h4>
                  <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#334155', lineHeight: 1.8 }}>
                    {(sh.missing || []).map((item, j) => (
                      <li key={j} style={{ color: '#991b1b' }}>{item}</li>
                    ))}
                    {(sh.missing || []).length === 0 && (
                      <li style={{ color: '#94a3b8', listStyle: 'none', marginLeft: -18 }}>All complete</li>
                    )}
                  </ul>
                </div>
              </div>
            </Card>
          ))}
          {(bd?.stakeholder_detail || []).length === 0 && (
            <Card><p style={{ color: '#94a3b8' }}>No stakeholder data available</p></Card>
          )}
        </div>
      )}

      {/* ─── BUSINESS CASE TAB ─── */}
      {tab === 'business_case' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 24 }}>
          {/* Cost Decrease */}
          <div>
            <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#10b981' }}>Cost Decrease Levers</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {(bd?.business_case?.cost_decrease || []).map((lever, i) => (
                <div key={i} style={{
                  background: '#f0fdf4', borderRadius: 12, padding: 16,
                  borderLeft: '4px solid #10b981'
                }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#166534', marginBottom: 6 }}>{lever.name || 'Unnamed'}</div>
                  <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.5 }}>{lever.impact || 'No impact data'}</div>
                </div>
              ))}
              {(bd?.business_case?.cost_decrease || []).length === 0 && (
                <p style={{ color: '#94a3b8', fontSize: 13 }}>No cost decrease levers defined</p>
              )}
            </div>
          </div>

          {/* Revenue Increase */}
          <div>
            <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#3b82f6' }}>Revenue Increase Levers</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {(bd?.business_case?.revenue_increase || []).map((lever, i) => (
                <div key={i} style={{
                  background: '#eff6ff', borderRadius: 12, padding: 16,
                  borderLeft: '4px solid #3b82f6'
                }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e40af', marginBottom: 6 }}>{lever.name || 'Unnamed'}</div>
                  <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.5 }}>{lever.impact || 'No impact data'}</div>
                </div>
              ))}
              {(bd?.business_case?.revenue_increase || []).length === 0 && (
                <p style={{ color: '#94a3b8', fontSize: 13 }}>No revenue increase levers defined</p>
              )}
            </div>
          </div>

          {/* Productivity Increase */}
          <div>
            <h3 style={{ margin: '0 0 12px', fontSize: 16, color: '#8b5cf6' }}>Productivity Increase Levers</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {(bd?.business_case?.productivity_increase || []).map((lever, i) => (
                <div key={i} style={{
                  background: '#f5f3ff', borderRadius: 12, padding: 16,
                  borderLeft: '4px solid #8b5cf6'
                }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#5b21b6', marginBottom: 6 }}>{lever.name || 'Unnamed'}</div>
                  <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.5 }}>{lever.impact || 'No impact data'}</div>
                </div>
              ))}
              {(bd?.business_case?.productivity_increase || []).length === 0 && (
                <p style={{ color: '#94a3b8', fontSize: 13 }}>No productivity increase levers defined</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ─── ROADMAP TAB ─── */}
      {tab === 'roadmap' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 0 }}>
          {(bd?.implementation_roadmap || []).map((phase, i) => {
            const isBuilt = phase.status === 'built'
            return (
              <div key={i} style={{ display: 'flex', gap: 16, position: 'relative' }}>
                {/* Vertical line + circle */}
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 32 }}>
                  <div style={{
                    width: 20, height: 20, borderRadius: '50%',
                    background: isBuilt ? '#10b981' : '#94a3b8',
                    border: `3px solid ${isBuilt ? '#d1fae5' : '#e2e8f0'}`,
                    zIndex: 1
                  }} />
                  {i < (bd?.implementation_roadmap || []).length - 1 && (
                    <div style={{
                      width: 2, flex: 1, background: '#e2e8f0', minHeight: 40
                    }} />
                  )}
                </div>

                {/* Phase content */}
                <div style={{
                  background: '#fff', borderRadius: 12, padding: 16, marginBottom: 16,
                  boxShadow: '0 1px 3px rgba(0,0,0,.08)', flex: 1,
                  borderLeft: `4px solid ${isBuilt ? '#10b981' : '#94a3b8'}`
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span style={{ fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{phase.phase || `Phase ${i + 1}`}</span>
                    <span style={{
                      padding: '2px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                      background: isBuilt ? '#dcfce7' : '#f1f5f9',
                      color: isBuilt ? '#166534' : '#64748b'
                    }}>{isBuilt ? 'Built' : 'Missing'}</span>
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{phase.scope || 'No scope defined'}</div>
                </div>
              </div>
            )
          })}
          {(bd?.implementation_roadmap || []).length === 0 && (
            <Card><p style={{ color: '#94a3b8' }}>No roadmap data available</p></Card>
          )}
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <Card title="Glossary of Terms" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '10px 14px', color: '#334155', fontWeight: 600 }}>Term</th>
                  <th style={{ padding: '10px 14px', color: '#334155', fontWeight: 600 }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs?.terms || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 14px', fontWeight: 600, color: '#1e293b', whiteSpace: 'nowrap' }}>{d.term}</td>
                    <td style={{ padding: '10px 14px', color: '#475569', lineHeight: 1.6 }}>{d.definition}</td>
                  </tr>
                ))}
                {(defs?.terms || []).length === 0 && (
                  <tr><td colSpan={2} style={{ padding: '16px', color: '#94a3b8', textAlign: 'center' }}>No definitions available</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
