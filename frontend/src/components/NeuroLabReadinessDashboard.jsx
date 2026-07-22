import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const STATUS_COLORS = { built: '#10b981', partial: '#f59e0b', missing: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'stakeholders', label: 'Stakeholders' },
  { id: 'business', label: 'Business Case' },
  { id: 'roadmap', label: 'Roadmap' },
  { id: 'definitions', label: 'Definitions' },
]

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
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading NeuroLab Readiness data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>NeuroLab Readiness Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Deployment readiness for neuro-lab AI system — stakeholder coverage, process maturity, business case, roadmap
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'stakeholders' && breakdown && <StakeholdersTab data={breakdown} />}
      {tab === 'business' && breakdown && <BusinessCaseTab data={breakdown.business_case} />}
      {tab === 'roadmap' && breakdown && <RoadmapTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

/* -- Status badge helper -------------------------------------------------- */
function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11,
      fontWeight: 600, background: `${color}18`, color, textTransform: 'capitalize',
    }}>{status}</span>
  )
}

/* -- Overview Tab --------------------------------------------------------- */
function OverviewTab({ data }) {
  const kpis = data.kpis || {}

  // Stakeholder readiness for the per-stakeholder radar
  const stakeholderRadar = (data.stakeholder_readiness || []).map(s => ({
    role: s.role,
    readiness: s.readiness_pct,
    fullMark: 100,
  }))

  // Pie chart: built vs missing functionality
  const funcPie = [
    { name: 'Built', value: kpis.functionality_built || 0 },
    { name: 'Missing', value: kpis.functionality_missing || 0 },
  ].filter(d => d.value > 0)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Readiness Score">
        <KPI label="Overall readiness" value={`${kpis.readiness_pct}%`}
          color={kpis.readiness_pct >= 70 ? '#10b981' : kpis.readiness_pct >= 40 ? '#f59e0b' : '#ef4444'} />
      </Card>
      <Card title="Built Items">
        <KPI label="Capabilities built" value={kpis.total_built_items} color="#10b981" />
      </Card>
      <Card title="Missing Items">
        <KPI label="Capabilities missing" value={kpis.total_missing_items} color="#ef4444" />
      </Card>
      <Card title="Stakeholders">
        <KPI label="Roles tracked" value={kpis.total_stakeholders} color="#3b82f6" />
      </Card>
      <Card title="Processes">
        <KPI label={`${kpis.processes_built} built / ${kpis.processes_partial} partial`}
          value={kpis.total_processes} color="#8b5cf6" />
      </Card>

      <Card title="Stakeholder Readiness Radar" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={stakeholderRadar} cx="50%" cy="50%" outerRadius="75%">
            <PolarGrid />
            <PolarAngleAxis dataKey="role" fontSize={11} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} fontSize={10} />
            <Radar name="Readiness %" dataKey="readiness" stroke="#3b82f6"
              fill="#3b82f6" fillOpacity={0.25} />
            <Tooltip formatter={(v) => [`${v}%`, 'Readiness']} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Functionality: Built vs Missing" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={funcPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, value }) => `${name} (${value})`}
              labelLine={false} fontSize={12}>
              <Cell fill="#10b981" />
              <Cell fill="#ef4444" />
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Process Status" span={5}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {(data.process_status || []).map((p, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '6px 14px',
              borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0',
            }}>
              <span style={{
                width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
                background: STATUS_COLORS[p.status] || '#94a3b8',
              }} />
              <span style={{ fontSize: 13, color: '#334155' }}>{p.name}</span>
              <StatusBadge status={p.status} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

/* -- Stakeholders Tab ----------------------------------------------------- */
function StakeholdersTab({ data }) {
  const stakeholders = data.stakeholder_detail || []

  // Bar chart data: built vs missing per role
  const barData = stakeholders.map(s => ({
    role: s.role,
    built: s.built.length,
    missing: s.missing.length,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Built vs Missing by Role" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="role" fontSize={11} angle={-15} textAnchor="end" height={60} />
            <YAxis fontSize={12} />
            <Tooltip />
            <Legend />
            <Bar dataKey="built" fill="#10b981" name="Built" radius={[4, 4, 0, 0]} />
            <Bar dataKey="missing" fill="#ef4444" name="Missing" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {stakeholders.map((s, idx) => (
        <Card key={idx} title={null}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            {s.icon && <span style={{ fontSize: 24 }}>{s.icon}</span>}
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: '#1e293b' }}>{s.role}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{s.readiness_pct}% ready</div>
            </div>
          </div>

          {/* Readiness bar */}
          <div style={{ background: '#f1f5f9', borderRadius: 6, height: 8, marginBottom: 14 }}>
            <div style={{
              background: s.readiness_pct >= 70 ? '#10b981' : s.readiness_pct >= 40 ? '#f59e0b' : '#ef4444',
              height: '100%', borderRadius: 6, width: `${Math.min(s.readiness_pct, 100)}%`,
              transition: 'width 0.4s ease',
            }} />
          </div>

          {/* Built items */}
          {s.built.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 6,
                textTransform: 'uppercase' }}>Built</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {s.built.map((item, i) => (
                  <span key={i} style={{
                    display: 'inline-block', padding: '3px 8px', borderRadius: 6, fontSize: 11,
                    background: '#dcfce7', color: '#166534', fontWeight: 500,
                  }}>{item}</span>
                ))}
              </div>
            </div>
          )}

          {/* Missing items */}
          {s.missing.length > 0 && (
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 6,
                textTransform: 'uppercase' }}>Missing</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {s.missing.map((item, i) => (
                  <span key={i} style={{
                    display: 'inline-block', padding: '3px 8px', borderRadius: 6, fontSize: 11,
                    background: '#fef2f2', color: '#991b1b', fontWeight: 500,
                  }}>{item}</span>
                ))}
              </div>
            </div>
          )}
        </Card>
      ))}
    </div>
  )
}

/* -- Business Case Tab ---------------------------------------------------- */
function BusinessCaseTab({ data }) {
  const sections = [
    { key: 'cost_decrease', title: 'Cost Decrease', color: '#10b981', bg: '#f0fdf4' },
    { key: 'revenue_increase', title: 'Revenue Increase', color: '#3b82f6', bg: '#eff6ff' },
    { key: 'productivity_increase', title: 'Productivity Increase', color: '#8b5cf6', bg: '#f5f3ff' },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((sec) => {
        const items = data[sec.key] || []
        return (
          <Card key={sec.key} title={null}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
              <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 32, height: 32, borderRadius: 8, background: `${sec.color}15`,
                color: sec.color, fontSize: 18, fontWeight: 700,
              }}>{sec.key === 'cost_decrease' ? '\u2193' : sec.key === 'revenue_increase' ? '\u2191' : '\u26A1'}</span>
              <h3 style={{ margin: 0, fontSize: 16, color: '#1e293b' }}>{sec.title}</h3>
              <span style={{ fontSize: 12, color: '#64748b', marginLeft: 'auto' }}>
                {items.length} lever{items.length !== 1 ? 's' : ''}
              </span>
            </div>
            {items.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left',
                      borderBottom: '2px solid #e2e8f0', width: '50%' }}>Lever</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left',
                      borderBottom: '2px solid #e2e8f0' }}>Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, i) => (
                    <tr key={i}>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                        fontWeight: 600, color: '#334155' }}>
                        {item.lever || item.name || '--'}
                      </td>
                      <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                        color: '#475569' }}>
                        {item.impact || item.description || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No levers defined</p>
            )}
          </Card>
        )
      })}
    </div>
  )
}

/* -- Roadmap Tab ---------------------------------------------------------- */
function RoadmapTab({ data }) {
  const phases = data.implementation_roadmap || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Implementation Phases">
        <div style={{ position: 'relative', paddingLeft: 28 }}>
          {phases.map((ph, i) => {
            const color = STATUS_COLORS[ph.status] || '#94a3b8'
            const isLast = i === phases.length - 1
            return (
              <div key={i} style={{ position: 'relative', paddingBottom: isLast ? 0 : 28,
                minHeight: 48 }}>
                {/* Vertical line */}
                {!isLast && (
                  <div style={{
                    position: 'absolute', left: -20, top: 16, bottom: 0, width: 2,
                    background: '#e2e8f0',
                  }} />
                )}
                {/* Circle node */}
                <div style={{
                  position: 'absolute', left: -26, top: 4, width: 14, height: 14,
                  borderRadius: '50%', background: color, border: '2px solid #fff',
                  boxShadow: `0 0 0 2px ${color}40`,
                }} />
                {/* Content */}
                <div style={{
                  padding: '8px 16px', borderRadius: 8,
                  background: ph.is_current ? `${color}10` : '#f8fafc',
                  border: ph.is_current ? `2px solid ${color}` : '1px solid #e2e8f0',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>
                      {ph.phase}
                    </span>
                    <StatusBadge status={ph.status} />
                    {ph.is_current && (
                      <span style={{
                        fontSize: 10, fontWeight: 700, color: '#3b82f6', background: '#dbeafe',
                        padding: '1px 8px', borderRadius: 8, textTransform: 'uppercase',
                      }}>Current</span>
                    )}
                  </div>
                  {ph.scope && (
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{ph.scope}</div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </Card>

      {/* Gap analysis cards if present */}
      {(data.gap_analysis || []).length > 0 && (
        <Card title="Gap Analysis by Stakeholder">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {data.gap_analysis.map((ga, i) => (
              <div key={i} style={{
                padding: 14, borderRadius: 8, background: '#fef2f2',
                border: '1px solid #fecaca',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                  {ga.icon && <span style={{ fontSize: 18 }}>{ga.icon}</span>}
                  <span style={{ fontWeight: 700, fontSize: 14, color: '#991b1b' }}>{ga.role}</span>
                  <span style={{ fontSize: 11, color: '#64748b', marginLeft: 'auto' }}>
                    {ga.total_gaps} gap{ga.total_gaps !== 1 ? 's' : ''}
                  </span>
                </div>
                {Object.entries(ga.categories || {}).map(([cat, items], j) => (
                  <div key={j} style={{ marginBottom: 6 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: '#92400e',
                      textTransform: 'capitalize', marginBottom: 2 }}>
                      {cat.replace(/_/g, ' ')}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                      {items.map((item, k) => (
                        <span key={k} style={{
                          fontSize: 10, padding: '2px 6px', borderRadius: 4,
                          background: '#fee2e2', color: '#991b1b',
                        }}>{item}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

/* -- Definitions Tab ------------------------------------------------------ */
function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Status Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0', width: 120 }}>Status</th>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0' }}>Meaning</th>
            </tr>
          </thead>
          <tbody>
            {[
              { status: 'built', meaning: 'Fully implemented and verified -- ready for production use.' },
              { status: 'partial', meaning: 'Some steps automated or scaffolded, but not yet complete end-to-end.' },
              { status: 'missing', meaning: 'Not yet implemented -- required for full deployment readiness.' },
            ].map((row, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0' }}>
                  <StatusBadge status={row.status} />
                </td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                  color: '#475569' }}>
                  {row.meaning}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Key Terms">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0', width: 200 }}>Term</th>
              <th style={{ padding: '8px 10px', textAlign: 'left',
                borderBottom: '2px solid #e2e8f0' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(data.terms || []).map((t, i) => (
              <tr key={i}>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                  fontWeight: 600, color: '#1e293b' }}>
                  {t.term}
                </td>
                <td style={{ padding: '8px 10px', borderBottom: '1px solid #e2e8f0',
                  color: '#475569', lineHeight: 1.5 }}>
                  {t.definition}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
