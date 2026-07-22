import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const TRAJECTORY_COLORS = {
  improving: '#22c55e',
  stable: '#3b82f6',
  declining: '#ef4444'
}
const TRAJECTORY_LABELS = {
  improving: 'Improving',
  stable: 'Stable',
  declining: 'Declining'
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function TrajectoryBadge({ trajectory }) {
  const color = TRAJECTORY_COLORS[trajectory] || '#94a3b8'
  const label = TRAJECTORY_LABELS[trajectory] || trajectory || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function RecoveryTrajectoryDashboard() {
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
          axios.get(`${API_URL}/api/recovery-trajectory/overview`),
          axios.get(`${API_URL}/api/recovery-trajectory/breakdown`),
          axios.get(`${API_URL}/api/recovery-trajectory/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Recovery Trajectory data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No recovery trajectory data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patients' },
    { id: 'predictions', label: 'Predictions' },
    { id: 'trends', label: 'Trends' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const trajectoryDistData = overview?.trajectory_distribution || []
  const trajectoryDistColored = trajectoryDistData.map(d => ({
    ...d,
    color: TRAJECTORY_COLORS[d.name] || '#94a3b8'
  }))
  const monthlyTrendData = overview?.monthly_trend || []
  const riskFactorsData = overview?.risk_factors || []

  /* Breakdown data prep */
  const patientTrajectories = breakdown?.patient_trajectories || []
  const domainComparisonData = breakdown?.domain_comparison || []
  const predictionFactorsData = breakdown?.prediction_factors || []
  const decliningPatients = breakdown?.declining_patients || []

  /* Derived KPIs for Patients tab */
  const intensiveRehabCount = patientTrajectories.filter(p => p.intensive_rehab_recommended).length
  const avgSlope = patientTrajectories.length > 0
    ? patientTrajectories.reduce((sum, p) => sum + (p.slope || 0), 0) / patientTrajectories.length
    : null

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Recovery Trajectory Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Patient recovery tracking, trajectory classification, and rehabilitation planning
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Cards */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview.kpis?.total_patients)} />
              <KPI label="Improving" value={fmt(overview.kpis?.improving_count)} color="#22c55e" />
              <KPI label="Declining" value={fmt(overview.kpis?.declining_count)} color="#ef4444" />
              <KPI label="Avg Function Rating" value={fmt(overview.kpis?.avg_function_rating)} color="#3b82f6" />
            </div>
          </Card>

          {/* Pie: Trajectory Distribution */}
          <Card title="Trajectory Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={trajectoryDistColored} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {trajectoryDistColored.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {trajectoryDistColored.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {TRAJECTORY_LABELS[d.name] || d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Line: Monthly Function Trend */}
          <Card title="Monthly Function Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={monthlyTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="avg_function" stroke="#3b82f6" name="Avg Function" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar: Risk Factors */}
          <Card title="Risk Factors" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={riskFactorsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="factor" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Patients */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Patient KPIs */}
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              <KPI label="Intensive Rehab Recommended" value={fmt(intensiveRehabCount)} color="#ef4444" />
              <KPI label="Avg Slope" value={fmt(avgSlope)} color="#3b82f6" />
            </div>
          </Card>

          {/* Patient Trajectory Table */}
          <Card title="Patient Trajectories" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Disease</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Trajectory</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Function Rating</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Slope</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Assessments</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rehab</th>
                  </tr>
                </thead>
                <tbody>
                  {patientTrajectories.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.age || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.disease || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><TrajectoryBadge trajectory={row.trajectory_class} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{fmt(row.latest_function_rating)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: row.slope < 0 ? '#ef4444' : row.slope > 0 ? '#22c55e' : '#475569' }}>{fmt(row.slope)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.num_assessments || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>
                        {row.intensive_rehab_recommended
                          ? <span style={{ color: '#ef4444', fontWeight: 600 }}>Yes</span>
                          : <span style={{ color: '#94a3b8' }}>No</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Stacked Bar: Domain Comparison */}
          <Card title="Domain Comparison by Trajectory" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={domainComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="improving" fill={TRAJECTORY_COLORS.improving} name="Improving" stackId="a" />
                <Bar dataKey="stable" fill={TRAJECTORY_COLORS.stable} name="Stable" stackId="a" />
                <Bar dataKey="declining" fill={TRAJECTORY_COLORS.declining} name="Declining" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 3: Predictions */}
      {tab === 'predictions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Bar: Prediction Factors */}
          <Card title="Prediction Factors (Correlation with Declining)" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(200, predictionFactorsData.length * 30)}>
              <BarChart data={predictionFactorsData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="variable" type="category" tick={{ fontSize: 11 }} width={140} />
                <Tooltip />
                <Bar dataKey="correlation" fill="#ef4444" name="Correlation" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Declining Patients Concern List */}
          <Card title="Declining Patients — Concerns">
            {decliningPatients.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No declining patients.</div>
            ) : (
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rating</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Slope</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decliningPatients.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#ef4444' }}>{fmt(row.latest_function_rating)}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#ef4444' }}>{fmt(row.slope)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Intensive Rehab Recommendation List */}
          <Card title="Intensive Rehab Recommended">
            {patientTrajectories.filter(p => p.intensive_rehab_recommended).length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No intensive rehab recommendations.</div>
            ) : (
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Disease</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Trajectory</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rating</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patientTrajectories.filter(p => p.intensive_rehab_recommended).map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.disease || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><TrajectoryBadge trajectory={row.trajectory_class} /></td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{fmt(row.latest_function_rating)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 4: Trends */}
      {tab === 'trends' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Grouped Bar: Functional Domains by Trajectory Class */}
          <Card title="Functional Domains by Trajectory Class" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={domainComparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="improving" fill={TRAJECTORY_COLORS.improving} name="Improving" />
                <Bar dataKey="stable" fill={TRAJECTORY_COLORS.stable} name="Stable" />
                <Bar dataKey="declining" fill={TRAJECTORY_COLORS.declining} name="Declining" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Slope Distribution */}
          <Card title="Slope Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={patientTrajectories.map(p => ({ name: p.name || p.patient_id, slope: p.slope })).sort((a, b) => a.slope - b.slope)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="slope" name="Slope">
                  {patientTrajectories.sort((a, b) => (a.slope || 0) - (b.slope || 0)).map((p, i) => (
                    <Cell key={i} fill={p.slope < 0 ? TRAJECTORY_COLORS.declining : p.slope > 0 ? TRAJECTORY_COLORS.improving : TRAJECTORY_COLORS.stable} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Improvement Timeline */}
          <Card title="Improvement Timeline">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={monthlyTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="avg_function" stroke="#22c55e" name="Avg Function" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Metrics */}
          {defs.metrics && defs.metrics.length > 0 && (
            <Card title="Metrics">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.metrics.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.metric || d.name}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition || d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Rehab Criteria */}
          {defs.rehab_criteria && defs.rehab_criteria.length > 0 && (
            <Card title="Rehab Criteria">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.rehab_criteria.map((rc, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8 }}>
                    <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>{rc.name || rc.criterion}</h4>
                    <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{rc.description}</p>
                    {rc.threshold && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
                        Threshold: <strong style={{ color: '#3b82f6' }}>{rc.threshold}</strong>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Rating Scales */}
          {defs.rating_scales && defs.rating_scales.length > 0 && (
            <Card title="Rating Scales">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.rating_scales.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.scale || d.name}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.description || d.range}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.glossary.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.term}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
