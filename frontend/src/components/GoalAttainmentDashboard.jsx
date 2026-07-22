import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const SCORE_COLORS = {
  '-2': '#dc2626',
  '-1': '#f97316',
  '0':  '#3b82f6',
  '+1': '#22c55e',
  '+2': '#16a34a'
}
const SCORE_LABELS = {
  '-2': 'Much Less',
  '-1': 'Less',
  '0':  'Expected',
  '+1': 'More',
  '+2': 'Much More'
}

const DOMAIN_COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']

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

function ScoreBadge({ score }) {
  const key = score > 0 ? `+${score}` : String(score)
  const color = SCORE_COLORS[key] || '#94a3b8'
  const label = SCORE_LABELS[key] || key
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label} ({key})</span>
  )
}

function TrendArrow({ direction }) {
  if (direction === 'improving') return <span style={{ color: '#22c55e', fontWeight: 700 }}>&#9650;</span>
  if (direction === 'declining') return <span style={{ color: '#ef4444', fontWeight: 700 }}>&#9660;</span>
  return <span style={{ color: '#94a3b8' }}>&#9654;</span>
}

export default function GoalAttainmentDashboard() {
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
          axios.get(`${API_URL}/api/goal-attainment/overview`),
          axios.get(`${API_URL}/api/goal-attainment/breakdown`),
          axios.get(`${API_URL}/api/goal-attainment/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Goal-Attainment data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No goal-attainment data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patients' },
    { id: 'domains', label: 'Domains' },
    { id: 'trends', label: 'Trends' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const gasDistData = overview?.gas_distribution || []
  const gasDistColored = gasDistData.map(d => {
    const key = d.score > 0 ? `+${d.score}` : String(d.score)
    return { ...d, color: SCORE_COLORS[key] || '#94a3b8', display_label: d.score_label || key }
  })
  const domainPerfData = overview?.domain_performance || []
  const trendData = overview?.trend || []
  const topAchievers = overview?.top_achievers || []

  /* Breakdown data prep */
  const patientGoals = breakdown?.patient_goals || []
  const domainDrill = breakdown?.domain_drill || []
  const recentReviews = breakdown?.recent_reviews || []
  const atRisk = breakdown?.at_risk || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Goal-Attainment Scaling (GAS) Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Occupational therapy goal tracking, T-score trends, and domain performance
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview.kpi?.total_patients)} />
              <KPI label="Total Goals" value={fmt(overview.kpi?.total_goals)} />
              <KPI label="Avg T-Score" value={fmt(overview.kpi?.avg_gas_t_score)} color="#3b82f6" />
              <KPI label="Goals Met" value={`${fmt(overview.kpi?.pct_goals_met)}%`} color="#22c55e" />
              <KPI label="Exceeding" value={`${fmt(overview.kpi?.pct_exceeding)}%`} color="#16a34a" />
            </div>
          </Card>

          {/* Pie: GAS Score Distribution */}
          <Card title="GAS Score Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={gasDistColored} dataKey="count" nameKey="display_label" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {gasDistColored.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {gasDistColored.map((d, i) => (
                <span key={i} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.display_label}: {d.count}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Domain Performance */}
          <Card title="Domain Performance" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={domainPerfData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[-2, 2]} />
                <Tooltip />
                <Bar dataKey="avg_score" name="Avg Score">
                  {domainPerfData.map((d, i) => (
                    <Cell key={i} fill={d.avg_score >= 0 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: T-Score Trend */}
          <Card title="T-Score Trend (6 months)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis domain={[30, 70]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_t_score" stroke="#3b82f6" name="Avg T-Score" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="goals_met_pct" stroke="#22c55e" name="Goals Met %" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Top Achievers */}
          <Card title="Top Achievers">
            {topAchievers.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data.</div>
            ) : (
              <div>
                {topAchievers.map((p, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{p.name || p.patient_id}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{p.goals_met}/{p.total_goals || p.goals_met} goals met</div>
                    </div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#22c55e' }}>{fmt(p.t_score)}</div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 2: Patients */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Patient KPIs */}
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Patients with Goals" value={fmt(patientGoals.length)} />
              <KPI label="At Risk (T < 40)" value={fmt(atRisk.length)} color="#ef4444" />
              <KPI label="Review Due" value={fmt(overview?.kpi?.review_due_count)} color="#f59e0b" />
            </div>
          </Card>

          {/* Patient Goal Table */}
          <Card title="Patient Goal Attainment" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>T-Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Goals</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Met</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {patientGoals.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, fontWeight: 600, color: row.t_score >= 50 ? '#22c55e' : row.t_score >= 40 ? '#f59e0b' : '#ef4444' }}>{fmt(row.t_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{(row.goals || []).length}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>
                        {(row.goals || []).filter(g => g.current_score >= 0).length}
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.recommendation || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* At-Risk Patients */}
          <Card title="At-Risk Patients (T-Score < 40)" span={2}>
            {atRisk.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No at-risk patients.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>T-Score</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Below Expected Goals</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {atRisk.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, fontWeight: 600, color: '#ef4444' }}>{fmt(row.t_score)}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#ef4444' }}>{(row.below_expected_domains || row.risk_reasons || []).join(', ') || '--'}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.recommendation || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 3: Domains */}
      {tab === 'domains' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Domain Bar Chart */}
          <Card title="Domain Average Score & Met %" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={domainPerfData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" domain={[-2, 2]} label={{ value: 'Avg Score', angle: -90, position: 'insideLeft', style: { fontSize: 10 } }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 100]} label={{ value: '% Met', angle: 90, position: 'insideRight', style: { fontSize: 10 } }} />
                <Tooltip />
                <Bar yAxisId="left" dataKey="avg_score" fill="#3b82f6" name="Avg Score" />
                <Bar yAxisId="right" dataKey="pct_met" fill="#22c55e" name="% Met" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-domain detail cards */}
          {domainDrill.map((dom, di) => (
            <Card key={di} title={dom.domain}>
              <p style={{ margin: '0 0 8px', fontSize: 12, color: '#64748b' }}>{dom.description}</p>
              <div style={{ fontSize: 12, marginBottom: 8 }}>
                <strong>Scoring:</strong> <span style={{ color: '#475569' }}>{dom.scoring_criteria || '--'}</span>
              </div>
              {(dom.patients || []).length > 0 && (
                <div style={{ maxHeight: 200, overflow: 'auto' }}>
                  <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <th style={{ textAlign: 'left', padding: '4px 6px', color: '#64748b' }}>Patient</th>
                        <th style={{ textAlign: 'center', padding: '4px 6px', color: '#64748b' }}>Score</th>
                        <th style={{ textAlign: 'center', padding: '4px 6px', color: '#64748b' }}>Trend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dom.patients.map((p, pi) => (
                        <tr key={pi} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '4px 6px' }}>{p.name || p.patient_id}</td>
                          <td style={{ padding: '4px 6px', textAlign: 'center' }}><ScoreBadge score={p.score} /></td>
                          <td style={{ padding: '4px 6px', textAlign: 'center' }}><TrendArrow direction={p.trend_direction || p.trend} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* Tab 4: Trends */}
      {tab === 'trends' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* T-Score Trend Line */}
          <Card title="T-Score Trend" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis domain={[30, 70]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_t_score" stroke="#3b82f6" name="Avg T-Score" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Goals Met % Trend */}
          <Card title="Goals Met % Trend">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="goals_met_pct" stroke="#22c55e" name="Goals Met %" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Recent Reviews */}
          <Card title="Recent Goal Reviews">
            {recentReviews.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No recent reviews.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Reviewer</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Changes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentReviews.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.date}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.reviewer}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.changes || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Per-Patient T-Score Bar */}
          <Card title="Patient T-Score Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={patientGoals.map(p => ({ name: p.name || p.patient_id, t_score: p.t_score })).sort((a, b) => (b.t_score || 0) - (a.t_score || 0))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis domain={[0, 80]} />
                <Tooltip />
                <Bar dataKey="t_score" name="T-Score">
                  {patientGoals.sort((a, b) => (b.t_score || 0) - (a.t_score || 0)).map((p, i) => (
                    <Cell key={i} fill={p.t_score >= 50 ? '#22c55e' : p.t_score >= 40 ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* GAS Scale */}
          {defs.gas_scale && defs.gas_scale.length > 0 && (
            <Card title="GAS Scoring Scale">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
                {defs.gas_scale.map((s, i) => {
                  const color = SCORE_COLORS[s.score] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: 12, background: color + '11', border: `1px solid ${color}33`, borderRadius: 8, textAlign: 'center' }}>
                      <div style={{ fontSize: 20, fontWeight: 700, color }}>{s.score}</div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginTop: 4 }}>{s.label}</div>
                      <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.description}</div>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Metrics */}
          {defs.metrics && defs.metrics.length > 0 && (
            <Card title="Metrics">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.metrics.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.name}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.formula}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{d.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Domains */}
          {defs.domains && defs.domains.length > 0 && (
            <Card title="Goal Domains">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.domains.map((dom, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8 }}>
                    <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>{dom.name}</h4>
                    <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{dom.description}</p>
                    {dom.typical_goals && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
                        Typical goals: <strong style={{ color: '#3b82f6' }}>{dom.typical_goals}</strong>
                      </div>
                    )}
                  </div>
                ))}
              </div>
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
