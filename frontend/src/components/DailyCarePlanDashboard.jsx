import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const COMPLETION_COLORS = {
  high: '#22c55e',
  medium: '#3b82f6',
  low: '#eab308',
  veryLow: '#ef4444'
}

const ACTIVITY_COLORS = {
  medication: '#8b5cf6',
  meals: '#f59e0b',
  exercise: '#22c55e',
  sleep: '#3b82f6',
  mood: '#ec4899',
  seizure: '#ef4444'
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

function completionColor(pct) {
  if (pct >= 76) return COMPLETION_COLORS.high
  if (pct >= 51) return COMPLETION_COLORS.medium
  if (pct >= 26) return COMPLETION_COLORS.low
  return COMPLETION_COLORS.veryLow
}

export default function DailyCarePlanDashboard() {
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
          axios.get(`${API_URL}/api/daily-care-plan/overview`),
          axios.get(`${API_URL}/api/daily-care-plan/breakdown`),
          axios.get(`${API_URL}/api/daily-care-plan/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Daily Care Plan data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Summary' },
    { id: 'engagement', label: 'Engagement' },
    { id: 'recent', label: 'Recent Plans' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const completionDistData = overview?.completion_distribution
    ? Object.entries(overview.completion_distribution).map(([k, v]) => {
        const colorMap = { '0-25%': COMPLETION_COLORS.veryLow, '26-50%': COMPLETION_COLORS.low, '51-75%': COMPLETION_COLORS.medium, '76-100%': COMPLETION_COLORS.high }
        return { name: k, count: v, color: colorMap[k] || '#94a3b8' }
      })
    : []

  const activityTrendData = overview?.activity_trend || []

  const loggingRatesData = overview?.logging_rates
    ? Object.entries(overview.logging_rates).map(([k, v]) => ({
        name: k.charAt(0).toUpperCase() + k.slice(1),
        rate: typeof v === 'number' ? v : 0,
        color: ACTIVITY_COLORS[k] || '#94a3b8'
      }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Daily Care Plan Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Daily care plan tracking — medication, meals, exercise, sleep, mood, and seizure logging
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Plans" value={fmt(overview.total_plans)} />
              <KPI label="Total Patients" value={fmt(overview.total_patients)} />
              <KPI label="Avg Completion %" value={fmt(overview.avg_completion_pct)} color={completionColor(overview.avg_completion_pct || 0)} />
              <KPI label="Plans with Seizures" value={fmt(overview.plans_with_seizures)} color="#ef4444" />
              <KPI label="Medication Adherence Rate" value={fmt(overview.medication_adherence_rate)} color="#8b5cf6" />
              <KPI label="Active Days" value={fmt(overview.active_days)} />
            </div>
          </Card>

          {/* Completion Distribution Bar Chart */}
          <Card title="Completion Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={completionDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Plans">
                  {completionDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Activity Trend Line Chart */}
          <Card title="Activity Trend">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={activityTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_completion" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg Completion %" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Logging Rates Bar Chart */}
          <Card title="Logging Rates by Activity" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={loggingRatesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(v) => `${fmt(v)}%`} />
                <Bar dataKey="rate" name="Rate %">
                  {loggingRatesData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginTop: 8 }}>
              {loggingRatesData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}
                </span>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Tab 2: Patient Summary */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Patient Summary">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total Plans</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Completion %</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Days with Seizure</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Med Adherence %</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Top AI Suggestion</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || row.patient_name || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12 }}>{fmt(row.total_plans)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 600, color: completionColor(row.avg_completion_pct || 0) }}>
                          {fmt(row.avg_completion_pct)}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12, color: row.days_with_seizure > 0 ? '#ef4444' : '#64748b' }}>
                        {fmt(row.days_with_seizure)}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 600, color: (row.med_adherence_pct || 0) >= 80 ? '#22c55e' : '#eab308' }}>
                          {fmt(row.med_adherence_pct)}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {row.top_ai_suggestion || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Engagement */}
      {tab === 'engagement' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Weekly Heatmap Bar Chart */}
          <Card title="Weekly Completion Heatmap" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.weekly_heatmap || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip formatter={(v) => `${fmt(v)}%`} />
                <Bar dataKey="avg_completion" name="Avg Completion %">
                  {(breakdown.weekly_heatmap || []).map((d, i) => (
                    <Cell key={i} fill={completionColor(d.avg_completion || 0)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Top AI Suggestions */}
          <Card title="Top AI Suggestions" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              {(breakdown.ai_suggestions || []).length === 0 && (
                <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No AI suggestions available</div>
              )}
              {(breakdown.ai_suggestions || []).map((s, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 14px', borderBottom: '1px solid #f1f5f9'
                }}>
                  <span style={{ fontSize: 13, color: '#334155', flex: 1 }}>{s.suggestion || s.text || '--'}</span>
                  <span style={{
                    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
                    background: '#3b82f622', color: '#3b82f6', marginLeft: 12, whiteSpace: 'nowrap'
                  }}>{fmt(s.count)}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Tab 4: Recent Plans */}
      {tab === 'recent' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Recent Daily Plans (Last 20)">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Med Reminders</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Meals</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Exercise</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Sleep</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Mood</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b', fontSize: 10 }}>Seizure</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Completion %</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>AI Suggestion</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_plans || []).slice(0, 20).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{row.date || '--'}</td>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || row.patient || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.medication, fontWeight: 600 }}>{row.med_reminders != null ? (row.med_reminders ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.meals, fontWeight: 600 }}>{row.meals != null ? (row.meals ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.exercise, fontWeight: 600 }}>{row.exercise != null ? (row.exercise ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.sleep, fontWeight: 600 }}>{row.sleep != null ? (row.sleep ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.mood, fontWeight: 600 }}>{row.mood != null ? (row.mood ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ color: ACTIVITY_COLORS.seizure, fontWeight: 600 }}>{row.seizure != null ? (row.seizure ? 'Yes' : 'No') : '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 700, color: completionColor(row.completion_pct || 0) }}>
                          {fmt(row.completion_pct)}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {row.ai_suggestion || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Metrics Definitions */}
          {defs.metrics && defs.metrics.length > 0 && (
            <Card title="Metrics Definitions">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.metrics.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.name || d.metric}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.description || d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Activity Type Descriptions */}
          {defs.activity_types && defs.activity_types.length > 0 && (
            <Card title="Activity Type Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.activity_types.map((a, i) => {
                  const color = ACTIVITY_COLORS[a.type] || ACTIVITY_COLORS[a.name?.toLowerCase()] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 5, background: color }} />
                        <h4 style={{ margin: 0, fontSize: 13, color: '#334155' }}>{a.name || a.type}</h4>
                      </div>
                      <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{a.description}</p>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Completion Scoring Methodology */}
          {defs.completion_scoring && (
            <Card title="Completion Scoring Methodology">
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                <p style={{ margin: '0 0 12px', fontSize: 13, color: '#475569' }}>{defs.completion_scoring.description}</p>
                {defs.completion_scoring.levels && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                    {defs.completion_scoring.levels.map((lvl, i) => {
                      const colorMap = { '76-100%': COMPLETION_COLORS.high, '51-75%': COMPLETION_COLORS.medium, '26-50%': COMPLETION_COLORS.low, '0-25%': COMPLETION_COLORS.veryLow }
                      const c = colorMap[lvl.range] || '#94a3b8'
                      return (
                        <div key={i} style={{ textAlign: 'center', padding: '8px 12px', background: c + '15', borderRadius: 8, border: `1px solid ${c}33` }}>
                          <div style={{ fontSize: 14, fontWeight: 700, color: c }}>{lvl.range}</div>
                          <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{lvl.label}</div>
                        </div>
                      )
                    })}
                  </div>
                )}
                {defs.completion_scoring.formula && (
                  <div style={{ marginTop: 12, padding: '8px 12px', background: '#fff', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 12, color: '#334155', fontFamily: 'monospace' }}>
                    {defs.completion_scoring.formula}
                  </div>
                )}
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

          {/* References */}
          {defs.references && (
            <Card>
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>References</h4>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                  {defs.references.map((ref, i) => <li key={i} style={{ marginBottom: 4 }}>{ref}</li>)}
                </ul>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
