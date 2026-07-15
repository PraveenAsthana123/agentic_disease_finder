import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']
const ACTIVITY_COLORS = {
  medication_reminders: '#3b82f6',
  meals_logged: '#10b981',
  exercise_logged: '#f59e0b',
  sleep_logged: '#8b5cf6',
  mood_logged: '#ec4899',
  seizure_logged: '#ef4444',
}
const COMPLETION_COLORS = {
  '0-25%': '#ef4444',
  '26-50%': '#f59e0b',
  '51-75%': '#3b82f6',
  '76-100%': '#10b981',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DailyPlansDashboard() {
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
      axios.get(`${API_URL}/api/daily-plans/overview`),
      axios.get(`${API_URL}/api/daily-plans/breakdown`),
      axios.get(`${API_URL}/api/daily-plans/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Daily Plans data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Daily Plans Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Patient daily plan adherence analytics — activity completion rates, engagement trends, per-patient tracking, AI suggestions
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const completionData = Object.entries(data.completion_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const activityRateData = Object.entries(data.activity_rates || {}).map(([k, v]) => ({
    name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    rate: v,
    key: k,
  }))
  const radarData = activityRateData.map(d => ({ subject: d.name, rate: d.rate, fullMark: 100 }))
  const trendData = data.daily_trend || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Total Plans">
        <KPI value={data.total_plans} label="daily plans" color="#3b82f6" />
      </Card>
      <Card title="Patients">
        <KPI value={data.total_patients} label="unique patients" color="#8b5cf6" />
      </Card>
      <Card title="Avg Completion">
        <KPI value={`${data.avg_completion_pct}%`} label="average plan completion" color="#10b981" />
      </Card>
      <Card title="Days Tracked">
        <KPI value={data.total_days} label="unique days" color="#f59e0b" />
      </Card>
      <Card title="Range">
        <KPI value={`${data.min_completion_pct}–${data.max_completion_pct}%`} label="min–max completion" color="#64748b" />
      </Card>

      {/* Activity Rates Radar */}
      <Card title="Activity Engagement Rates (%)" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Radar name="Rate" dataKey="rate" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
            <Tooltip formatter={v => `${v}%`} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      {/* Completion Distribution */}
      <Card title="Completion Distribution" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={completionData} cx="50%" cy="50%" outerRadius={100} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
              {completionData.map((entry, i) => (
                <Cell key={i} fill={COMPLETION_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Activity Totals Bar Chart */}
      <Card title="Activity Totals" span={3}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={Object.entries(data.activity_totals || {}).map(([k, v]) => ({
            name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            count: v,
            key: k,
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {Object.keys(data.activity_totals || {}).map((k, i) => (
                <Cell key={i} fill={ACTIVITY_COLORS[k] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Daily Completion Trend */}
      <Card title="Daily Average Completion Trend" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => `${v}%`} />
            <Line type="monotone" dataKey="avg_completion" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg Completion %" />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Date Range Info */}
      <Card title="Date Range" span={5}>
        <div style={{ display: 'flex', gap: 40, justifyContent: 'center', fontSize: 14, color: '#475569' }}>
          <span><strong>Start:</strong> {data.date_range?.start}</span>
          <span><strong>End:</strong> {data.date_range?.end}</span>
          <span><strong>Plans per day:</strong> {data.total_plans && data.total_days ? Math.round(data.total_plans / data.total_days) : '--'}</span>
        </div>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const perPatient = data.per_patient || []
  const recentPlans = data.recent_plans || []
  const lowAdherence = data.low_adherence || []
  const highAdherence = data.high_adherence || []
  const weeklyPattern = data.weekly_pattern || []

  const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569', whiteSpace: 'nowrap' }
  const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12, color: '#334155' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Weekly Pattern */}
      <Card title="Weekly Activity Pattern (Day of Week)">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={weeklyPattern}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day_of_week" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => `${v}%`} />
            <Legend />
            <Bar dataKey="avg_completion" fill="#3b82f6" name="Avg Completion %" radius={[4, 4, 0, 0]} />
            <Bar dataKey="med_rate" fill={ACTIVITY_COLORS.medication_reminders} name="Medication %" radius={[4, 4, 0, 0]} />
            <Bar dataKey="meal_rate" fill={ACTIVITY_COLORS.meals_logged} name="Meals %" radius={[4, 4, 0, 0]} />
            <Bar dataKey="exercise_rate" fill={ACTIVITY_COLORS.exercise_logged} name="Exercise %" radius={[4, 4, 0, 0]} />
            <Bar dataKey="sleep_rate" fill={ACTIVITY_COLORS.sleep_logged} name="Sleep %" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Low & High Adherence side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title={`Low Adherence Patients (${lowAdherence.length})`}>
          {lowAdherence.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 13 }}>No low-adherence patients</p>
          ) : (
            <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Avg %</th>
                  <th style={thStyle}>Plans</th>
                </tr></thead>
                <tbody>
                  {lowAdherence.map((p, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{p.patient_id}</td>
                      <td style={{ ...tdStyle, color: '#ef4444', fontWeight: 600 }}>{p.avg_completion}%</td>
                      <td style={tdStyle}>{p.plan_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card title={`High Adherence Patients (${highAdherence.length})`}>
          {highAdherence.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 13 }}>No high-adherence patients</p>
          ) : (
            <div style={{ background: '#f0fdf4', borderRadius: 8, padding: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Avg %</th>
                  <th style={thStyle}>Plans</th>
                </tr></thead>
                <tbody>
                  {highAdherence.map((p, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{p.patient_id}</td>
                      <td style={{ ...tdStyle, color: '#10b981', fontWeight: 600 }}>{p.avg_completion}%</td>
                      <td style={tdStyle}>{p.plan_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* Per-Patient Summary */}
      <Card title={`Per-Patient Engagement (${perPatient.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Patient</th>
              <th style={thStyle}>Plans</th>
              <th style={thStyle}>Avg %</th>
              <th style={thStyle}>Meds</th>
              <th style={thStyle}>Meals</th>
              <th style={thStyle}>Exercise</th>
              <th style={thStyle}>Sleep</th>
              <th style={thStyle}>Mood</th>
              <th style={thStyle}>Seizure</th>
              <th style={thStyle}>First</th>
              <th style={thStyle}>Last</th>
            </tr></thead>
            <tbody>
              {perPatient.map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={tdStyle}>{p.total_plans}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{
                        width: 50, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${p.avg_completion}%`, height: '100%', borderRadius: 3,
                          background: p.avg_completion >= 70 ? '#10b981' : p.avg_completion >= 40 ? '#f59e0b' : '#ef4444'
                        }} />
                      </div>
                      <span>{p.avg_completion}%</span>
                    </div>
                  </td>
                  <td style={tdStyle}>{p.total_med_reminders}</td>
                  <td style={tdStyle}>{p.total_meals}</td>
                  <td style={tdStyle}>{p.total_exercise}</td>
                  <td style={tdStyle}>{p.total_sleep}</td>
                  <td style={tdStyle}>{p.total_mood}</td>
                  <td style={tdStyle}>{p.total_seizure}</td>
                  <td style={tdStyle}>{p.first_plan}</td>
                  <td style={tdStyle}>{p.last_plan}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Recent Plans */}
      <Card title="Recent Plans (last 30)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Patient</th>
              <th style={thStyle}>Date</th>
              <th style={thStyle}>Meds</th>
              <th style={thStyle}>Meals</th>
              <th style={thStyle}>Exercise</th>
              <th style={thStyle}>Sleep</th>
              <th style={thStyle}>Mood</th>
              <th style={thStyle}>Seizure</th>
              <th style={thStyle}>Completion</th>
              <th style={thStyle}>AI Suggestion</th>
            </tr></thead>
            <tbody>
              {recentPlans.map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={tdStyle}>{p.plan_date}</td>
                  <td style={tdStyle}>{p.medication_reminders_set}</td>
                  <td style={tdStyle}>{p.meals_logged}</td>
                  <td style={tdStyle}>{p.exercise_logged}</td>
                  <td style={tdStyle}>{p.sleep_logged}</td>
                  <td style={tdStyle}>{p.mood_logged}</td>
                  <td style={tdStyle}>{p.seizure_logged}</td>
                  <td style={tdStyle}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                      background: p.plan_completion_pct >= 70 ? '#dcfce7' : p.plan_completion_pct >= 40 ? '#fef3c7' : '#fee2e2',
                      color: p.plan_completion_pct >= 70 ? '#166534' : p.plan_completion_pct >= 40 ? '#92400e' : '#991b1b',
                    }}>{p.plan_completion_pct}%</span>
                  </td>
                  <td style={{ ...tdStyle, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={p.ai_suggestion}>{p.ai_suggestion}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* AI Suggestions count */}
      <Card title="AI Suggestions">
        <p style={{ fontSize: 14, color: '#475569' }}>
          <strong>{data.ai_suggestion_count}</strong> personalized AI suggestions generated across all daily plans.
        </p>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sectionStyle = { marginBottom: 24 }
  const dlStyle = { display: 'grid', gridTemplateColumns: '200px 1fr', gap: '6px 16px', fontSize: 13 }
  const dtStyle = { fontWeight: 600, color: '#334155' }
  const ddStyle = { color: '#64748b', margin: 0 }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Field Definitions">
        <div style={dlStyle}>
          {Object.entries(data.fields || {}).map(([k, v]) => (
            <React.Fragment key={k}>
              <dt style={dtStyle}>{k}</dt>
              <dd style={ddStyle}>{v}</dd>
            </React.Fragment>
          ))}
        </div>
      </Card>

      <Card title="Completion Tiers">
        <div style={dlStyle}>
          {Object.entries(data.completion_tiers || {}).map(([k, v]) => (
            <React.Fragment key={k}>
              <dt style={dtStyle}>{k}</dt>
              <dd style={ddStyle}>{v}</dd>
            </React.Fragment>
          ))}
        </div>
      </Card>

      <Card title="Activity Types">
        <div style={dlStyle}>
          {Object.entries(data.activity_types || {}).map(([k, v]) => (
            <React.Fragment key={k}>
              <dt style={dtStyle}>{k}</dt>
              <dd style={ddStyle}>{v}</dd>
            </React.Fragment>
          ))}
        </div>
      </Card>

      <Card title="Glossary">
        <div style={dlStyle}>
          {Object.entries(data.glossary || {}).map(([k, v]) => (
            <React.Fragment key={k}>
              <dt style={dtStyle}>{k}</dt>
              <dd style={ddStyle}>{v}</dd>
            </React.Fragment>
          ))}
        </div>
      </Card>
    </div>
  )
}
