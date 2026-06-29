import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#795548']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

export default function ExecutiveScorecardDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/executive-scorecard/overview`),
          axios.get(`${API_URL}/executive-scorecard/breakdown`),
          axios.get(`${API_URL}/executive-scorecard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load executive scorecard data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading executive scorecard...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Executive scorecard data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const deptCensus = overview.department_census || []
  const instrumentUsage = overview.instrument_usage || []
  const dailyOps = overview.daily_operations || []
  const topComponents = overview.top_components || []
  const severityDist = overview.severity_distribution || {}
  const deptDetail = breakdown?.department_detail || []
  const instrumentStats = breakdown?.instrument_stats || []
  const monthlySeizures = breakdown?.monthly_seizures || []
  const defsList = defs?.definitions || []

  const severityData = Object.entries(severityDist).map(([name, value]) => ({ name, value }))

  const kpiCards = [
    { label: 'Total Patients', value: s.total_patients, color: '#1e88e5', icon: '👥' },
    { label: 'Assessments', value: s.total_assessments, color: '#7c4dff', icon: '📋' },
    { label: 'Seizure Events', value: s.total_seizure_events, color: '#f44336', icon: '⚡' },
    { label: 'AI Operations', value: s.total_ai_operations, color: '#4caf50', icon: '🤖' },
    { label: 'Instruments', value: s.instruments_used, color: '#ff9800', icon: '🔬' },
    { label: 'Expert Reviews', value: s.total_expert_reviews, color: '#00bcd4', icon: '👨‍⚕️' },
    { label: 'HITL Reviews', value: s.total_hitl_reviews, color: '#e91e63', icon: '🔍' },
    { label: 'Alert Rate', value: `${s.alert_rate_pct}%`, color: s.alert_rate_pct > 20 ? '#f44336' : '#4caf50', icon: '🚨' },
  ]

  const cardStyle = (color) => ({
    background: '#fff',
    border: `2px solid ${color}22`,
    borderRadius: 12,
    padding: '16px 20px',
    textAlign: 'center',
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  })

  const sectionStyle = {
    background: '#fff',
    border: '1px solid #e2e8f0',
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
  }

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: 0, color: '#0f172a' }}>Executive Scorecard</h2>
          <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 14 }}>
            Aggregated clinical + AI KPIs from real clinical.db data
          </p>
        </div>
        <button
          onClick={() => setShowDefs(!showDefs)}
          style={{
            background: showDefs ? '#1e88e5' : '#f1f5f9',
            color: showDefs ? '#fff' : '#475569',
            border: 'none', borderRadius: 8, padding: '8px 16px', cursor: 'pointer', fontSize: 13,
          }}
        >
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {showDefs && (
        <div style={{ ...sectionStyle, background: '#f0f9ff', borderColor: '#bae6fd', marginBottom: 20 }}>
          <h3 style={{ marginTop: 0, color: '#0c4a6e' }}>Metric Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
            {defsList.map((d, i) => (
              <div key={i} style={{ background: '#fff', borderRadius: 8, padding: 12, border: '1px solid #e0f2fe' }}>
                <strong style={{ color: '#0f172a' }}>{d.metric}</strong>
                <p style={{ margin: '4px 0 2px', fontSize: 13, color: '#475569' }}>{d.description}</p>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Source: {d.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12, marginBottom: 24 }}>
        {kpiCards.map((k, i) => (
          <div key={i} style={cardStyle(k.color)}>
            <div style={{ fontSize: 22, marginBottom: 4 }}>{k.icon}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: k.color }}>{typeof k.value === 'number' ? fmt(k.value) : k.value}</div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{k.label}</div>
          </div>
        ))}
      </div>

      {/* Row: Department Census + Severity Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div style={sectionStyle}>
          <h3 style={{ marginTop: 0, color: '#0f172a' }}>Department Census</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={deptCensus} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="department" tick={{ fontSize: 12 }} width={80} />
              <Tooltip />
              <Bar dataKey="count" fill="#1e88e5" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={sectionStyle}>
          <h3 style={{ marginTop: 0, color: '#0f172a' }}>Seizure Severity</h3>
          {severityData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={severityData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                  {severityData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', textAlign: 'center' }}>No seizure data recorded</p>
          )}
        </div>
      </div>

      {/* AI Operations Trend */}
      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0, color: '#0f172a' }}>AI Operations — 7-Day Trend</h3>
        {dailyOps.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dailyOps}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="operations" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: '#94a3b8', textAlign: 'center' }}>No recent operations data</p>
        )}
      </div>

      {/* Row: Top Components + Instrument Usage */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <div style={sectionStyle}>
          <h3 style={{ marginTop: 0, color: '#0f172a' }}>Top AI Components</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={topComponents}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="component" tick={{ fontSize: 10, angle: -30 }} height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="operations" fill="#7c4dff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={sectionStyle}>
          <h3 style={{ marginTop: 0, color: '#0f172a' }}>Assessment Instruments</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={instrumentUsage}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="instrument" tick={{ fontSize: 10, angle: -30 }} height={60} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#ff9800" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Department Detail Table */}
      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0, color: '#0f172a' }}>Department Detail</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Department</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Patients</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Assessments</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg/Patient</th>
            </tr>
          </thead>
          <tbody>
            {deptDetail.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 500 }}>{d.dept}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.patients)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.assessments)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>
                  {d.patients > 0 ? (d.assessments / d.patients).toFixed(1) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Instrument Statistics */}
      <div style={sectionStyle}>
        <h3 style={{ marginTop: 0, color: '#0f172a' }}>Instrument Statistics</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Instrument</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Total</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Score</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Max</th>
              <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Alerts</th>
            </tr>
          </thead>
          <tbody>
            {instrumentStats.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 500 }}>{s.instrument}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(s.total)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(s.avg_score, 1)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(s.avg_max, 1)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'right', color: s.alerts > 0 ? '#f44336' : '#64748b' }}>
                  {fmt(s.alerts)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Monthly Seizures */}
      {monthlySeizures.length > 0 && (
        <div style={sectionStyle}>
          <h3 style={{ marginTop: 0, color: '#0f172a' }}>Monthly Seizure Activity</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={monthlySeizures}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="events" name="Total Events" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              <Bar dataKey="severe" name="Severe" fill="#f44336" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 11, marginTop: 16 }}>
        Avg assessments/patient: {s.avg_assessments_per_patient} · Medications tracked: {fmt(s.total_medications)} · Data from clinical.db
      </div>
    </div>
  )
}
