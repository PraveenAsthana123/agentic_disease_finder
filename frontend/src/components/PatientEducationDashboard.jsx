import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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

function StatusBadge({ status }) {
  const colorMap = { high: '#22c55e', medium: '#eab308', low: '#ef4444', complete: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8', pass: '#22c55e', fail: '#ef4444', 'at-risk': '#ef4444' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function PatientEducationDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sortCol, setSortCol] = useState(null)
  const [sortAsc, setSortAsc] = useState(true)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/patient-education/overview`),
          axios.get(`${API_URL}/patient-education/breakdown`),
          axios.get(`${API_URL}/patient-education/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Patient Education data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'topics', label: 'Topic Analysis' },
    { id: 'progress', label: 'Student Progress' },
    { id: 'quiz', label: 'Quiz Performance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const byTopic = overview?.by_topic || []
  const byFormat = overview?.by_format || []
  const completionDist = overview?.completion_distribution || []
  const monthlyTrend = overview?.monthly_trend || []

  /* --- Breakdown data --- */
  const perPatient = breakdown?.per_patient || []
  const perTopicFormat = breakdown?.per_topic_format || []
  const quizPerformance = breakdown?.quiz_performance || []
  const engagementByFormat = breakdown?.engagement_by_format || []
  const atRiskPatients = breakdown?.at_risk_patients || []

  /* --- Definitions data --- */
  const terms = definitions?.terms || []
  const moduleDescriptions = definitions?.module_descriptions || []

  /* --- Sorting helper for per_patient table --- */
  const handleSort = (col) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(true)
    }
  }

  const sortedPatients = [...perPatient].sort((a, b) => {
    if (!sortCol) return 0
    const av = a[sortCol], bv = b[sortCol]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    if (typeof av === 'number') return sortAsc ? av - bv : bv - av
    return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
  })

  const sortIcon = (col) => sortCol === col ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''

  const thStyle = { textAlign: 'center', padding: '6px 8px', color: '#64748b', cursor: 'pointer', userSelect: 'none' }

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Patient Education Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Tracking educational module completion, quiz performance, and patient engagement across topics and formats
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Modules" value={fmt(overview.total_modules)} />
              <KPI label="Unique Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Avg Completion" value={fmtPct(overview.avg_completion_pct)} color={
                overview.avg_completion_pct >= 0.8 ? '#22c55e' : overview.avg_completion_pct >= 0.5 ? '#eab308' : '#ef4444'
              } />
              <KPI label="Avg Quiz Score" value={fmtPct(overview.avg_quiz_score)} color={
                overview.avg_quiz_score >= 0.8 ? '#22c55e' : overview.avg_quiz_score >= 0.6 ? '#eab308' : '#ef4444'
              } />
              <KPI label="Completion Rate" value={
                <StatusBadge status={
                  overview.completion_rate >= 0.8 ? 'high' : overview.completion_rate >= 0.5 ? 'medium' : 'low'
                } />
              } sub={fmtPct(overview.completion_rate)} />
            </div>
          </Card>

          {/* Topic Performance Bar Chart */}
          <Card title="Avg Completion by Topic" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={byTopic}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="topic" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => fmtPct(v)} />
                <Bar dataKey="avg_completion" fill={COLORS[0]} name="Avg Completion" radius={[4, 4, 0, 0]}>
                  {byTopic.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Format Distribution Pie Chart */}
          <Card title="Format Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={byFormat} dataKey="count" nameKey="format" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {byFormat.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Completion Distribution Bar Chart */}
          <Card title="Completion Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={completionDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill={COLORS[3]} name="Patients" radius={[4, 4, 0, 0]}>
                  {completionDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly Trend Line Chart */}
          <Card title="Monthly Trend" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={monthlyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="new_starts" stroke={COLORS[0]} strokeWidth={2} dot={{ r: 3 }} name="New Starts" />
                <Line type="monotone" dataKey="completions" stroke={COLORS[1]} strokeWidth={2} dot={{ r: 3 }} name="Completions" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== TOPIC ANALYSIS TAB ======================== */}
      {tab === 'topics' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* By Topic Table */}
          <Card title="Topic Summary" span={2}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Topic</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Modules</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Patients</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Completion</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Quiz Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Time (min)</th>
                  </tr>
                </thead>
                <tbody>
                  {byTopic.map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{t.topic}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(t.modules)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(t.patients)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(t.avg_completion)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(t.avg_quiz_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(t.avg_time_minutes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Per-Topic Format Breakdown Table */}
          <Card title="Format Breakdown by Topic">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Topic</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Format</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Count</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Completion</th>
                  </tr>
                </thead>
                <tbody>
                  {perTopicFormat.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.topic}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.format}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.count)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(r.avg_completion)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Quiz Performance Bar Chart (avg/min/max by topic) */}
          <Card title="Quiz Score Range by Topic">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={quizPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="topic" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => fmtPct(v)} />
                <Legend />
                <Bar dataKey="avg_score" fill={COLORS[0]} name="Avg Score" radius={[4, 4, 0, 0]} />
                <Bar dataKey="min_score" fill={COLORS[4]} name="Min Score" radius={[4, 4, 0, 0]} />
                <Bar dataKey="max_score" fill={COLORS[1]} name="Max Score" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== STUDENT PROGRESS TAB ======================== */}
      {tab === 'progress' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Patient Table (sortable) */}
          <Card title="Patient Progress" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ ...thStyle, textAlign: 'left' }} onClick={() => handleSort('patient_id')}>Patient{sortIcon('patient_id')}</th>
                    <th style={thStyle} onClick={() => handleSort('modules_assigned')}>Assigned{sortIcon('modules_assigned')}</th>
                    <th style={thStyle} onClick={() => handleSort('modules_completed')}>Completed{sortIcon('modules_completed')}</th>
                    <th style={thStyle} onClick={() => handleSort('completion_pct')}>Completion %{sortIcon('completion_pct')}</th>
                    <th style={thStyle} onClick={() => handleSort('avg_quiz_score')}>Avg Quiz{sortIcon('avg_quiz_score')}</th>
                    <th style={thStyle} onClick={() => handleSort('total_time_minutes')}>Time (min){sortIcon('total_time_minutes')}</th>
                    <th style={thStyle} onClick={() => handleSort('last_activity')}>Last Activity{sortIcon('last_activity')}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPatients.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.modules_assigned)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.modules_completed)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(p.completion_pct)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(p.avg_quiz_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.total_time_minutes)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{p.last_activity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* At-Risk Patients */}
          <Card title="At-Risk Patients">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              {atRiskPatients.length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No at-risk patients identified</div>
              ) : (
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Completion</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Quiz Score</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Risk Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {atRiskPatients.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(p.completion_pct)}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(p.avg_quiz_score)}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                          <StatusBadge status={'at-risk'} />
                          <span style={{ marginLeft: 6, fontSize: 11, color: '#64748b' }}>{p.risk_reason}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </Card>

          {/* Engagement by Format Horizontal Bar Chart */}
          <Card title="Engagement by Format">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={engagementByFormat} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="format" width={70} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_time_minutes" name="Avg Time (min)" radius={[0, 4, 4, 0]}>
                  {engagementByFormat.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== QUIZ PERFORMANCE TAB ======================== */}
      {tab === 'quiz' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Quiz Performance Table */}
          <Card title="Quiz Performance by Topic" span={2}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Topic</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Attempts</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Min Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Max Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Pass Rate</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Passes</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Fails</th>
                  </tr>
                </thead>
                <tbody>
                  {quizPerformance.map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{q.topic}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(q.attempts)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(q.avg_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(q.min_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(q.max_score)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(q.pass_rate)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#22c55e' }}>{fmt(q.passes)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#ef4444' }}>{fmt(q.fails)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Pass/Fail Stacked Bar Chart by Topic */}
          <Card title="Pass / Fail by Topic">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={quizPerformance}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="topic" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="passes" stackId="pf" fill={COLORS[1]} name="Passes" radius={[0, 0, 0, 0]} />
                <Bar dataKey="fails" stackId="pf" fill={COLORS[4]} name="Fails" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Quiz Score Radar Chart by Topic */}
          <Card title="Quiz Score Radar by Topic">
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={quizPerformance}>
                <PolarGrid />
                <PolarAngleAxis dataKey="topic" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fontSize: 10 }} />
                <Tooltip formatter={v => fmtPct(v)} />
                <Legend />
                <Radar name="Avg Score" dataKey="avg_score" stroke={COLORS[0]} fill={COLORS[0]} fillOpacity={0.3} />
                <Radar name="Pass Rate" dataKey="pass_rate" stroke={COLORS[1]} fill={COLORS[1]} fillOpacity={0.2} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Terms */}
          <Card title="Terms & Definitions" span={terms.length > 0 ? 2 : 1}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {terms.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Module Descriptions */}
          <Card title="Module Descriptions" span={moduleDescriptions.length > 0 ? 2 : 1}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 12px', color: '#64748b' }}>Module</th>
                  <th style={{ textAlign: 'left', padding: '6px 12px', color: '#64748b' }}>Topic</th>
                  <th style={{ textAlign: 'left', padding: '6px 12px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {moduleDescriptions.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155', whiteSpace: 'nowrap' }}>{m.module}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b', whiteSpace: 'nowrap' }}>{m.topic}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
