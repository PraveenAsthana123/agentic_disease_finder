import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
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
  const colorMap = { completed: '#22c55e', pending: '#eab308', expired: '#ef4444', opened: '#3b82f6', complete: '#22c55e', failed: '#ef4444' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function SurveyLinkDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/survey-link/overview`),
          axios.get(`${API_URL}/survey-link/breakdown`),
          axios.get(`${API_URL}/survey-link/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Survey Link data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'assessments', label: 'Assessments' },
    { id: 'patients', label: 'Patients' },
    { id: 'tracking', label: 'Tracking' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const assessmentDistribution = overview?.assessment_distribution || []
  const statusBreakdown = overview?.status_breakdown || []
  const completionByType = overview?.completion_by_type || []
  const deliveryDistribution = overview?.delivery_distribution || []
  const responseTimeDistribution = overview?.response_time_distribution || []
  const expiryDistribution = overview?.expiry_distribution || []

  /* --- Breakdown data --- */
  const perPatientSummary = breakdown?.per_patient_summary || []
  const perAssessmentSummary = breakdown?.per_assessment_summary || []
  const recentLinks = breakdown?.recent_links || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Survey Link Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Tokenized self-service assessment link generation, tracking, and completion analytics
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
              <KPI label="Total Links Generated" value={fmt(overview.total_links_generated)} />
              <KPI label="Completed" value={fmt(overview.completed)} color="#22c55e" />
              <KPI label="Pending" value={fmt(overview.pending)} color="#eab308" />
              <KPI label="Completion Rate" value={fmtPct(overview.completion_rate)} color="#3b82f6" />
              <KPI label="Avg Response Time" value={`${fmt(overview.avg_response_hours)}h`} color="#8b5cf6" />
            </div>
          </Card>

          {/* Status Breakdown Pie */}
          <Card title="Link Status Breakdown">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={statusBreakdown} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {statusBreakdown.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Assessment Type Distribution */}
          <Card title="Links by Assessment Type">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={assessmentDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="assessment" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Delivery Method Distribution */}
          <Card title="Delivery Methods">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={deliveryDistribution} dataKey="count" nameKey="method" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {deliveryDistribution.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Response Time Distribution */}
          <Card title="Response Time Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={responseTimeDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#22c55e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Expiry Configuration */}
          <Card title="Link Expiry Settings">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={expiryDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="expiry_days" label={{ value: 'Days', position: 'bottom' }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f97316" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== ASSESSMENTS TAB ======================== */}
      {tab === 'assessments' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Completion Rate by Assessment Type */}
          <Card title="Completion Rate by Assessment" span={3}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={completionByType}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="assessment" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v, name) => name === 'completion_rate' ? fmtPct(v) : v} />
                <Bar dataKey="completion_rate" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Completion Rate" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Assessment Summary Table */}
          <Card title="Assessment Type Summary" span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Assessment</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Sent</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Completed</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Rate</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Score</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Min</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Max</th>
                  </tr>
                </thead>
                <tbody>
                  {perAssessmentSummary.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{a.assessment_type}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(a.total_sent)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(a.completed)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmtPct(a.completion_rate)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(a.avg_score)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(a.min_score)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(a.max_score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Score Distribution Radar */}
          <Card title="Average Scores by Assessment" span={3}>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={perAssessmentSummary.map(a => ({ ...a, avg_score_norm: a.avg_score != null ? a.avg_score : 0 }))}>
                <PolarGrid />
                <PolarAngleAxis dataKey="assessment_type" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis />
                <Radar name="Avg Score" dataKey="avg_score_norm" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== PATIENTS TAB ======================== */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Patient Summary KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview?.total_patients)} />
              <KPI label="Total Links" value={fmt(overview?.total_links_generated)} color="#3b82f6" />
              <KPI label="Expired" value={fmt(overview?.expired)} color="#ef4444" />
              <KPI label="Avg Response" value={`${fmt(overview?.avg_response_hours)}h`} color="#8b5cf6" />
            </div>
          </Card>

          {/* Per-Patient Table */}
          <Card title="Patient Survey Summary" span={3}>
            <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Patient</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Links</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Completed</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Pending</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Expired</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Response (h)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Assessments</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatientSummary.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_name}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.total_links)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#22c55e' }}>{fmt(p.completed)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#eab308' }}>{fmt(p.pending)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#ef4444' }}>{fmt(p.expired)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.avg_response_hours)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11 }}>{(p.assessments || []).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== TRACKING TAB ======================== */}
      {tab === 'tracking' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Recent Links Table */}
          <Card title="Recent Survey Links" span={3}>
            <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Patient</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Assessment</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Token</th>
                    <th style={{ padding: '6px 10px', textAlign: 'center' }}>Status</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right' }}>Score</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right' }}>Response (h)</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left' }}>Delivery</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right' }}>Expiry (d)</th>
                  </tr>
                </thead>
                <tbody>
                  {recentLinks.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px' }}>{l.patient_name}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{l.assessment_type}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{l.token}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={l.status} /></td>
                      <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(l.score)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(l.response_hours)}</td>
                      <td style={{ padding: '6px 10px' }}>{(l.delivery_method || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right' }}>{l.expiry_days}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={definitions.title}>
            {(definitions.definitions || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 18, paddingBottom: 14, borderBottom: i < definitions.definitions.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{d.term}
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#94a3b8', fontWeight: 400 }}>[{d.category}]</span>
                </div>
                <div style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{d.definition}</div>
                <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>{d.clinical_relevance}</div>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
