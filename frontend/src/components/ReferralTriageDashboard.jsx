import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'

const URGENCY_COLORS = {
  emergent: '#ef4444',
  urgent: '#f97316',
  routine: '#3b82f6',
  elective: '#22c55e'
}
const URGENCY_LABELS = {
  emergent: 'Emergent',
  urgent: 'Urgent',
  routine: 'Routine',
  elective: 'Elective'
}

const TRIAGE_STATUS_COLORS = {
  pending_triage: '#eab308',
  triaged: '#3b82f6',
  scheduled: '#8b5cf6',
  in_progress: '#06b6d4',
  completed: '#22c55e',
  cancelled: '#94a3b8'
}
const TRIAGE_STATUS_LABELS = {
  pending_triage: 'Pending Triage',
  triaged: 'Triaged',
  scheduled: 'Scheduled',
  in_progress: 'In Progress',
  completed: 'Completed',
  cancelled: 'Cancelled'
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

function UrgencyBadge({ urgency }) {
  const color = URGENCY_COLORS[urgency] || '#94a3b8'
  const label = URGENCY_LABELS[urgency] || urgency || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function TriageStatusBadge({ status }) {
  const color = TRIAGE_STATUS_COLORS[status] || '#94a3b8'
  const label = TRIAGE_STATUS_LABELS[status] || status || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function ReferralTriageDashboard() {
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
          axios.get(`${API_URL}/referral-triage/overview`),
          axios.get(`${API_URL}/referral-triage/breakdown`),
          axios.get(`${API_URL}/referral-triage/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Referral Triage data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No referral triage data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'referrals', label: 'Referrals' },
    { id: 'triage', label: 'Triage' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const urgencyDistData = overview?.urgency_distribution
    ? Object.entries(overview.urgency_distribution).map(([k, v]) => ({
        name: URGENCY_LABELS[k] || k, value: v, color: URGENCY_COLORS[k] || '#94a3b8'
      }))
    : []

  const sourceDistData = overview?.source_distribution
    ? Object.entries(overview.source_distribution).map(([k, v]) => ({
        name: k, count: typeof v === 'object' ? (v.total || 0) : v
      }))
    : []

  const timelineData = overview?.triage_timeline || []

  /* Breakdown data prep */
  const reasonDistData = breakdown?.reason_distribution
    ? Object.entries(breakdown.reason_distribution).map(([k, v]) => ({
        name: k, count: typeof v === 'object' ? (v.total || 0) : v
      }))
    : []

  const urgencyBySourceData = breakdown?.urgency_by_source
    ? Object.entries(breakdown.urgency_by_source).map(([source, urgencies]) => ({
        name: source,
        ...(typeof urgencies === 'object' ? urgencies : {})
      }))
    : []

  const recentReferrals = breakdown?.recent_referrals || []

  /* Triage data prep */
  const statusSummaryData = breakdown?.status_summary
    ? Object.entries(breakdown.status_summary).map(([k, v]) => ({
        name: TRIAGE_STATUS_LABELS[k] || k, value: v, color: TRIAGE_STATUS_COLORS[k] || '#94a3b8'
      }))
    : []

  const providerWorkloadData = breakdown?.provider_workload || []

  const pendingTriageList = recentReferrals.filter(r => r.status === 'pending_triage')

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Referral Intake & Triage Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Referral tracking, urgency classification, and triage workflow management
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
              <KPI label="Total Referrals" value={fmt(overview.total_referrals)} />
              <KPI label="Pending Triage" value={fmt(overview.pending_triage)} color="#eab308" />
              <KPI label="Avg Triage Time (hrs)" value={fmt(overview.avg_triage_time_hrs)} />
              <KPI label="Urgent Pending" value={fmt(overview.urgent_pending)} color="#f97316" />
              <KPI label="Completion Rate" value={overview.completion_rate_pct != null ? fmt(overview.completion_rate_pct) + '%' : '--'} color="#22c55e" />
            </div>
          </Card>

          {/* Pie: Urgency Distribution */}
          <Card title="Urgency Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={urgencyDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {urgencyDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {urgencyDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Source Distribution */}
          <Card title="Referral Sources" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sourceDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" name="Referrals" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: Triage Timeline */}
          <Card title="Triage Timeline (30 days)" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="referrals" stroke="#3b82f6" name="Referrals" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="triaged" stroke="#22c55e" name="Triaged" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Referrals */}
      {tab === 'referrals' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Referral KPIs */}
          {breakdown.referral_kpis && (
            <Card span={2}>
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${(breakdown.referral_kpis || []).length || 4}, 1fr)`, gap: 16 }}>
                {(breakdown.referral_kpis || []).map((kpi, i) => (
                  <KPI key={i} label={kpi.label} value={fmt(kpi.value)} sub={kpi.sub} color={kpi.color} />
                ))}
              </div>
            </Card>
          )}

          {/* Bar: Reason Distribution */}
          <Card title="Referral Reasons">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={reasonDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Stacked Bar: Urgency by Source */}
          <Card title="Urgency by Source">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={urgencyBySourceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="emergent" fill={URGENCY_COLORS.emergent} name="Emergent" stackId="a" />
                <Bar dataKey="urgent" fill={URGENCY_COLORS.urgent} name="Urgent" stackId="a" />
                <Bar dataKey="routine" fill={URGENCY_COLORS.routine} name="Routine" stackId="a" />
                <Bar dataKey="elective" fill={URGENCY_COLORS.elective} name="Elective" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Recent Referrals Table */}
          <Card title="Recent Referrals" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Source</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Reason</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Urgency</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Assigned To</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Referred</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Triaged</th>
                  </tr>
                </thead>
                <tbody>
                  {recentReferrals.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_name || row.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.source || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.reason || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><UrgencyBadge urgency={row.urgency} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><TriageStatusBadge status={row.status} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.assigned_to || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{row.referred_date || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{row.triaged_date || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Triage */}
      {tab === 'triage' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Pie: Status Summary */}
          <Card title="Triage Status Summary">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={statusSummaryData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {statusSummaryData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {statusSummaryData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Provider Workload */}
          <Card title="Provider Workload">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={providerWorkloadData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="provider" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="assigned" fill="#3b82f6" name="Assigned" />
                <Bar dataKey="completed" fill="#22c55e" name="Completed" />
                <Bar dataKey="pending" fill="#eab308" name="Pending" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pending Triage List */}
          <Card title="Pending Triage" span={2}>
            {pendingTriageList.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No referrals pending triage.</div>
            ) : (
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Source</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Reason</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Urgency</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Referred Date</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Assigned To</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pendingTriageList.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.source || '--'}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.reason || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><UrgencyBadge urgency={row.urgency} /></td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{row.referred_date || '--'}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.assigned_to || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 4: Analytics */}
      {tab === 'analytics' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Urgency by Source Cross-Tab */}
          <Card title="Urgency by Source" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={urgencyBySourceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="emergent" fill={URGENCY_COLORS.emergent} name="Emergent" stackId="a" />
                <Bar dataKey="urgent" fill={URGENCY_COLORS.urgent} name="Urgent" stackId="a" />
                <Bar dataKey="routine" fill={URGENCY_COLORS.routine} name="Routine" stackId="a" />
                <Bar dataKey="elective" fill={URGENCY_COLORS.elective} name="Elective" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Triage Timeline with Trend */}
          <Card title="Triage Timeline Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="referrals" stroke="#3b82f6" name="Referrals" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="triaged" stroke="#22c55e" name="Triaged" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Reason Distribution Horizontal Bar */}
          <Card title="Reason Distribution" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(200, reasonDistData.length * 30)}>
              <BarChart data={reasonDistData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={140} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Metric Definitions */}
          {defs.metric_definitions && defs.metric_definitions.length > 0 && (
            <Card title="Metric Definitions">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.metric_definitions.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.metric || d.name}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition || d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Urgency Level Criteria */}
          {defs.urgency_criteria && defs.urgency_criteria.length > 0 && (
            <Card title="Urgency Level Criteria">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.urgency_criteria.map((uc, i) => {
                  const color = URGENCY_COLORS[uc.level] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: color + '11', border: `1px solid ${color}33`, borderRadius: 8 }}>
                      <h4 style={{ margin: '0 0 8px', fontSize: 13, color }}>{URGENCY_LABELS[uc.level] || uc.level}</h4>
                      <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{uc.description}</p>
                      {uc.criteria && (
                        <ul style={{ margin: '8px 0 0', padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                          {uc.criteria.map((c, j) => <li key={j} style={{ marginBottom: 4 }}>{c}</li>)}
                        </ul>
                      )}
                      {uc.response_time && (
                        <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
                          Response time: <strong style={{ color }}>{uc.response_time}</strong>
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Triage Scoring */}
          {defs.triage_scoring && (
            <Card title="Triage Scoring">
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                {defs.triage_scoring.description && (
                  <p style={{ margin: '0 0 12px', fontSize: 13, color: '#475569' }}>{defs.triage_scoring.description}</p>
                )}
                {defs.triage_scoring.factors && defs.triage_scoring.factors.length > 0 && (
                  <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Factor</th>
                        <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Weight</th>
                        <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {defs.triage_scoring.factors.map((f, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155' }}>{f.factor || f.name}</td>
                          <td style={{ padding: '6px 8px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{f.weight || '--'}</td>
                          <td style={{ padding: '6px 8px', color: '#475569' }}>{f.description || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
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
