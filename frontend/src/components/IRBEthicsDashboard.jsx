import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '--')

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

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high' || s === 'critical') return '#ef4444'
  if (s === 'moderate' || s === 'medium') return '#f59e0b'
  if (s === 'low' || s === 'minimal') return '#22c55e'
  return '#94a3b8'
}

const statusColor = (status) => {
  const s = (status || '').toLowerCase()
  if (s === 'approved' || s === 'completed' || s === 'accepted' || s === 'consented') return '#22c55e'
  if (s === 'pending' || s === 'in_review' || s === 'in review') return '#f59e0b'
  if (s === 'rejected' || s === 'denied' || s === 'overridden' || s === 'withdrawn') return '#ef4444'
  if (s === 'waived' || s === 'exempt') return '#7c4dff'
  return '#94a3b8'
}

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }
const tdStyle = (i) => ({ padding: '8px 12px', borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' })

export default function IRBEthicsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/irb-ethics/overview`),
      axios.get(`${API_URL}/api/irb-ethics/breakdown`),
      axios.get(`${API_URL}/api/irb-ethics/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading IRB / Ethics Officer Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No data available</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'protocol', label: 'Protocol Compliance' },
    { id: 'consent', label: 'Consent Tracking' },
    { id: 'risk_benefit', label: 'Risk-Benefit Analysis' },
    { id: 'profiles', label: 'Patient Ethics Profiles' },
    { id: 'audit', label: 'Audit Trail' },
    { id: 'vulnerable', label: 'Vulnerable Populations' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}
  const patients = breakdown?.per_patient_ethics_profile || []
  const actorAuditData = breakdown?.actor_audit || []
  const componentAuditData = breakdown?.component_audit || {}
  const dataAccessLogData = breakdown?.data_access_log || []
  const vulnerableFlagsData = breakdown?.vulnerable_population_flags || []
  const workflowSteps = overview?.protocol_compliance?.workflow_steps || []
  const consentList = overview?.consent_tracking || []
  const riskBenefit = overview?.risk_benefit_summary || {}
  const aiOversight = overview?.ai_oversight_summary || {}
  const timeline = overview?.data_action_timeline || []
  const confidenceDist = riskBenefit.confidence_distribution || []

  // Derived charts
  const consentPie = [
    { name: 'Consented', value: consentList.filter(c => c.has_consent).length },
    { name: 'No Consent', value: consentList.filter(c => !c.has_consent).length },
  ].filter(d => d.value > 0)

  const riskCounts = patients.reduce((acc, p) => { acc[p.risk_level] = (acc[p.risk_level] || 0) + 1; return acc }, {})
  const riskDistribution = Object.entries(riskCounts).map(([name, value]) => ({ name, value }))

  const decisionOutcomes = [
    { name: 'AI Accepted', value: riskBenefit.ai_accepted || 0 },
    { name: 'AI Overridden', value: riskBenefit.ai_overridden || 0 },
    { name: 'HITL Overrides', value: riskBenefit.hitl_overrides || 0 },
  ].filter(d => d.value > 0)

  const componentChartData = Object.entries(componentAuditData).map(([k, v]) => ({
    component: k, count: typeof v === 'number' ? v : (v && v.count) || 0
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        IRB / Ethics Officer Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Protocol compliance, informed consent, risk-benefit analysis, audit trails, and vulnerable population oversight
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#1e88e5' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Tab: Overview ── */}
      {tab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients Enrolled" value={fmt(k.total_patients_enrolled)} color="#1e88e5" /></Card>
            <Card><KPI label="Data Actions" value={fmt(k.total_data_actions)} color="#7c4dff" /></Card>
            <Card><KPI label="Consent Coverage" value={k.informed_consent_coverage_pct != null ? `${k.informed_consent_coverage_pct}%` : '--'} color="#22c55e" /></Card>
            <Card><KPI label="Expert Reviews" value={fmt(k.expert_reviews_conducted)} color="#f59e0b" /></Card>
            <Card><KPI label="Human Overrides" value={fmt(k.human_overrides)} color="#ef4444" /></Card>
            <Card><KPI label="Vulnerable Patients" value={fmt(vulnerableFlagsData.length)} color="#ec4899" /></Card>
            <Card><KPI label="Assessments" value={fmt(k.assessments_administered)} color="#6366f1" /></Card>
            <Card><KPI label="Audit Completeness" value={k.audit_trail_completeness_pct != null ? `${k.audit_trail_completeness_pct}%` : '--'} color="#14b8a6" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Protocol Compliance Bar */}
            <Card title="Protocol Compliance (Workflow Steps)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={workflowSteps} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="step" width={130} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="pct" fill="#1e88e5" radius={[0, 4, 4, 0]}>
                    {workflowSteps.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Data Action Timeline */}
            <Card title="Data Action Timeline">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} name="Actions" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Decision Outcome Pie */}
            <Card title="AI Decision Outcomes">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={decisionOutcomes} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    {decisionOutcomes.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Risk Level Distribution */}
            <Card title="Patient Risk Level Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={riskDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {riskDistribution.map((entry, i) => (
                      <Cell key={i} fill={riskColor(entry.name)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Tab: Protocol Compliance ── */}
      {tab === 'protocol' && (
        <>
          <Card title="Workflow Step Completion Rates">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={workflowSteps} margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="pct" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                  {workflowSteps.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Protocol Step Details">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Step</th>
                    <th style={thStyle}>Total Patients</th>
                    <th style={thStyle}>Completed</th>
                    <th style={thStyle}>Completion Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {workflowSteps.map((step, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{step.step || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(step.total)}</td>
                      <td style={tdStyle(i)}>{fmt(step.patients_completed)}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(step.pct >= 80 ? '#22c55e' : step.pct >= 50 ? '#f59e0b' : '#ef4444')}>
                          {step.pct != null ? `${step.pct}%` : '--'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Tab: Consent Tracking ── */}
      {tab === 'consent' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="Consent Status Distribution">
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={consentPie} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, value }) => `${name}: ${value}`}>
                    {consentPie.map((_, i) => <Cell key={i} fill={i === 0 ? '#22c55e' : '#ef4444'} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="AI Oversight Agreement">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, padding: 20 }}>
                <KPI label="Expert Reviews" value={fmt(aiOversight.expert_reviews_total)} color="#1e88e5" />
                <KPI label="Agreement Rate" value={aiOversight.agreement_rate_pct != null ? `${aiOversight.agreement_rate_pct}%` : '--'} color={aiOversight.agreement_rate_pct >= 80 ? '#22c55e' : '#f59e0b'} />
                <KPI label="Agrees with AI" value={fmt(aiOversight.expert_agrees_with_ai)} color="#22c55e" />
                <KPI label="Disagrees with AI" value={fmt(aiOversight.expert_disagrees_with_ai)} color="#ef4444" />
              </div>
            </Card>
          </div>
          <Card title={`Consent Tracking (${consentList.length} patients)`}>
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient ID</th>
                    <th style={thStyle}>Consent Status</th>
                  </tr>
                </thead>
                <tbody>
                  {consentList.map((rec, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{rec.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(rec.has_consent ? '#22c55e' : '#ef4444')}>{rec.has_consent ? 'Consented' : 'No Consent'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Tab: Risk-Benefit Analysis ── */}
      {tab === 'risk_benefit' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Acceptance vs Override */}
            <Card title="AI Decision Acceptance vs Override">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={decisionOutcomes} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    {decisionOutcomes.map((_, i) => (
                      <Cell key={i} fill={['#22c55e', '#ef4444', '#f59e0b'][i] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Confidence Distribution */}
            <Card title="AI Confidence Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={confidenceDist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#7c4dff" radius={[4, 4, 0, 0]} name="Decisions" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Risk-Benefit Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: 12 }}>
              <KPI label="AI Decisions Total" value={fmt(riskBenefit.ai_decisions_total)} color="#1e88e5" />
              <KPI label="AI Accepted" value={fmt(riskBenefit.ai_accepted)} color="#22c55e" />
              <KPI label="AI Overridden" value={fmt(riskBenefit.ai_overridden)} color="#ef4444" />
              <KPI label="HITL Overrides" value={fmt(riskBenefit.hitl_overrides)} color="#f59e0b" />
            </div>
          </Card>
        </>
      )}

      {/* ── Tab: Patient Ethics Profiles ── */}
      {tab === 'profiles' && (
        <Card title={`Patient Ethics Profiles (${patients.length})`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient ID</th>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Age</th>
                  <th style={thStyle}>Gender</th>
                  <th style={thStyle}>Consent</th>
                  <th style={thStyle}>Assessments</th>
                  <th style={thStyle}>Data Actions</th>
                  <th style={thStyle}>Expert Reviews</th>
                  <th style={thStyle}>Risk Level</th>
                  <th style={thStyle}>Risk Reasons</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((pat, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{pat.patient_id || '--'}</td>
                    <td style={tdStyle(i)}>{pat.name || '--'}</td>
                    <td style={tdStyle(i)}>{pat.age || '--'}</td>
                    <td style={tdStyle(i)}>{pat.gender || '--'}</td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(pat.has_consent ? '#22c55e' : '#ef4444')}>{pat.has_consent ? 'Yes' : 'No'}</span>
                    </td>
                    <td style={tdStyle(i)}>{fmt(pat.n_assessments)}</td>
                    <td style={tdStyle(i)}>{fmt(pat.n_data_actions)}</td>
                    <td style={tdStyle(i)}>{(pat.expert_reviews || []).length}</td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(riskColor(pat.risk_level))}>{pat.risk_level || '--'}</span>
                    </td>
                    <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>
                      {(pat.risk_reasons || []).length > 0 ? pat.risk_reasons.join('; ') : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {patients.length === 0 && (
            <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No patient profiles available</div>
          )}
        </Card>
      )}

      {/* ── Tab: Audit Trail ── */}
      {tab === 'audit' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16, marginBottom: 16 }}>
            {/* Actor Audit Bar */}
            <Card title="Actions by Actor">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={actorAuditData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="actor" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="action_count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Actions" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Component Audit Bar */}
            <Card title="Actions by Component">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={componentChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="component" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Actions" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Recent Data Access Log */}
          <Card title={`Recent Data Access Log (${dataAccessLogData.length} entries)`}>
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>ID</th>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Action</th>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Detail</th>
                    <th style={thStyle}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {dataAccessLogData.map((entry, i) => (
                    <tr key={i}>
                      <td style={tdStyle(i)}>{entry.id}</td>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{entry.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle('#6366f1')}>{entry.component || '--'}</span>
                      </td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(entry.action === 'delete' ? '#ef4444' : '#1e88e5')}>
                          {entry.action || '--'}
                        </span>
                      </td>
                      <td style={tdStyle(i)}>{entry.actor || '--'}</td>
                      <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={entry.detail}>{entry.detail || '--'}</td>
                      <td style={{ ...tdStyle(i), fontSize: 11 }}>{entry.ts_utc ? entry.ts_utc.slice(0, 16).replace('T', ' ') : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Tab: Vulnerable Populations ── */}
      {tab === 'vulnerable' && (
        <>
          <Card title="Vulnerable Population Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Flagged" value={fmt(vulnerableFlagsData.length)} color="#ec4899" />
              <KPI label="Elderly (>65)" value={fmt(vulnerableFlagsData.filter(v => v.flag === 'elderly').length)} color="#7c4dff" />
              <KPI label="Pediatric (<18)" value={fmt(vulnerableFlagsData.filter(v => v.flag === 'pediatric').length)} color="#f59e0b" />
            </div>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Flagged Patients">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient ID</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Age</th>
                    <th style={thStyle}>Flag</th>
                    <th style={thStyle}>IRB Note</th>
                  </tr>
                </thead>
                <tbody>
                  {vulnerableFlagsData.map((pat, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{pat.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>{pat.name || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(pat.age)}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(pat.flag === 'elderly' ? '#7c4dff' : pat.flag === 'pediatric' ? '#f59e0b' : '#ef4444')}>
                          {pat.flag || '--'}
                        </span>
                      </td>
                      <td style={{ ...tdStyle(i), fontSize: 12, maxWidth: 400 }}>{pat.irb_note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {vulnerableFlagsData.length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No vulnerable population records identified</div>
            )}
          </Card>
        </>
      )}

      {/* ── Tab: Definitions ── */}
      {tab === 'definitions' && defs && (
        <Card title="IRB / Research Ethics Definitions">
          {(defs.terms || []).map((c, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < (defs.terms || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{c.term}</div>
              <div style={{ fontSize: 13, color: '#475569', marginTop: 4, lineHeight: 1.6 }}>{c.definition}</div>
            </div>
          ))}
        </Card>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: irb_ethics_dashboard.py &middot; Patients ({fmt(k.total_patients_enrolled)}) &middot; Data Actions ({fmt(k.total_data_actions)}) &middot; Expert Reviews ({fmt(k.expert_reviews_conducted)})
      </div>
    </div>
  )
}
