import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, FunnelChart, Funnel, LabelList
} from 'recharts'

const API_URL = '/api'
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
      axios.get(`${API_URL}/irb-ethics/overview`),
      axios.get(`${API_URL}/irb-ethics/breakdown`),
      axios.get(`${API_URL}/irb-ethics/definitions`),
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
            <Card><KPI label="Total Patients" value={k.total_patients} color="#1e88e5" /></Card>
            <Card><KPI label="Protocols Reviewed" value={k.protocols_reviewed} color="#7c4dff" /></Card>
            <Card><KPI label="Consent Rate" value={k.consent_rate != null ? `${k.consent_rate}%` : '--'} color="#22c55e" /></Card>
            <Card><KPI label="Pending Reviews" value={k.pending_reviews} color="#f59e0b" /></Card>
            <Card><KPI label="AI Overrides" value={k.ai_overrides} color="#ef4444" /></Card>
            <Card><KPI label="Vulnerable Patients" value={k.vulnerable_patients} color="#ec4899" /></Card>
            <Card><KPI label="Audit Events" value={k.audit_events} color="#6366f1" /></Card>
            <Card><KPI label="Compliance Score" value={k.compliance_score != null ? `${k.compliance_score}%` : '--'} sub={k.compliance_grade || ''} color="#14b8a6" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Protocol Compliance Funnel */}
            <Card title="Protocol Compliance Funnel">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={overview.protocol_funnel || []} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#1e88e5" radius={[0, 4, 4, 0]}>
                    {(overview.protocol_funnel || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Data Action Timeline */}
            <Card title="Data Action Timeline">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={overview.data_action_timeline || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="actions" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Decision Outcome Pie */}
            <Card title="AI Decision Outcomes">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={overview.decision_outcomes || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    {(overview.decision_outcomes || []).map((_, i) => (
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
                <BarChart data={overview.risk_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {(overview.risk_distribution || []).map((entry, i) => (
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
              <BarChart data={breakdown?.protocol_steps || []} margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="completion_rate" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                  {(breakdown?.protocol_steps || []).map((_, i) => (
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
                    <th style={thStyle}>Total</th>
                    <th style={thStyle}>Completed</th>
                    <th style={thStyle}>Completion Rate</th>
                    <th style={thStyle}>Avg Duration</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.protocol_steps || []).map((step, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{step.step || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(step.total)}</td>
                      <td style={tdStyle(i)}>{fmt(step.completed)}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(step.completion_rate >= 80 ? '#22c55e' : step.completion_rate >= 50 ? '#f59e0b' : '#ef4444')}>
                          {step.completion_rate != null ? `${step.completion_rate}%` : '--'}
                        </span>
                      </td>
                      <td style={tdStyle(i)}>{step.avg_duration || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(statusColor(step.status))}>{step.status || '--'}</span>
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
        <Card title={`Consent Status (${(breakdown?.consent_records || []).length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Consent Status</th>
                  <th style={thStyle}>Consent Date</th>
                  <th style={thStyle}>Consent Type</th>
                  <th style={thStyle}>IRB Protocol</th>
                  <th style={thStyle}>Withdrawal Date</th>
                  <th style={thStyle}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.consent_records || []).map((rec, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{rec.patient_id || '--'}</td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(statusColor(rec.consent_status))}>{rec.consent_status || '--'}</span>
                    </td>
                    <td style={tdStyle(i)}>{rec.consent_date || '--'}</td>
                    <td style={tdStyle(i)}>{rec.consent_type || '--'}</td>
                    <td style={tdStyle(i)}>{rec.irb_protocol || '--'}</td>
                    <td style={tdStyle(i)}>{rec.withdrawal_date || '--'}</td>
                    <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>{rec.notes || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(breakdown?.consent_records || []).length === 0 && (
            <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No consent records available</div>
          )}
        </Card>
      )}

      {/* ── Tab: Risk-Benefit Analysis ── */}
      {tab === 'risk_benefit' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Acceptance vs Override */}
            <Card title="AI Decision Acceptance vs Override">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={breakdown?.acceptance_override || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    <Cell fill="#22c55e" />
                    <Cell fill="#ef4444" />
                    <Cell fill="#f59e0b" />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Confidence Distribution */}
            <Card title="AI Confidence Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={breakdown?.confidence_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Risk-Benefit Detail Log">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>AI Decision</th>
                    <th style={thStyle}>Confidence</th>
                    <th style={thStyle}>Clinician Action</th>
                    <th style={thStyle}>Override Reason</th>
                    <th style={thStyle}>Risk Level</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.risk_benefit_log || []).map((entry, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{entry.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>{entry.ai_decision || '--'}</td>
                      <td style={tdStyle(i)}>
                        {entry.confidence != null
                          ? <span style={badgeStyle(entry.confidence >= 0.8 ? '#22c55e' : entry.confidence >= 0.5 ? '#f59e0b' : '#ef4444')}>
                              {(entry.confidence * 100).toFixed(0)}%
                            </span>
                          : '--'}
                      </td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(statusColor(entry.clinician_action))}>{entry.clinician_action || '--'}</span>
                      </td>
                      <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>{entry.override_reason || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(riskColor(entry.risk_level))}>{entry.risk_level || '--'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.risk_benefit_log || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No risk-benefit records available</div>
            )}
          </Card>
        </>
      )}

      {/* ── Tab: Patient Ethics Profiles ── */}
      {tab === 'profiles' && (
        <>
          {(breakdown?.patient_profiles || []).map((pat, pi) => (
            <div key={pi} style={{ marginBottom: 16 }}>
              <Card title={`Patient ${pat.patient_id}`}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Age</div>
                    <div style={{ fontSize: 15, fontWeight: 600 }}>{fmt(pat.age)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Risk Level</div>
                    <span style={badgeStyle(riskColor(pat.risk_level))}>{pat.risk_level || '--'}</span>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Consent Status</div>
                    <span style={badgeStyle(statusColor(pat.consent_status))}>{pat.consent_status || '--'}</span>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>Vulnerable</div>
                    <span style={badgeStyle(pat.is_vulnerable ? '#ef4444' : '#22c55e')}>{pat.is_vulnerable ? 'Yes' : 'No'}</span>
                  </div>
                </div>

                {(pat.risk_factors || []).length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Risk Factors</div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {pat.risk_factors.map((rf, i) => (
                        <span key={i} style={badgeStyle('#f59e0b')}>{rf}</span>
                      ))}
                    </div>
                  </div>
                )}

                {(pat.ethical_flags || []).length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>Ethical Flags</div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {pat.ethical_flags.map((fl, i) => (
                        <span key={i} style={badgeStyle('#ef4444')}>{fl}</span>
                      ))}
                    </div>
                  </div>
                )}

                {pat.ai_decisions_count != null && (
                  <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>
                    AI Decisions: {fmt(pat.ai_decisions_count)} | Overrides: {fmt(pat.override_count)} | Last Review: {pat.last_review_date || '--'}
                  </div>
                )}
              </Card>
            </div>
          ))}
          {(breakdown?.patient_profiles || []).length === 0 && (
            <Card><div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No patient profiles available</div></Card>
          )}
        </>
      )}

      {/* ── Tab: Audit Trail ── */}
      {tab === 'audit' && (
        <>
          {/* Component Audit */}
          <Card title="Component Audit Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Total Actions</th>
                    <th style={thStyle}>Last Accessed</th>
                    <th style={thStyle}>Access Frequency</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.component_audit || []).map((comp, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{comp.component || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(comp.total_actions)}</td>
                      <td style={tdStyle(i)}>{comp.last_accessed || '--'}</td>
                      <td style={tdStyle(i)}>{comp.access_frequency || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(statusColor(comp.status))}>{comp.status || '--'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.component_audit || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No component audit data available</div>
            )}
          </Card>

          <div style={{ marginTop: 16 }} />

          {/* Actor Audit */}
          <Card title="Actor Audit Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Actions Performed</th>
                    <th style={thStyle}>Last Active</th>
                    <th style={thStyle}>Overrides</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.actor_audit || []).map((actor, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{actor.actor || '--'}</td>
                      <td style={tdStyle(i)}>{actor.role || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(actor.actions_performed)}</td>
                      <td style={tdStyle(i)}>{actor.last_active || '--'}</td>
                      <td style={tdStyle(i)}>
                        {actor.overrides > 0
                          ? <span style={badgeStyle('#ef4444')}>{actor.overrides}</span>
                          : <span style={{ color: '#22c55e', fontWeight: 600, fontSize: 12 }}>0</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.actor_audit || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No actor audit data available</div>
            )}
          </Card>

          <div style={{ marginTop: 16 }} />

          {/* Recent Data Access Log */}
          <Card title="Recent Data Access Log">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Timestamp</th>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Action</th>
                    <th style={thStyle}>Resource</th>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>IP / Source</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.access_log || []).map((entry, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontSize: 11 }}>{entry.timestamp || '--'}</td>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{entry.actor || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(entry.action === 'write' || entry.action === 'delete' ? '#ef4444' : '#1e88e5')}>
                          {entry.action || '--'}
                        </span>
                      </td>
                      <td style={tdStyle(i)}>{entry.resource || '--'}</td>
                      <td style={tdStyle(i)}>{entry.patient_id || '--'}</td>
                      <td style={{ ...tdStyle(i), fontSize: 11 }}>{entry.source || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.access_log || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No access log entries available</div>
            )}
          </Card>
        </>
      )}

      {/* ── Tab: Vulnerable Populations ── */}
      {tab === 'vulnerable' && (
        <>
          <Card title="Vulnerable Population Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 16 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#ec4899' }}>{breakdown?.vulnerable_summary?.total || '--'}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Total Vulnerable</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#f59e0b' }}>{breakdown?.vulnerable_summary?.under_18 || '--'}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Under 18 (Pediatric)</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#7c4dff' }}>{breakdown?.vulnerable_summary?.over_65 || '--'}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Over 65 (Geriatric)</div>
              </div>
            </div>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Flagged Patients">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Age</th>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Consent Status</th>
                    <th style={thStyle}>Guardian/Proxy</th>
                    <th style={thStyle}>IRB Scrutiny Level</th>
                    <th style={thStyle}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.vulnerable_patients || []).map((pat, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{pat.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(pat.age)}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(pat.category === 'pediatric' ? '#f59e0b' : '#7c4dff')}>
                          {pat.category || '--'}
                        </span>
                      </td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(statusColor(pat.consent_status))}>{pat.consent_status || '--'}</span>
                      </td>
                      <td style={tdStyle(i)}>{pat.guardian || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(riskColor(pat.irb_scrutiny_level))}>{pat.irb_scrutiny_level || '--'}</span>
                      </td>
                      <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>{pat.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.vulnerable_patients || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No vulnerable population records available</div>
            )}
          </Card>
        </>
      )}

      {/* ── Tab: Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="IRB / Ethics Concepts">
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{c.term}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{c.definition}</div>
              </div>
            ))}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Quality Metrics">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Metric</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Target</th>
                </tr>
              </thead>
              <tbody>
                {(defs.quality_metrics || []).map((m, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{m.metric}</td>
                    <td style={tdStyle(i)}>{m.description}</td>
                    <td style={tdStyle(i)}><span style={badgeStyle('#22c55e')}>{m.target}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Compliance References">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Standard</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.compliance_references || []).map((r, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{r.standard}</td>
                    <td style={tdStyle(i)}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Remediation Strategies">
            {(defs.remediation_strategies || []).map((s, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.remediation_strategies.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{s.strategy}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{s.description}</div>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: irb_ethics_module &middot; Patients ({fmt(k.total_patients)}) &middot; Protocols ({fmt(k.protocols_reviewed)}) &middot; Audit Events ({fmt(k.audit_events)})
      </div>
    </div>
  )
}
