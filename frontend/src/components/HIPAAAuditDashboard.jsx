import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const map = {
    granted: '#10b981', compliant: '#10b981', healthy: '#10b981',
    pending: '#f59e0b', partial: '#f59e0b',
    declined: '#ef4444', expired: '#ef4444', degraded: '#ef4444',
    withdrawn: '#8b5cf6'
  }
  const color = map[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 11, textTransform: 'uppercase'
    }}>{status}</span>
  )
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const TH = { padding: '8px 10px', textAlign: 'left', fontSize: 12, color: '#475569', background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }
const TD = { padding: '8px 10px', fontSize: 12, color: '#334155', borderBottom: '1px solid #f1f5f9' }

export default function HIPAAAuditDashboard() {
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
          axios.get(`${API_URL}/api/hipaa-audit/overview`),
          axios.get(`${API_URL}/api/hipaa-audit/breakdown`),
          axios.get(`${API_URL}/api/hipaa-audit/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load HIPAA audit data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading HIPAA audit data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>HIPAA audit data not available</div>

  const tabs = ['overview', 'consents', 'audit', 'compliance', 'definitions']
  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>HIPAA Audit Pack Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        HIPAA compliance monitoring — {fmt(k.total_consents)} consent records, {fmt(k.total_audit_events)} audit events, compliance score {fmt(k.compliance_score)}%
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Consents" value={k.total_consents} />
              <KPI label="Granted" value={k.granted} color="#10b981" />
              <KPI label="Pending" value={k.pending} color="#f59e0b" />
              <KPI label="Consent Rate" value={k.consent_rate} sub="%" color="#10b981" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Audit Events" value={k.total_audit_events} />
              <KPI label="CAPAs Opened" value={k.capas_opened} color="#ef4444" />
              <KPI label="Documents" value={k.total_documents} />
              <KPI label="Compliance Score" value={k.compliance_score} sub="%" color="#3b82f6" />
            </div>
          </Card>

          <Card title="Consent Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.consent_status_dist || []} dataKey="count" nameKey="status"
                  cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`}>
                  {(overview.consent_status_dist || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Audit Events by Category" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.events_by_category || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Consent by Type" span={4}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.consent_by_type || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="granted" name="Granted" stackId="a" fill="#10b981" />
                <Bar dataKey="pending" name="Pending" stackId="a" fill="#f59e0b" />
                <Bar dataKey="declined" name="Declined" stackId="a" fill="#ef4444" />
                <Bar dataKey="expired" name="Expired" stackId="a" fill="#64748b" />
                <Bar dataKey="withdrawn" name="Withdrawn" stackId="a" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Audit Events by Action" span={4}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.events_by_action || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="action" type="category" tick={{ fontSize: 11 }} width={160} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Consents ── */}
      {tab === 'consents' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Pending alerts */}
          {(breakdown.pending_consents || []).length > 0 && (
            <Card title="Pending Consent Alerts" span={1}>
              <div style={{ background: '#fef3c7', border: '1px solid #f59e0b', borderRadius: 8, padding: 12, marginBottom: 12, fontSize: 12, color: '#92400e' }}>
                {breakdown.pending_consents.length} consent(s) awaiting patient action
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr><th style={TH}>Patient</th><th style={TH}>Consent Type</th><th style={TH}>Notes</th></tr></thead>
                  <tbody>
                    {breakdown.pending_consents.map((c, i) => (
                      <tr key={i}><td style={TD}>{c.patient_id}</td><td style={TD}>{c.consent_type}</td><td style={TD}>{c.notes || '--'}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Expired alerts */}
          {(breakdown.expired_consents || []).length > 0 && (
            <Card title="Expired Consent Alerts">
              <div style={{ background: '#fef2f2', border: '1px solid #ef4444', borderRadius: 8, padding: 12, marginBottom: 12, fontSize: 12, color: '#991b1b' }}>
                {breakdown.expired_consents.length} consent(s) have expired — renewal required
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr><th style={TH}>Patient</th><th style={TH}>Consent Type</th><th style={TH}>Expiry Date</th><th style={TH}>Notes</th></tr></thead>
                  <tbody>
                    {breakdown.expired_consents.map((c, i) => (
                      <tr key={i}><td style={TD}>{c.patient_id}</td><td style={TD}>{c.consent_type}</td><td style={TD}>{c.expiry_date || '--'}</td><td style={TD}>{c.notes || '--'}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Per-patient consent matrix */}
          <Card title="Per-Patient Consent Matrix">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={TH}>Patient</th><th style={TH}>Granted</th><th style={TH}>Pending</th>
                  <th style={TH}>Declined</th><th style={TH}>Expired</th><th style={TH}>Withdrawn</th>
                  <th style={TH}>Total</th><th style={TH}>Compliance</th>
                </tr></thead>
                <tbody>
                  {(breakdown.patient_consent || []).map((p, i) => (
                    <tr key={i}>
                      <td style={TD}>{p.patient_id}</td>
                      <td style={{ ...TD, color: '#10b981', fontWeight: 600 }}>{p.granted}</td>
                      <td style={{ ...TD, color: '#f59e0b', fontWeight: 600 }}>{p.pending}</td>
                      <td style={{ ...TD, color: '#ef4444' }}>{p.declined}</td>
                      <td style={{ ...TD, color: '#64748b' }}>{p.expired}</td>
                      <td style={{ ...TD, color: '#8b5cf6' }}>{p.withdrawn}</td>
                      <td style={TD}>{p.total}</td>
                      <td style={TD}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div style={{ flex: 1, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(100, p.compliance_pct)}%`, height: '100%',
                              background: p.compliance_pct >= 80 ? '#10b981' : p.compliance_pct >= 50 ? '#f59e0b' : '#ef4444',
                              borderRadius: 4
                            }} />
                          </div>
                          <span style={{ fontSize: 11, color: '#64748b', minWidth: 36 }}>{fmt(p.compliance_pct)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Audit Trail ── */}
      {tab === 'audit' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Actor Workload">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr><th style={TH}>Actor</th><th style={TH}>Event Count</th></tr></thead>
                <tbody>
                  {(breakdown.actor_workload || []).map((a, i) => (
                    <tr key={i}><td style={TD}>{a.actor}</td><td style={TD}>{a.event_count}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* CAPA detail */}
          {(breakdown.capa_detail || []).length > 0 && (
            <Card title="CAPA (Corrective & Preventive Actions)">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr style={{ background: '#fef2f2' }}>
                    <th style={{ ...TH, color: '#991b1b', background: '#fef2f2' }}>Submission</th>
                    <th style={{ ...TH, color: '#991b1b', background: '#fef2f2' }}>Actor</th>
                    <th style={{ ...TH, color: '#991b1b', background: '#fef2f2' }}>Timestamp</th>
                    <th style={{ ...TH, color: '#991b1b', background: '#fef2f2' }}>Details</th>
                    <th style={{ ...TH, color: '#991b1b', background: '#fef2f2' }}>Document Ref</th>
                  </tr></thead>
                  <tbody>
                    {breakdown.capa_detail.map((c, i) => (
                      <tr key={i}>
                        <td style={TD}>{c.submission_id}</td><td style={TD}>{c.actor}</td>
                        <td style={TD}>{c.timestamp}</td><td style={TD}>{c.details}</td>
                        <td style={TD}>{c.document_ref}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Deviation detail */}
          {(breakdown.deviation_detail || []).length > 0 && (
            <Card title="Deviations Logged">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr>
                    <th style={TH}>Submission</th><th style={TH}>Actor</th><th style={TH}>Timestamp</th>
                    <th style={TH}>Details</th><th style={TH}>Document Ref</th>
                  </tr></thead>
                  <tbody>
                    {breakdown.deviation_detail.map((d, i) => (
                      <tr key={i}>
                        <td style={TD}>{d.submission_id}</td><td style={TD}>{d.actor}</td>
                        <td style={TD}>{d.timestamp}</td><td style={TD}>{d.details}</td>
                        <td style={TD}>{d.document_ref}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Recent audit events */}
          <Card title="Recent Audit Events (Last 20)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={TH}>Timestamp</th><th style={TH}>Action</th><th style={TH}>Actor</th>
                  <th style={TH}>Category</th><th style={TH}>Submission</th><th style={TH}>Document</th>
                </tr></thead>
                <tbody>
                  {(breakdown.recent_events || []).map((e, i) => (
                    <tr key={i}>
                      <td style={TD}>{e.timestamp}</td>
                      <td style={TD}>{e.action}</td>
                      <td style={TD}>{e.actor}</td>
                      <td style={TD}><StatusBadge status={e.category?.toLowerCase() || ''} /></td>
                      <td style={TD}>{e.submission_id}</td>
                      <td style={TD}>{e.document_ref}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* System security checks */}
          <Card title="System Security Checks">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={TH}>Timestamp</th><th style={TH}>Component</th><th style={TH}>Status</th>
                  <th style={TH}>Response (ms)</th><th style={TH}>CPU %</th><th style={TH}>Memory %</th>
                  <th style={TH}>Disk %</th><th style={TH}>Errors</th>
                </tr></thead>
                <tbody>
                  {(breakdown.security_checks || []).map((s, i) => (
                    <tr key={i}>
                      <td style={TD}>{s.timestamp}</td>
                      <td style={TD}>{s.component}</td>
                      <td style={TD}><StatusBadge status={s.status} /></td>
                      <td style={TD}>{s.response_time_ms}</td>
                      <td style={TD}>{fmt(s.cpu_pct)}</td>
                      <td style={TD}>{fmt(s.memory_pct)}</td>
                      <td style={TD}>{fmt(s.disk_pct)}</td>
                      <td style={{ ...TD, color: s.error_count > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{s.error_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Compliance ── */}
      {tab === 'compliance' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="HIPAA Rule Compliance Mapping">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={TH}>HIPAA Rule</th><th style={TH}>Area</th><th style={TH}>Status</th><th style={TH}>Evidence</th>
                </tr></thead>
                <tbody>
                  {(breakdown.hipaa_rules || []).map((r, i) => (
                    <tr key={i}>
                      <td style={{ ...TD, fontWeight: 600 }}>{r.rule}</td>
                      <td style={TD}>{r.area}</td>
                      <td style={TD}><StatusBadge status={r.status} /></td>
                      <td style={{ ...TD, fontSize: 11 }}>{r.evidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Security Posture Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="System Uptime" value={k.uptime_pct} sub="%" color="#10b981" />
              <KPI label="Total Errors" value={k.total_errors} color={k.total_errors > 0 ? '#ef4444' : '#10b981'} />
              <KPI label="Avg CPU" value={k.avg_cpu_pct} sub="%" color="#3b82f6" />
              <KPI label="Avg Memory" value={k.avg_memory_pct} sub="%" color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Compliance Score Breakdown">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {[
                { label: 'Consent Coverage (30%)', value: k.consent_rate, color: '#10b981' },
                { label: 'Audit Completeness (20%)', value: Math.min(100, (k.total_audit_events / 100) * 100), color: '#3b82f6' },
                { label: 'Document Coverage (25%)', value: k.doc_coverage_pct, color: '#8b5cf6' },
                { label: 'Security Uptime (25%)', value: k.uptime_pct, color: '#06b6d4' }
              ].map((c, i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>{c.label}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{ flex: 1, height: 12, background: '#e2e8f0', borderRadius: 6, overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(100, c.value || 0)}%`, height: '100%', background: c.color, borderRadius: 6 }} />
                    </div>
                    <span style={{ fontSize: 14, fontWeight: 700, color: c.color, minWidth: 48 }}>{fmt(c.value)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="HIPAA Rules" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {(defs.hipaa_rules || []).map((r, i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{r.rule}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{r.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Consent Types">
            {(defs.consent_types || []).map((c, i) => (
              <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{c.type}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{c.description}</div>
              </div>
            ))}
          </Card>

          <Card title="Audit Categories">
            {(defs.audit_categories || []).map((c, i) => (
              <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{c.category}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{c.description}</div>
              </div>
            ))}
          </Card>

          <Card title="Compliance Scoring" span={2}>
            <p style={{ fontSize: 12, color: '#475569', marginBottom: 10 }}>{defs.compliance_scoring?.description}</p>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={TH}>Component</th><th style={TH}>Weight</th><th style={TH}>Metric</th></tr></thead>
              <tbody>
                {(defs.compliance_scoring?.weights || []).map((w, i) => (
                  <tr key={i}>
                    <td style={{ ...TD, fontWeight: 600 }}>{w.component}</td>
                    <td style={TD}>{w.weight}</td>
                    <td style={TD}>{w.metric}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <span style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{g.term}</span>
                  <span style={{ fontSize: 12, color: '#475569' }}> — {g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{r}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
