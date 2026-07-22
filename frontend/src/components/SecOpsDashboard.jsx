import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#f59e0b',
  medium: '#3b82f6',
  low: '#64748b'
}

const RISK_COLORS = {
  elevated: '#ef4444',
  normal: '#10b981',
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#64748b'
}

const OWASP_STATUS_COLORS = {
  covered: '#10b981',
  partial: '#f59e0b',
  planned: '#3b82f6',
  gap: '#ef4444'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function SeverityBadge({ severity }) {
  const color = SEVERITY_COLORS[severity] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(severity || '').replace(/_/g, ' ')}</span>
  )
}

function RiskBadge({ risk }) {
  const color = RISK_COLORS[risk] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(risk || '').replace(/_/g, ' ')}</span>
  )
}

function OwaspBadge({ status }) {
  const color = OWASP_STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(status || '').replace(/_/g, ' ')}</span>
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

export default function SecOpsDashboard() {
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
          axios.get(`${API_URL}/api/sec-ops/overview`),
          axios.get(`${API_URL}/api/sec-ops/breakdown`),
          axios.get(`${API_URL}/api/sec-ops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load SecOps data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading SecOps data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>SecOps data not available</div>

  const tabs = ['overview', 'access', 'threats', 'owasp', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>SecOps Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Security operations — {fmt(kpis.total_transactions)} transactions, {fmt(kpis.total_conversations)} conversations, compliance score {fmt(kpis.compliance_score)}%
      </p>

      {/* Tab bar */}
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

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Transactions" value={kpis.total_transactions} />
              <KPI label="Conversations" value={kpis.total_conversations} />
              <KPI label="Threats Detected" value={kpis.total_threats_detected} color={kpis.total_threats_detected > 0 ? '#ef4444' : '#10b981'} />
              <KPI label="Compliance Score" value={kpis.compliance_score} sub="%" color={kpis.compliance_score >= 80 ? '#10b981' : '#f59e0b'} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Guardrail Events" value={kpis.guardrail_events} />
              <KPI label="Blocked" value={kpis.blocked_events} color="#ef4444" />
              <KPI label="Total Actors" value={kpis.total_actors} sub={`${fmt(kpis.privileged_actors)} privileged`} />
              <KPI label="Oversight Rate" value={kpis.oversight_rate_pct} sub="%" />
            </div>
          </Card>

          <Card title="PII Pattern Inventory" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.pii_inventory || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="detections" name="Detections" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Injection Pattern Inventory" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.injection_inventory || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="pattern_label" type="category" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="detections" name="Detections" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Daily Security Trend" span={4}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={(breakdown?.daily_security || []).slice(-30)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} name="Total" dot={false} />
                <Line type="monotone" dataKey="blocked" stroke="#ef4444" strokeWidth={2} name="Blocked" dot={false} />
                <Line type="monotone" dataKey="signoffs" stroke="#10b981" strokeWidth={2} name="Sign-offs" dot={false} />
                <Line type="monotone" dataKey="mutations" stroke="#f59e0b" strokeWidth={2} name="Mutations" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Security Agent Events" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Action', 'Component', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.security_agent_events || []).map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{e.actor}</td>
                      <td style={{ padding: '8px 12px' }}><SeverityBadge severity={e.action === 'blocked' ? 'critical' : 'medium'} /></td>
                      <td style={{ padding: '8px 12px' }}>{e.component}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{e.timestamp?.slice(0, 16).replace('T', ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Guardrail Log" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Action', 'Component', 'Patient', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.guardrail_log || []).map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{e.actor}</td>
                      <td style={{ padding: '8px 12px' }}>{e.action?.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px' }}>{e.component}</td>
                      <td style={{ padding: '8px 12px' }}>{e.patient_id}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{e.timestamp?.slice(0, 16).replace('T', ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Access tab ── */}
      {tab === 'access' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Actor Privilege Matrix" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Transactions', 'Components', 'Actions', 'Patients', 'Privileged', 'Risk'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.actor_privilege_matrix || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{a.actor}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.transactions)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.components)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.distinct_actions)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.patients_accessed)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.privileged_actions)}</td>
                      <td style={{ padding: '8px 12px' }}><RiskBadge risk={a.risk_level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Attack Surface — Component Exposure" span={4}>
            <ResponsiveContainer width="100%" height={Math.max(280, (breakdown?.attack_surface || []).length * 28)}>
              <BarChart data={(breakdown?.attack_surface || []).slice(0, 15)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="component" type="category" tick={{ fontSize: 11 }} width={160} />
                <Tooltip />
                <Bar dataKey="distinct_actors" name="Distinct Actors" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Patient Access Audit" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Actors', 'Components', 'Events', 'Risk'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.patient_access_audit || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.distinct_actors)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.distinct_components)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(p.total_events)}</td>
                      <td style={{ padding: '8px 12px' }}><RiskBadge risk={p.risk} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Action Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(breakdown?.action_distribution || []).slice(0, 12)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="action" type="category" tick={{ fontSize: 11 }} width={130} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Conversation Roles" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={breakdown?.conversation_roles || []} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={100} label={({ role, count }) => `${role}: ${count}`}>
                  {(breakdown?.conversation_roles || []).map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Threats tab ── */}
      {tab === 'threats' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Threats Detected" value={kpis.total_threats_detected} color="#ef4444" />
              <KPI label="Critical" value={kpis.critical_threats} color={kpis.critical_threats > 0 ? '#ef4444' : '#10b981'} />
              <KPI label="High" value={kpis.high_threats} color="#f59e0b" />
              <KPI label="PII Patterns" value={kpis.pii_patterns_active} sub="active" />
            </div>
          </Card>

          <Card title="PII Inventory" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Pattern', 'Severity', 'Regulation', 'Detections'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#991b1b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.pii_inventory || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.name}</td>
                      <td style={{ padding: '8px 12px' }}><SeverityBadge severity={p.severity} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{p.regulation}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: p.detections > 0 ? '#ef4444' : '#10b981' }}>{fmt(p.detections)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Injection Patterns" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Pattern', 'Severity', 'Detections'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#991b1b', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.injection_inventory || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.pattern_label?.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px' }}><SeverityBadge severity={p.severity} /></td>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: p.detections > 0 ? '#ef4444' : '#10b981' }}>{fmt(p.detections)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Incident Timeline" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Action', 'Component', 'Patient', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.incident_timeline || []).slice(0, 20).map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{e.actor}</td>
                      <td style={{ padding: '8px 12px' }}>{e.action?.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px' }}>{e.component}</td>
                      <td style={{ padding: '8px 12px' }}>{e.patient_id}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{e.timestamp?.slice(0, 16).replace('T', ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── OWASP tab ── */}
      {tab === 'owasp' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="OWASP LLM Top-10 Coverage" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['ID', 'Name', 'Status', 'Controls'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.owasp_coverage || []).map((o, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{o.id}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{o.name}</td>
                      <td style={{ padding: '8px 12px' }}><OwaspBadge status={o.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{o.controls}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Coverage Summary" span={4}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={breakdown?.owasp_coverage || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="id" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} tickFormatter={v => v === 1 ? 'Covered' : v === 0.5 ? 'Partial' : 'Gap'} />
                <Tooltip formatter={(v) => v === 1 ? 'Covered' : v === 0.5 ? 'Partial' : 'Gap'} />
                <Bar dataKey="id" name="Status">
                  {(breakdown?.owasp_coverage || []).map((o, i) => (
                    <Cell key={i} fill={OWASP_STATUS_COLORS[o.status] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8 }}>
              {Object.entries(OWASP_STATUS_COLORS).map(([label, color]) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
                  <div style={{ width: 12, height: 12, borderRadius: 2, background: color }} />
                  <span style={{ color: '#64748b', textTransform: 'capitalize' }}>{label}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {(defs?.sections || []).map((sec, si) => (
            <Card key={si} title={sec.title} span={4}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}

          <Card title="Data Sources" span={4}>
            <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>
              <p style={{ margin: '0 0 8px' }}><strong>transaction_log</strong> — {fmt(kpis.total_transactions)} rows — all system transactions (predictions, ingestions, sign-offs, guardrail events)</p>
              <p style={{ margin: '0 0 8px' }}><strong>conversation_log</strong> — {fmt(kpis.total_conversations)} rows — all AI agent conversations scanned for injection/PII patterns</p>
              <p style={{ margin: 0 }}><strong>Real-time</strong> — generated from live database queries, not cached snapshots</p>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
