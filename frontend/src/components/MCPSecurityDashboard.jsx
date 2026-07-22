import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function RiskBadge({ level }) {
  const map = { high: '#ef4444', medium: '#f59e0b', low: '#10b981', elevated: '#ef4444', normal: '#10b981' }
  const color = map[(level || '').toLowerCase()] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{level || 'unknown'}</span>
  )
}

function ExposureBadge({ level }) {
  const map = { high: '#ef4444', elevated: '#f59e0b', moderate: '#f59e0b', low: '#10b981' }
  const color = map[(level || '').toLowerCase()] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{level || 'unknown'}</span>
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

export default function MCPSecurityDashboard() {
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
          axios.get(`${API_URL}/api/mcp-security/overview`),
          axios.get(`${API_URL}/api/mcp-security/breakdown`),
          axios.get(`${API_URL}/api/mcp-security/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load MCP security data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading MCP security data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>MCP security data not available</div>

  const tabs = ['overview', 'access', 'events', 'definitions']
  const s = overview.summary || {}
  const actors = overview.actor_privileges || []
  const attack = overview.attack_surface || []
  const daily = breakdown?.daily_security || []
  const hourly = breakdown?.hourly_pattern || []
  const patientAudit = breakdown?.patient_access_audit || []
  const recentPriv = breakdown?.recent_privileged_events || []
  const hitl = breakdown?.hitl_reviews || []
  const convRoles = breakdown?.conversation_roles || []
  const definitions = defs?.definitions || []

  // Prepare actor risk distribution for pie chart
  const riskCounts = {}
  actors.forEach(a => { riskCounts[a.risk_level] = (riskCounts[a.risk_level] || 0) + 1 })
  const riskPie = Object.entries(riskCounts).map(([name, value]) => ({ name, value }))

  // Prepare access risk distribution for pie chart
  const accessRiskCounts = {}
  patientAudit.forEach(p => { accessRiskCounts[p.access_risk] = (accessRiskCounts[p.access_risk] || 0) + 1 })
  const accessRiskPie = Object.entries(accessRiskCounts).map(([name, value]) => ({ name, value }))

  // Exposure distribution bar
  const exposureData = attack.slice(0, 10).map(c => ({
    name: c.component,
    actors: c.distinct_actors,
    transactions: c.transactions
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>MCP Security Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Security posture, guardrail enforcement, actor privileges, and access audit — {fmt(s.total_transactions)} transactions, {fmt(actors.length)} actors
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
              <KPI label="Total Transactions" value={s.total_transactions} />
              <KPI label="Guardrail Events" value={s.guardrail_events} color="#f59e0b" />
              <KPI label="Guardrail Rate" value={s.guardrail_rate_pct != null ? s.guardrail_rate_pct + '%' : '--'} />
              <KPI label="Blocked Events" value={s.blocked_events} color="#ef4444" />
            </div>
          </Card>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Sign-offs" value={s.signoff_events} color="#3b82f6" />
              <KPI label="Human Decisions" value={s.human_decisions} color="#8b5cf6" />
              <KPI label="Human Oversight Events" value={s.human_oversight_events} color="#10b981" />
              <KPI label="Oversight Rate" value={s.oversight_rate_pct != null ? s.oversight_rate_pct + '%' : '--'} />
            </div>
          </Card>

          {/* Actor Risk Distribution Pie */}
          <Card title="Actor Risk Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={riskPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {riskPie.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Attack Surface Bar */}
          <Card title="Attack Surface — Top Components by Actor Count" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={exposureData} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
                <Tooltip />
                <Bar dataKey="actors" fill="#3b82f6" name="Distinct Actors" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Actor Privileges Table */}
          <Card title="Actor Privileges" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Transactions', 'Components', 'Actions', 'Patients', 'Security Actor', 'Privileged Actions', 'Risk'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {actors.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.actor}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.transactions)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.components)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.actions)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.patients_accessed)}</td>
                      <td style={{ padding: '8px 10px' }}>{a.is_security_actor ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '8px 10px', fontSize: 11 }}>{(a.privileged_actions || []).join(', ') || '—'}</td>
                      <td style={{ padding: '8px 10px' }}><RiskBadge level={a.risk_level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Daily Security Trend */}
          <Card title="Daily Security Events" span={4}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={daily}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="total_events" stroke="#3b82f6" name="Total Events" dot={false} />
                <Line type="monotone" dataKey="blocked" stroke="#ef4444" name="Blocked" dot={false} />
                <Line type="monotone" dataKey="signoffs" stroke="#10b981" name="Sign-offs" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Access tab ── */}
      {tab === 'access' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* Patient Access Risk Pie */}
          <Card title="Patient Access Risk Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={accessRiskPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {accessRiskPie.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Hourly Pattern */}
          <Card title="Hourly Activity Pattern" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={hourly}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="events" fill="#8b5cf6" name="Events" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Patient Access Audit Table */}
          <Card title="Patient Access Audit" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient ID', 'Distinct Actors', 'Distinct Components', 'Total Events', 'Access Risk'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {patientAudit.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.distinct_actors)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.distinct_components)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.total_events)}</td>
                      <td style={{ padding: '8px 10px' }}><RiskBadge level={p.access_risk} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Conversation Roles */}
          {convRoles.length > 0 && (
            <Card title="Conversation Roles" span={2}>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={convRoles}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="role" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#06b6d4" name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* HITL Reviews */}
          {hitl.length > 0 && (
            <Card title="HITL Reviews" span={convRoles.length > 0 ? 2 : 4}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      {['Patient ID', 'Detail', 'Timestamp'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {hitl.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12 }}>{r.detail}</td>
                        <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>{r.timestamp}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Events tab ── */}
      {tab === 'events' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* Recent Privileged Events */}
          <Card title="Recent Privileged Events" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Actor', 'Action', 'Component', 'Patient', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#991b1b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recentPriv.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{e.actor}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 8,
                          background: e.action === 'blocked' ? '#fef2f2' : '#f0fdf4',
                          color: e.action === 'blocked' ? '#991b1b' : '#166534',
                          fontWeight: 600, fontSize: 11
                        }}>{e.action}</span>
                      </td>
                      <td style={{ padding: '8px 10px', fontSize: 12 }}>{e.component}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12 }}>{e.patient_id || '—'}</td>
                      <td style={{ padding: '8px 10px', fontSize: 11, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                      <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>{e.timestamp}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Attack Surface Table */}
          <Card title="Component Attack Surface" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Component', 'Distinct Actors', 'Transactions', 'Exposure'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {attack.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.component}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(c.distinct_actors)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(c.transactions)}</td>
                      <td style={{ padding: '8px 10px' }}><ExposureBadge level={c.exposure} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {definitions.map((d, i) => (
            <Card key={i}>
              <div style={{ background: '#f8fafc', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
