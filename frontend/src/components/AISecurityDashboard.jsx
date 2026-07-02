import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316', '#14b8a6']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function RiskBadge({ level }) {
  const colors = {
    'HIGH': { bg: '#fef2f2', fg: '#991b1b' },
    'MEDIUM': { bg: '#fff7ed', fg: '#9a3412' },
    'LOW': { bg: '#ecfdf5', fg: '#065f46' },
  }
  const c = colors[level] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.fg
    }}>{level}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'phi', label: 'PHI Access' },
  { id: 'actors', label: 'Actor Profiles' },
  { id: 'threats', label: 'Threats & Anomalies' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AISecurityDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/ai-security/overview`),
      axios.get(`${API_URL}/ai-security/breakdown`),
      axios.get(`${API_URL}/ai-security/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading AI Security...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40 }}>Security data not available</div>

  const k = overview.kpis || {}

  const riskData = (overview.risk_distribution || []).map((r, i) => ({
    ...r, name: r.level, value: r.count,
    fill: r.level === 'HIGH' ? '#ef4444' : r.level === 'MEDIUM' ? '#f59e0b' : '#10b981'
  }))

  const actorCompData = (overview.actor_component_coverage || []).map((a, i) => ({
    ...a, fill: COLORS[i % COLORS.length]
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        AI Security Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Security posture, PHI access tracking, risk classification, and audit trail from transaction_log ({k.total_events} events)
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}
          >{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Events" value={k.total_events} color="#3b82f6" /></Card>
            <Card><KPI label="Security Events" value={k.security_events} sub="security + compliance agent" color="#ef4444" /></Card>
            <Card><KPI label="Distinct Actors" value={k.distinct_actors} color="#8b5cf6" /></Card>
            <Card><KPI label="Components" value={k.distinct_components} color="#10b981" /></Card>
            <Card><KPI label="HITL Reviews" value={k.hitl_review_count} sub="human-in-the-loop" color="#f59e0b" /></Card>
            <Card><KPI label="PHI Accesses" value={k.phi_access_count} sub={`${k.phi_access_rate_pct}% of total`} color="#ec4899" /></Card>
            <Card><KPI label="Human Oversight" value={`${k.human_oversight_rate_pct}%`} sub="sign-off + decisions" color="#06b6d4" /></Card>
            <Card><KPI label="Risk Score" value={k.phi_access_rate_pct > 10 ? 'Elevated' : 'Normal'} color={k.phi_access_rate_pct > 10 ? '#ef4444' : '#10b981'} /></Card>
          </div>

          {/* Risk distribution + actor coverage */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Risk Level Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={riskData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                    label={({ name, value }) => `${name}: ${value}`}>
                    {riskData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Actor-Component Coverage">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={actorCompData} layout="vertical" margin={{ left: 120, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="actor" fontSize={11} width={120} />
                  <Tooltip />
                  <Bar dataKey="components" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Components Accessed" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Daily security trend */}
          <Card title="Daily Security Event Trend">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={overview.daily_trend || []} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={11} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="events" stroke="#ef4444" fill="#fef2f2" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          {/* Hourly access pattern */}
          <div style={{ marginTop: 16 }}>
            <Card title="Hourly Access Pattern">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.hourly_pattern || []} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" fontSize={11} label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* PHI access by actor */}
          <div style={{ marginTop: 16 }}>
            <Card title="PHI Access by Actor">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={overview.phi_access_by_actor || []} margin={{ left: 120, right: 20 }} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="actor" fontSize={11} width={120} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ec4899" radius={[0, 4, 4, 0]} name="PHI Accesses" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── PHI Access Tab ── */}
      {tab === 'phi' && breakdown && (
        <>
          <Card title={`PHI Access Log (last ${(breakdown.phi_access_log || []).length} events on sensitive components)`}>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['#', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.phi_access_log || []).map(r => (
                    <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{r.id}</td>
                      <td>{r.patient_id}</td>
                      <td style={{ fontWeight: 600, color: '#ec4899' }}>{r.component}</td>
                      <td>{r.action}</td>
                      <td>{r.actor}</td>
                      <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>{r.detail}</td>
                      <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 19)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Governance events */}
          <div style={{ marginTop: 16 }}>
            <Card title={`Governance Events (${(breakdown.governance_events || []).length} sign-off / decision / block actions)`}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['#', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.governance_events || []).map(r => (
                      <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{r.id}</td>
                        <td>{r.patient_id}</td>
                        <td style={{ fontWeight: 600 }}>{r.component}</td>
                        <td><RiskBadge level="HIGH" /></td>
                        <td>{r.actor}</td>
                        <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>{r.detail}</td>
                        <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Actor Profiles Tab ── */}
      {tab === 'actors' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
            {(breakdown.actor_profiles || []).map(ap => (
              <Card key={ap.actor} title={`${ap.actor} (${ap.total_events} events)`}>
                <div style={{ fontSize: 12, lineHeight: 2 }}>
                  <div style={{ marginBottom: 8 }}>
                    <strong>Components:</strong>{' '}
                    {(ap.top_components || []).map(c => (
                      <span key={c.component} style={{
                        display: 'inline-block', padding: '1px 8px', borderRadius: 8,
                        background: '#f1f5f9', margin: '0 4px 4px 0', fontSize: 11
                      }}>{c.component} ({c.count})</span>
                    ))}
                  </div>
                  <div>
                    <strong>Actions:</strong>{' '}
                    {(ap.top_actions || []).map(a => (
                      <span key={a.action} style={{
                        display: 'inline-block', padding: '1px 8px', borderRadius: 8,
                        background: '#ecfdf5', margin: '0 4px 4px 0', fontSize: 11
                      }}>{a.action} ({a.count})</span>
                    ))}
                  </div>
                </div>
              </Card>
            ))}
          </div>

          {/* Actor-component access matrix */}
          {overview?.actor_component_matrix && (
            <div style={{ marginTop: 16 }}>
              <Card title="Actor-Component Access Matrix">
                <div style={{ overflowX: 'auto', marginTop: 8 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                        <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Actor</th>
                        {(overview.actor_component_matrix.components || []).map(c => (
                          <th key={c} style={{ textAlign: 'center', padding: '8px 4px', color: '#64748b', fontSize: 10, writingMode: 'vertical-lr' }}>{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(overview.actor_component_matrix.matrix || []).map(row => (
                        <tr key={row.actor} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px', fontWeight: 600 }}>{row.actor}</td>
                          {(overview.actor_component_matrix.components || []).map(c => {
                            const val = row.access?.[c] || 0
                            return (
                              <td key={c} style={{
                                textAlign: 'center', padding: '4px',
                                background: val > 0 ? `rgba(239,68,68,${Math.min(val / 50, 0.8)})` : 'transparent',
                                color: val > 20 ? '#fff' : '#1e293b', fontWeight: val > 0 ? 600 : 400
                              }}>{val || ''}</td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}
        </>
      )}

      {/* ── Threats & Anomalies Tab ── */}
      {tab === 'threats' && breakdown && (
        <>
          {/* Anomaly indicators */}
          <Card title="Anomaly Indicators">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
              {breakdown.anomaly_indicators && (
                <>
                  <div style={{ padding: 16, background: '#fef2f2', borderRadius: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#991b1b', marginBottom: 4 }}>Off-Hours Activity</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{breakdown.anomaly_indicators.off_hours_events}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{breakdown.anomaly_indicators.off_hours_pct}% of all events outside 8am-6pm</div>
                  </div>
                  <div style={{ padding: 16, background: '#fff7ed', borderRadius: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#9a3412', marginBottom: 4 }}>Wide-Access Actors</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{breakdown.anomaly_indicators.wide_access_actors}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>actors accessing &gt;15 components</div>
                  </div>
                </>
              )}
            </div>
          </Card>

          {/* High-risk action log */}
          <div style={{ marginTop: 16 }}>
            <Card title={`High-Risk Action Log (${(breakdown.high_risk_log || []).length} events: delete, sign-off, human_decision)`}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['#', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                        <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.high_risk_log || []).map(r => (
                      <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{r.id}</td>
                        <td>{r.patient_id}</td>
                        <td style={{ fontWeight: 600 }}>{r.component}</td>
                        <td><RiskBadge level="HIGH" /></td>
                        <td>{r.actor}</td>
                        <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>{r.detail}</td>
                        <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 19)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {Object.entries(defs.definitions).map(([section, items]) => (
            <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                {Object.entries(items).map(([term, desc]) => (
                  <div key={term} style={{ marginBottom: 8 }}>
                    <strong style={{ color: '#1e293b' }}>{term}:</strong>{' '}
                    <span style={{ color: '#475569' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
