import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const SEV_COLORS = { High: '#ef4444', Severe: '#ef4444', Critical: '#991b1b', Medium: '#f59e0b', Low: '#3b82f6', high: '#ef4444', medium: '#f59e0b', low: '#3b82f6' }
const STATUS_COLORS = { Open: '#ef4444', Mitigated: '#10b981', needs_review: '#ef4444', reviewed: '#10b981', open: '#ef4444', mitigated: '#10b981' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AIRiskDashboard() {
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
          axios.get(`${API_URL}/api/ai-risk/overview`),
          axios.get(`${API_URL}/api/ai-risk/breakdown`),
          axios.get(`${API_URL}/api/ai-risk/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI risk data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI Risk Management data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'AI Risk Management data not available.'}
    </div>
  )

  const k = overview.kpis || {}
  const riskByCat = overview.risk_by_category || []
  const sevDist = overview.severity_distribution || []
  const riskTrend = overview.risk_trend || []
  const recentEvents = overview.recent_risk_events || []
  const perPatient = breakdown?.per_patient_risks || []
  const register = breakdown?.risk_register || []
  const matrix = breakdown?.risk_matrix || []
  const mitigLog = breakdown?.mitigation_log || []
  const concepts = defs?.concepts || []
  const metrics = defs?.metrics || []
  const clinicalRel = defs?.clinical_relevance || []
  const remediation = defs?.remediation || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#ef4444' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: active ? '1px solid #dc2626' : '1px solid #cbd5e1'
  })
  const sevBadge = (sev) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${SEV_COLORS[sev] || '#94a3b8'}22`, color: SEV_COLORS[sev] || '#64748b'
  })
  const statusBadge = (st) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${STATUS_COLORS[st] || '#94a3b8'}22`, color: STATUS_COLORS[st] || '#64748b'
  })
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  const kpiItems = [
    { label: 'Total Risks', value: k.total_risks_identified, color: '#ef4444' },
    { label: 'Patients at Risk', value: k.patients_at_risk, color: '#f97316' },
    { label: 'Mitigation Rate', value: k.risk_mitigation_rate != null ? `${k.risk_mitigation_rate}%` : '--', color: '#10b981' },
    { label: 'Avg Severity', value: k.avg_assessment_severity, color: '#f59e0b' },
    { label: 'Open Risks', value: k.open_risks, color: '#dc2626' },
    { label: 'Risk Events (30d)', value: k.risk_events_30d, color: '#8b5cf6' },
    { label: 'Med Risk Flags', value: k.medication_risk_flags, color: '#ec4899' },
    { label: 'AI Low Confidence', value: k.ai_confidence_risk, color: '#06b6d4' },
  ]

  // Build 5x5 matrix grid
  const likelihoodLabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
  const impactLabels = ['Negligible', 'Low', 'Medium', 'High', 'Critical']
  const matrixGrid = {}
  for (const cell of matrix) {
    const key = `${cell.likelihood_idx}-${cell.impact_idx}`
    matrixGrid[key] = cell
  }
  const matrixColor = (score) => {
    if (score >= 16) return '#991b1b'
    if (score >= 10) return '#ef4444'
    if (score >= 6) return '#f59e0b'
    if (score >= 3) return '#3b82f6'
    return '#10b981'
  }

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>AI Risk Management Dashboard</h2>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
        {[
          ['overview', 'Overview'],
          ['register', 'Risk Register'],
          ['matrix', 'Risk Matrix'],
          ['mitigation', 'Mitigation Log'],
          ['definitions', 'Definitions'],
        ].map(([id, label]) => (
          <div key={id} style={tabStyle(tab === id)} onClick={() => setTab(id)}>{label}</div>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((ki, i) => (
              <div key={i} style={kpiStyle(ki.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{ki.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: ki.color }}>
                  {typeof ki.value === 'number' ? fmt(ki.value) : (ki.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Risk by category + severity distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk by Category</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={riskByCat.filter(d => d.value > 0)} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                    label={({ name, value }) => `${name}: ${value}`}>
                    {riskByCat.filter(d => d.value > 0).map((d, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Severity Distribution</h4>
              {sevDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={sevDist}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} name="Assessments" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No severity data available</div>
              )}
            </div>
          </div>

          {/* Risk trend */}
          {riskTrend.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Trend (Monthly)</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={riskTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="risks" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Risk Events" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Recent risk events */}
          {recentEvents.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Recent Risk Events</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>ID</th>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Action</th>
                      <th style={{ padding: '8px 10px' }}>Detail</th>
                      <th style={{ padding: '8px 10px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentEvents.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{e.id}</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{e.patient_id || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>{e.component || '--'}</td>
                        <td style={{ padding: '8px 10px' }}><span style={sevBadge('High')}>{e.action}</span></td>
                        <td style={{ padding: '8px 10px', color: '#475569', maxWidth: 300 }}>{e.detail || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{e.timestamp || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'register' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Register ({register.length} risks)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>ID</th>
                  <th style={{ padding: '8px 10px' }}>Category</th>
                  <th style={{ padding: '8px 10px' }}>Description</th>
                  <th style={{ padding: '8px 10px' }}>Severity</th>
                  <th style={{ padding: '8px 10px' }}>Likelihood</th>
                  <th style={{ padding: '8px 10px' }}>Impact</th>
                  <th style={{ padding: '8px 10px' }}>Mitigation</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Patient</th>
                </tr>
              </thead>
              <tbody>
                {register.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{r.id}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.category}</td>
                    <td style={{ padding: '8px 10px', color: '#1e293b', maxWidth: 300 }}>{r.description}</td>
                    <td style={{ padding: '8px 10px' }}><span style={sevBadge(r.severity)}>{r.severity}</span></td>
                    <td style={{ padding: '8px 10px' }}>{r.likelihood}</td>
                    <td style={{ padding: '8px 10px' }}>{r.impact}</td>
                    <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12, maxWidth: 200 }}>{r.mitigation}</td>
                    <td style={{ padding: '8px 10px' }}><span style={statusBadge(r.status)}>{r.status}</span></td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.patient_id || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'matrix' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Matrix (Likelihood x Impact)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', margin: '0 auto' }}>
                <thead>
                  <tr>
                    <th style={{ padding: 8, fontSize: 12, color: '#64748b' }}></th>
                    {impactLabels.map((imp, i) => (
                      <th key={i} style={{ padding: '8px 12px', fontSize: 11, color: '#334155', textAlign: 'center' }}>{imp}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {likelihoodLabels.slice().reverse().map((lik, li) => {
                    const likIdx = 4 - li
                    return (
                      <tr key={li}>
                        <td style={{ padding: '8px 12px', fontSize: 11, color: '#334155', fontWeight: 600, textAlign: 'right' }}>{lik}</td>
                        {impactLabels.map((imp, ii) => {
                          const cell = matrixGrid[`${likIdx}-${ii}`] || { count: 0, severity_score: 0 }
                          const bg = cell.count > 0 ? matrixColor(cell.severity_score) : '#f1f5f9'
                          const textColor = cell.count > 0 ? '#fff' : '#94a3b8'
                          return (
                            <td key={ii} style={{
                              padding: '12px 16px', textAlign: 'center', background: bg,
                              color: textColor, fontWeight: 700, fontSize: 16,
                              border: '2px solid #fff', borderRadius: 4, minWidth: 60
                            }}>
                              {cell.count || '-'}
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 14, fontSize: 11, color: '#64748b' }}>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#10b981', borderRadius: 2, marginRight: 4 }}></span>Low</span>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#3b82f6', borderRadius: 2, marginRight: 4 }}></span>Moderate</span>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#f59e0b', borderRadius: 2, marginRight: 4 }}></span>Significant</span>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#ef4444', borderRadius: 2, marginRight: 4 }}></span>High</span>
              <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#991b1b', borderRadius: 2, marginRight: 4 }}></span>Critical</span>
            </div>
          </div>

          {/* Per-patient risk scores */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Per-Patient Risk Profiles ({perPatient.length} patients)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Patient</th>
                    <th style={{ padding: '8px 10px' }}>Risk Score</th>
                    <th style={{ padding: '8px 10px' }}>Risk Factors</th>
                    <th style={{ padding: '8px 10px' }}>Assessments</th>
                    <th style={{ padding: '8px 10px' }}>Alerts</th>
                    <th style={{ padding: '8px 10px' }}>Seizures</th>
                    <th style={{ padding: '8px 10px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatient.slice(0, 30).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <div style={{
                            width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${p.risk_score}%`, height: '100%', borderRadius: 4,
                              background: p.risk_score >= 70 ? '#ef4444' : p.risk_score >= 40 ? '#f59e0b' : '#10b981'
                            }}></div>
                          </div>
                          <span style={{ fontSize: 12, fontWeight: 600 }}>{p.risk_score}</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569', maxWidth: 250 }}>
                        {(p.risk_factors || []).join(', ') || '--'}
                      </td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.assessment_count)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.alert_count)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(p.seizure_count)}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <span style={statusBadge(p.mitigation_status)}>{p.mitigation_status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {tab === 'mitigation' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Mitigation Log ({mitigLog.length} actions)</h4>
          {mitigLog.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>ID</th>
                    <th style={{ padding: '8px 10px' }}>Source</th>
                    <th style={{ padding: '8px 10px' }}>Patient</th>
                    <th style={{ padding: '8px 10px' }}>Action</th>
                    <th style={{ padding: '8px 10px' }}>Detail</th>
                    <th style={{ padding: '8px 10px' }}>AI Prediction</th>
                    <th style={{ padding: '8px 10px' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {mitigLog.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{m.id}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                          background: m.source === 'HITL Review' ? '#3b82f622' : '#10b98122',
                          color: m.source === 'HITL Review' ? '#3b82f6' : '#10b981'
                        }}>
                          {m.source}
                        </span>
                      </td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{m.patient_id || '--'}</td>
                      <td style={{ padding: '8px 10px' }}>{m.action || '--'}</td>
                      <td style={{ padding: '8px 10px', color: '#475569', maxWidth: 250 }}>{m.detail || '--'}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12 }}>{m.ai_prediction || '--'}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{m.date || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No mitigation actions recorded</div>
          )}
        </div>
      )}

      {tab === 'definitions' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Management Concepts</h4>
            {concepts.map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <strong style={{ color: '#1e293b' }}>{d.term}:</strong>{' '}
                <span style={{ color: '#475569', fontSize: 13 }}>{d.definition}</span>
              </div>
            ))}
          </div>

          {metrics.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Metrics</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Metric</th>
                    <th style={{ padding: '8px 10px' }}>Description</th>
                    <th style={{ padding: '8px 10px' }}>Formula</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{m.metric}</td>
                      <td style={{ padding: '8px 10px', color: '#475569' }}>{m.description}</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11, color: '#64748b' }}>{m.formula}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {clinicalRel.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Clinical & Regulatory Relevance</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Standard</th>
                    <th style={{ padding: '8px 10px' }}>Requirement</th>
                  </tr>
                </thead>
                <tbody>
                  {clinicalRel.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.standard}</td>
                      <td style={{ padding: '8px 10px', color: '#475569' }}>{c.requirement}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {remediation.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Remediation Strategies</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Risk Type</th>
                    <th style={{ padding: '8px 10px' }}>Strategy</th>
                  </tr>
                </thead>
                <tbody>
                  {remediation.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.risk_type}</td>
                      <td style={{ padding: '8px 10px', color: '#475569' }}>{r.strategy}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: clinical.db (assessments, seizure_diary, medications, clinical_decisions, expert_reviews, hitl_reviews, transaction_log, model_governance)
      </div>
    </div>
  )
}
