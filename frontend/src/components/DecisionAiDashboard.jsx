import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const ROUTE_COLORS = { auto_approve: '#10b981', review: '#f59e0b', escalate: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function RouteBadge({ route }) {
  const color = ROUTE_COLORS[route] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(route || '').replace(/_/g, ' ')}</span>
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function DecisionAiDashboard() {
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
          axios.get(`${API_URL}/decision-ai/overview`),
          axios.get(`${API_URL}/decision-ai/breakdown`),
          axios.get(`${API_URL}/decision-ai/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load decision AI data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#129302;</div>
      Loading decision AI data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No decision AI data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analyses', label: 'Per-Analysis' },
    { id: 'hitl', label: 'HITL Reviews' },
    { id: 'calibration', label: 'Calibration' }
  ]

  const kpi = overview.kpis || {}
  const routeDist = overview.route_distribution || []
  const confHist = overview.confidence_histogram || []
  const thresholds = overview.thresholds || {}
  const diseaseSummary = overview.disease_summary || []
  const auditSummary = overview.audit_summary || {}
  const analyses = (breakdown && breakdown.per_analysis) || []
  const hitlReviews = (breakdown && breakdown.hitl_reviews) || []
  const calibration = (breakdown && breakdown.calibration) || []
  const auditTimeline = (breakdown && breakdown.audit_timeline) || []

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Decision AI Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(kpi.total_analyses)} analyses | avg confidence {fmtPct(kpi.avg_confidence)} | {fmt(kpi.audit_events)} audit events
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Analyses" value={fmt(kpi.total_analyses)} color="#3b82f6" /></Card>
            <Card><KPI label="Avg Confidence" value={fmtPct(kpi.avg_confidence)} color="#8b5cf6" /></Card>
            <Card><KPI label="Auto-Approved" value={fmt(kpi.auto_approve_count)} color="#10b981" /></Card>
            <Card><KPI label="Review" value={fmt(kpi.review_count)} sub="needs clinician" color="#f59e0b" /></Card>
            <Card><KPI label="Escalated" value={fmt(kpi.escalate_count)} sub="low confidence" color="#ef4444" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Routing Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={routeDist} dataKey="count" nameKey="route" cx="50%" cy="50%" outerRadius={80} label={({ route, count }) => `${route}: ${count}`}>
                    {routeDist.map((d, i) => <Cell key={i} fill={ROUTE_COLORS[d.route] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', fontSize: 12, color: '#64748b', marginTop: 4 }}>
                Thresholds: auto &ge; {fmtPct(thresholds.auto_approve)} | review &ge; {fmtPct(thresholds.review)} | escalate &lt; {fmtPct(thresholds.review)}
              </div>
            </Card>
            <Card title="Confidence Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={confHist} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Analyses" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Disease Summary">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px' }}>Disease</th>
                      <th style={{ padding: '6px 10px' }}>Count</th>
                      <th style={{ padding: '6px 10px' }}>Avg Conf</th>
                      <th style={{ padding: '6px 10px' }}>Routes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diseaseSummary.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600, textTransform: 'capitalize' }}>{d.disease}</td>
                        <td style={{ padding: '6px 10px' }}>{d.count}</td>
                        <td style={{ padding: '6px 10px' }}>{fmtPct(d.avg_confidence)}</td>
                        <td style={{ padding: '6px 10px' }}>
                          {Object.entries(d.routes || {}).map(([r, c]) => (
                            <span key={r} style={{ marginRight: 8 }}><RouteBadge route={r} /> {c}</span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card title="Audit Summary">
              <div style={{ fontSize: 13, color: '#334155' }}>
                <div style={{ marginBottom: 8 }}>Total events: <strong>{fmt(auditSummary.total_events)}</strong> across <strong>{auditSummary.components}</strong> components</div>
                <div style={{ marginBottom: 4, fontWeight: 600, color: '#64748b' }}>Top Components:</div>
                {(auditSummary.top_components || []).slice(0, 8).map((c, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0', borderBottom: '1px solid #f8fafc' }}>
                    <span>{c.name}</span>
                    <span style={{ color: '#64748b' }}>{fmt(c.count)}</span>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 12, fontSize: 12, color: '#64748b' }}>
                HITL: {fmt(kpi.hitl_overrides)} overrides, {fmt(kpi.hitl_confirms)} confirms
              </div>
            </Card>
          </div>
        </>
      )}

      {tab === 'analyses' && (
        <Card title={`Per-Analysis Detail (${analyses.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>ID</th>
                  <th style={{ padding: '8px 10px' }}>Patient</th>
                  <th style={{ padding: '8px 10px' }}>Disease</th>
                  <th style={{ padding: '8px 10px' }}>Prediction</th>
                  <th style={{ padding: '8px 10px' }}>Confidence</th>
                  <th style={{ padding: '8px 10px' }}>Route</th>
                  <th style={{ padding: '8px 10px' }}>Signal</th>
                  <th style={{ padding: '8px 10px' }}>Class Probabilities</th>
                </tr>
              </thead>
              <tbody>
                {analyses.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px' }}>{a.id}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.patient_id}</td>
                    <td style={{ padding: '8px 10px', textTransform: 'capitalize' }}>{a.disease}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.predicted_label}</td>
                    <td style={{ padding: '8px 10px' }}>{fmtPct(a.confidence)}</td>
                    <td style={{ padding: '8px 10px' }}><RouteBadge route={a.route} /></td>
                    <td style={{ padding: '8px 10px', color: a.signal_quality === 'Good' ? '#10b981' : '#f59e0b' }}>{a.signal_quality}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>
                      {Object.entries(a.class_probs || {}).map(([c, p]) => `${c}: ${fmtPct(p)}`).join(' | ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'hitl' && (
        <>
          <Card title={`Human-in-the-Loop Reviews (${hitlReviews.length})`}>
            {hitlReviews.length === 0 ? (
              <div style={{ color: '#64748b', fontSize: 13 }}>No HITL reviews recorded yet.</div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px' }}>Patient</th>
                      <th style={{ padding: '8px 12px' }}>Analysis</th>
                      <th style={{ padding: '8px 12px' }}>Decision</th>
                      <th style={{ padding: '8px 12px' }}>AI Prediction</th>
                      <th style={{ padding: '8px 12px' }}>Human Decision</th>
                      <th style={{ padding: '8px 12px' }}>Reason</th>
                      <th style={{ padding: '8px 12px' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hitlReviews.map((h, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{h.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>#{h.analysis_id}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{
                            padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600,
                            background: h.decision === 'override' ? '#ef444422' : '#10b98122',
                            color: h.decision === 'override' ? '#ef4444' : '#10b981'
                          }}>{h.decision}</span>
                        </td>
                        <td style={{ padding: '8px 12px' }}>{h.ai_prediction}</td>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{h.human_decision}</td>
                        <td style={{ padding: '8px 12px', color: '#64748b' }}>{h.reason_code}</td>
                        <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{(h.created_at || '').slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}

      {tab === 'calibration' && (
        <>
          <Card title="Confidence Calibration">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={calibration} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="total" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Total" />
                <Bar dataKey="reviewed" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Reviewed" />
                <Bar dataKey="overridden" fill="#ef4444" radius={[4, 4, 0, 0]} name="Overridden" />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ overflowX: 'auto', marginTop: 16 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '6px 10px' }}>Bucket</th>
                    <th style={{ padding: '6px 10px' }}>Total</th>
                    <th style={{ padding: '6px 10px' }}>Reviewed</th>
                    <th style={{ padding: '6px 10px' }}>Overridden</th>
                    <th style={{ padding: '6px 10px' }}>Agreement</th>
                  </tr>
                </thead>
                <tbody>
                  {calibration.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.bucket}</td>
                      <td style={{ padding: '6px 10px' }}>{c.total}</td>
                      <td style={{ padding: '6px 10px' }}>{c.reviewed}</td>
                      <td style={{ padding: '6px 10px', color: c.overridden > 0 ? '#ef4444' : undefined }}>{c.overridden}</td>
                      <td style={{ padding: '6px 10px' }}>{c.agreement_rate != null ? fmtPct(c.agreement_rate) : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {auditTimeline.length > 0 && (
            <Card title="Audit Event Timeline" span={2}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px' }}>Month</th>
                      {Object.keys(auditTimeline[0] || {}).filter(k => k !== 'month').map(k => (
                        <th key={k} style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {auditTimeline.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{row.month}</td>
                        {Object.entries(row).filter(([k]) => k !== 'month').map(([k, v]) => (
                          <td key={k} style={{ padding: '6px 10px' }}>{fmt(v)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      {defs && (
        <div style={{ marginTop: 20, padding: 16, background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
          <strong>{defs.title}</strong> — {(defs.sections || []).map(s => s.title || s.name).join(', ')}
        </div>
      )}
    </div>
  )
}
