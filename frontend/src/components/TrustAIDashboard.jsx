import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const OK_COLOR = '#10b981'
const WARN_COLOR = '#f59e0b'
const ERR_COLOR = '#ef4444'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

function pct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function fmt(v, digits = 2) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(digits)) : String(v)
}

function trustColor(score) {
  if (score == null) return '#64748b'
  if (score >= 75) return OK_COLOR
  if (score >= 50) return WARN_COLOR
  return ERR_COLOR
}

export default function TrustAIDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/trust-ai/overview`),
      axios.get(`${API}/api/trust-ai/breakdown`),
      axios.get(`${API}/api/trust-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'concordance', label: 'Concordance & Experts' },
    { id: 'hitl', label: 'HITL & Decisions' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Trust AI...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}
  const confidenceDist = breakdown?.confidence_distribution || []
  const confidenceByLabel = breakdown?.confidence_by_label || []
  const expertByRole = breakdown?.expert_reviews_by_role || []
  const concordanceBands = breakdown?.concordance_by_confidence_band || []
  const trustTrend = breakdown?.trust_trend || []
  const hitlDecisions = breakdown?.hitl_decisions || []
  const clinicalLog = breakdown?.clinical_decision_log || []
  const defs = definitions || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI Cards */}
      <Card>
        <KPI label="Trust Score" value={fmt(kpis.trust_score, 0)} sub="composite 0-100" color={trustColor(kpis.trust_score)} />
      </Card>
      <Card>
        <KPI label="Mean Confidence" value={kpis.mean_confidence != null ? fmt(kpis.mean_confidence) : '--'} sub="AI prediction confidence" color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Total Analyses" value={fmt(kpis.total_analyses, 0)} sub="predictions made" color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Expert Reviews" value={fmt(kpis.expert_reviews, 0)} sub="expert evaluations" color="#06b6d4" />
      </Card>
      <Card>
        <KPI label="Expert Agree Rate" value={pct(kpis.expert_agree_rate)} sub="concordance" color={kpis.expert_agree_rate >= 0.8 ? OK_COLOR : WARN_COLOR} />
      </Card>
      <Card>
        <KPI label="HITL Reviews" value={fmt(kpis.hitl_reviews, 0)} sub="human-in-the-loop" color="#ec4899" />
      </Card>
      <Card>
        <KPI label="HITL Accept Rate" value={pct(kpis.hitl_accept_rate)} sub="AI accepted" color={kpis.hitl_accept_rate >= 0.7 ? OK_COLOR : WARN_COLOR} />
      </Card>
      <Card>
        <KPI label="Clinical Decisions" value={fmt(kpis.clinical_decisions, 0)} sub="final decisions logged" color="#64748b" />
      </Card>

      {/* Confidence Distribution */}
      <Card title="Confidence Distribution" span={2}>
        {confidenceDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={confidenceDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Count" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No confidence distribution data</div>}
      </Card>

      {/* Confidence by Predicted Label */}
      <Card title="Confidence by Predicted Label" span={2}>
        {confidenceByLabel.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={confidenceByLabel}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => pct(v)} />
              <Tooltip formatter={v => fmt(v)} />
              <Bar dataKey="mean_confidence" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Mean Confidence" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8' }}>No confidence by label data</div>}
      </Card>
    </div>
  )

  const renderConcordance = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Expert Reviews by Role */}
      <Card title="Expert Reviews by Role" span={2}>
        {expertByRole.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Role</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Total</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Agree</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Disagree</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Agree Rate</th>
                </tr>
              </thead>
              <tbody>
                {expertByRole.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.role || '--'}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{fmt(r.total, 0)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{fmt(r.agree, 0)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{fmt(r.disagree, 0)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                      <Badge text={pct(r.agree_rate)} color={r.agree_rate >= 0.8 ? OK_COLOR : r.agree_rate >= 0.5 ? WARN_COLOR : ERR_COLOR} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No expert review data</div>}
      </Card>

      {/* Concordance by Confidence Band */}
      <Card title="Concordance by Confidence Band" span={2}>
        {concordanceBands.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Band</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Agree</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Disagree</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Agree Rate</th>
                </tr>
              </thead>
              <tbody>
                {concordanceBands.map((b, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>
                      <Badge text={b.band || '--'} color={b.band === 'High' ? OK_COLOR : b.band === 'Mid' ? WARN_COLOR : ERR_COLOR} />
                    </td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{fmt(b.agree, 0)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{fmt(b.disagree, 0)}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{pct(b.agree_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No concordance band data</div>}
      </Card>

      {/* Trust Trend Line Chart */}
      {trustTrend.length > 0 && (
        <Card title="Trust Score Trend" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trustTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="trust_score" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Trust Score" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}
    </div>
  )

  const renderHitl = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* HITL Decisions */}
      <Card title="HITL Decisions">
        {hitlDecisions.length > 0 ? (
          <div style={{ maxHeight: 350, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Patient ID</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Decision</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Reason Code</th>
                </tr>
              </thead>
              <tbody>
                {hitlDecisions.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{d.patient_id || '--'}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                      <Badge
                        text={d.decision || '--'}
                        color={d.decision === 'accept' ? OK_COLOR : d.decision === 'override' ? ERR_COLOR : WARN_COLOR}
                      />
                    </td>
                    <td style={{ padding: '6px 8px' }}>{d.reason_code || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No HITL decision data</div>}
      </Card>

      {/* Clinical Decision Log */}
      <Card title="Clinical Decision Log">
        {clinicalLog.length > 0 ? (
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Patient ID</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>AI Prediction</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>AI Confidence</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Final Decision</th>
                  <th style={{ textAlign: 'center', padding: '6px 8px' }}>Neurologist Agreement</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Reviewer</th>
                </tr>
              </thead>
              <tbody>
                {clinicalLog.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 11 }}>{c.patient_id || '--'}</td>
                    <td style={{ padding: '6px 8px' }}>{c.ai_prediction || '--'}</td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>{c.ai_confidence != null ? fmt(c.ai_confidence) : '--'}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <Badge
                        text={c.final_decision || '--'}
                        color={c.final_decision === c.ai_prediction ? OK_COLOR : WARN_COLOR}
                      />
                    </td>
                    <td style={{ textAlign: 'center', padding: '6px 8px' }}>
                      {c.neurologist_agreement != null ? (
                        <Badge
                          text={c.neurologist_agreement ? 'Yes' : 'No'}
                          color={c.neurologist_agreement ? OK_COLOR : ERR_COLOR}
                        />
                      ) : '--'}
                    </td>
                    <td style={{ padding: '6px 8px' }}>{c.reviewer || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No clinical decision data</div>}
      </Card>
    </div>
  )

  const renderDefinitions = () => {
    const sections = defs.sections || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {sections.map((sec, i) => (
          <Card key={i} title={sec.name}>
            {sec.description && (
              <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: '0 0 10px' }}>{sec.description}</p>
            )}
            {sec.fields && (
              <ul style={{ fontSize: 13, color: '#475569', lineHeight: 1.8, margin: 0, paddingLeft: 20 }}>
                {sec.fields.map((f, j) => (
                  <li key={j}><strong>{f.name}:</strong> {f.description}</li>
                ))}
              </ul>
            )}
            {sec.items && (
              <ul style={{ fontSize: 13, color: '#475569', lineHeight: 1.8, margin: 0, paddingLeft: 20 }}>
                {sec.items.map((item, j) => (
                  <li key={j}>{typeof item === 'string' ? item : <><strong>{item.name || item.term}:</strong> {item.description || item.definition}</>}</li>
                ))}
              </ul>
            )}
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Trust AI Dashboard</h2>
        <p style={{ margin: 0, fontSize: 13, color: '#64748b' }}>
          AI confidence calibration, expert concordance, human-in-the-loop decisions, clinical trust metrics
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'concordance' && renderConcordance()}
      {tab === 'hitl' && renderHitl()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
