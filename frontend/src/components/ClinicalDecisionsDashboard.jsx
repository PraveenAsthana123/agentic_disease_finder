import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const AGREEMENT_COLORS = {
  Agree: '#22c55e',
  Partial: '#f59e0b',
  Disagree: '#ef4444'
}

const DECISION_COLORS = {
  Confirm: '#22c55e',
  Override: '#ef4444',
  Defer: '#f59e0b',
  Escalate: '#8b5cf6'
}

const ARTIFACT_COLORS = {
  None: '#22c55e',
  Low: '#3b82f6',
  Medium: '#f59e0b',
  High: '#ef4444'
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

function AgreementBadge({ agreement }) {
  const color = AGREEMENT_COLORS[agreement] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{agreement || 'unknown'}</span>
  )
}

function DecisionBadge({ decision }) {
  const color = DECISION_COLORS[decision] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{decision || 'unknown'}</span>
  )
}

function ArtifactBadge({ risk }) {
  const color = ARTIFACT_COLORS[risk] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{risk || 'unknown'}</span>
  )
}

export default function ClinicalDecisionsDashboard() {
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
          axios.get(`${API_URL}/api/clinical-decisions/overview`),
          axios.get(`${API_URL}/api/clinical-decisions/breakdown`),
          axios.get(`${API_URL}/api/clinical-decisions/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Clinical Decisions data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No clinical decisions data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const agreementDistData = overview?.agreement_distribution
    ? Object.entries(overview.agreement_distribution).map(([k, v]) => ({
        name: k, value: v, color: AGREEMENT_COLORS[k] || '#94a3b8'
      }))
    : []

  const decisionDistData = overview?.decision_distribution
    ? Object.entries(overview.decision_distribution).map(([k, v]) => ({
        name: k, count: v, color: DECISION_COLORS[k] || '#94a3b8'
      }))
    : []

  const timelineData = overview?.monthly_decisions || []
  const confidenceData = overview?.confidence_distribution || []

  /* Breakdown data prep */
  const reviewerWorkload = breakdown?.reviewer_workload || []
  const disagreementCases = breakdown?.disagreement_cases || []
  const reviewerPerformance = breakdown?.reviewer_performance || []
  const artifactVsDisagreement = breakdown?.artifact_vs_disagreement || []
  const recentDecisions = breakdown?.recent_decisions || []
  const predictionCross = breakdown?.prediction_decision_cross || {}

  // Cross-tab data for table
  const predictions = Object.keys(predictionCross)
  const allDecisionTypes = ['Confirm', 'Override', 'Defer', 'Escalate']

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Clinical Decisions Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Human-in-the-Loop AI oversight — neurologist confirm/override of AI predictions
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Decisions" value={fmt(overview.total_decisions)} />
              <KPI label="Agreement Rate" value={fmt(overview.agreement_rate_pct) + '%'} color="#22c55e" />
              <KPI label="Override Rate" value={fmt(overview.override_rate_pct) + '%'} color="#ef4444" />
              <KPI label="Avg AI Confidence" value={overview.avg_confidence} color="#3b82f6" />
              <KPI label="Reviewers" value={fmt(overview.unique_reviewers)} color="#8b5cf6" />
              <KPI label="Patients" value={fmt(overview.unique_patients)} color="#06b6d4" />
            </div>
          </Card>

          {/* Pie: Agreement Distribution */}
          <Card title="Agreement Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={agreementDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {agreementDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {agreementDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Decision Distribution */}
          <Card title="Final Decision Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={decisionDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Count">
                  {decisionDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: Monthly Timeline */}
          <Card title="Monthly Decision Volume" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="decisions" stroke="#3b82f6" name="Decisions" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar: Confidence Distribution */}
          <Card title="AI Confidence Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={confidenceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" name="Cases" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Breakdown */}
      {tab === 'breakdown' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Reviewer Workload Table */}
          <Card title="Reviewer Workload" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Reviewer</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Total</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Agrees</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Disagrees</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Overrides</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Avg Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {reviewerWorkload.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 500 }}>{r.reviewer}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.total}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#22c55e' }}>{r.agrees}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#ef4444' }}>{r.disagrees}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#f59e0b' }}>{r.overrides}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.avg_confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Prediction x Decision Cross-Tab */}
          <Card title="Prediction x Decision Cross-Tab" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>AI Prediction</th>
                    {allDecisionTypes.map(d => (
                      <th key={d} style={{ padding: '8px 10px', textAlign: 'center', color: DECISION_COLORS[d] }}>{d}</th>
                    ))}
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((pred, i) => {
                    const row = predictionCross[pred] || {}
                    const total = allDecisionTypes.reduce((s, d) => s + (row[d] || 0), 0)
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 500 }}>{pred}</td>
                        {allDecisionTypes.map(d => (
                          <td key={d} style={{ padding: '8px 10px', textAlign: 'center' }}>{row[d] || 0}</td>
                        ))}
                        <td style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600 }}>{total}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Artifact Risk vs Disagreement */}
          <Card title="Artifact Risk vs Disagreement">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Artifact Risk</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Total</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Disagree</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {artifactVsDisagreement.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px' }}><ArtifactBadge risk={r.artifact_risk} /></td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.total}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>{r.disagree_count}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: r.disagree_rate > 30 ? '#ef4444' : '#64748b' }}>{r.disagree_rate}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Reviewer Performance */}
          <Card title="Reviewer Performance">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Reviewer</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Agree %</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Override %</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Escalate %</th>
                  </tr>
                </thead>
                <tbody>
                  {reviewerPerformance.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 500 }}>{r.reviewer}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#22c55e' }}>{r.agree_rate}%</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#ef4444' }}>{r.override_rate}%</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', color: '#8b5cf6' }}>{r.escalate_rate}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Disagreement Cases (red header) */}
          <Card title="" span={2}>
            <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12, marginBottom: 12 }}>
              <h4 style={{ margin: 0, color: '#dc2626', fontSize: 14 }}>High Disagreement Cases ({disagreementCases.length})</h4>
              <p style={{ margin: '4px 0 0', fontSize: 11, color: '#991b1b' }}>Cases where neurologist fully disagreed with AI prediction</p>
            </div>
            <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#fef2f2', borderBottom: '1px solid #fecaca' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>AI Prediction</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Confidence</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Channels</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Artifact</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Decision</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Reviewer</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {disagreementCases.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 500 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{r.ai_prediction}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: r.ai_confidence < 0.6 ? '#ef4444' : '#64748b' }}>{r.ai_confidence}</td>
                      <td style={{ padding: '6px 8px', fontSize: 10 }}>{r.top_channels}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ArtifactBadge risk={r.artifact_risk} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><DecisionBadge decision={r.final_decision} /></td>
                      <td style={{ padding: '6px 8px' }}>{r.reviewer}</td>
                      <td style={{ padding: '6px 8px', fontSize: 10 }}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Recent Decisions */}
          <Card title="Recent Decisions" span={2}>
            <div style={{ overflowX: 'auto', maxHeight: 350, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>AI Prediction</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Confidence</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Channels</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Artifact</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Agreement</th>
                    <th style={{ padding: '6px 8px', textAlign: 'center' }}>Decision</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Reviewer</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {recentDecisions.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 500 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{r.ai_prediction}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.ai_confidence}</td>
                      <td style={{ padding: '6px 8px', fontSize: 10 }}>{r.top_channels}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ArtifactBadge risk={r.artifact_risk} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><AgreementBadge agreement={r.neurologist_agreement} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><DecisionBadge decision={r.final_decision} /></td>
                      <td style={{ padding: '6px 8px' }}>{r.reviewer}</td>
                      <td style={{ padding: '6px 8px', fontSize: 10 }}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Decision Types */}
          <Card title="Decision Types">
            {(defs.decision_types || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <DecisionBadge decision={d.type} />
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{d.description}</span>
              </div>
            ))}
          </Card>

          {/* Agreement Levels */}
          <Card title="Agreement Levels">
            {(defs.agreement_levels || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <AgreementBadge agreement={d.level} />
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{d.description}</span>
              </div>
            ))}
          </Card>

          {/* Artifact Risk Levels */}
          <Card title="Artifact Risk Levels">
            {(defs.artifact_risk_levels || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <ArtifactBadge risk={d.level} />
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{d.description}</span>
              </div>
            ))}
          </Card>

          {/* AI Prediction Categories */}
          <Card title="AI Prediction Categories">
            {(defs.ai_prediction_categories || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{d.category}</span>
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{d.description}</span>
              </div>
            ))}
          </Card>

          {/* EEG Channel Descriptions */}
          <Card title="EEG Channel Descriptions">
            {(defs.eeg_channel_descriptions || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b', fontFamily: 'monospace' }}>{d.channel}</span>
                <span style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>{d.description}</span>
              </div>
            ))}
          </Card>

          {/* Clinical Notes */}
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
              {(defs.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          </Card>

          {/* Glossary */}
          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{g.term}</span>
                  <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{g.definition}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
