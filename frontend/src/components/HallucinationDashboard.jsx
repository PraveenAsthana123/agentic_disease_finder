import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#8bc34a']

const SEVERITY_COLORS = {
  critical: '#dc2626',
  high: '#f59e0b',
  medium: '#3b82f6',
  low: '#10b981',
}

const RISK_COLORS = {
  low: '#10b981',
  moderate: '#3b82f6',
  elevated: '#f59e0b',
  high: '#dc2626',
}

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function HallucinationDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/hallucination/overview`),
          axios.get(`${API_URL}/api/hallucination/breakdown`),
          axios.get(`${API_URL}/api/hallucination/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load hallucination data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128270;</div>
      Loading hallucination analytics...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Hallucination data not available. Ingest documents and run analyses first.'}
    </div>
  )

  const s = overview.summary || {}
  const riskBreakdown = overview.risk_breakdown || []
  const groundingDist = overview.grounding_distribution || {}
  const typeCoverage = overview.type_coverage || []
  const confStats = overview.confidence_stats || {}
  const patientG = breakdown?.patient_grounding || []
  const diseaseStats = breakdown?.disease_coverage || []
  const interactionStats = breakdown?.interaction_faithfulness || {}
  const hitlStats = breakdown?.hitl_verification || {}
  const mitigations = breakdown?.mitigations || []
  const defsList = defs?.metrics || []

  const riskColor = RISK_COLORS[s.risk_level] || '#64748b'

  const kpiItems = [
    { label: 'Risk Score', value: s.overall_risk_score, color: riskColor, unit: '/100', decimals: 1 },
    { label: 'Grounding', value: s.grounding_score, color: COLORS[0], unit: '%', decimals: 1 },
    { label: 'Citation Rate', value: s.citation_rate, color: COLORS[2], unit: '%', decimals: 1 },
    { label: 'Faithfulness', value: s.faithfulness_rate, color: COLORS[1], unit: '%', decimals: 1 },
    { label: 'Embeddings', value: s.total_embeddings, color: COLORS[3] },
    { label: 'HITL Reviews', value: s.hitl_reviews, color: COLORS[5] },
  ]

  const groundingPieData = [
    { name: 'Grounded', value: groundingDist.grounded || 0 },
    { name: 'Partial', value: groundingDist.partially_grounded || 0 },
    { name: 'Ungrounded', value: groundingDist.ungrounded || 0 },
  ]
  const groundingPieColors = ['#10b981', '#f59e0b', '#ef4444']

  const radarData = riskBreakdown.map(r => ({
    type: r.label.replace(/ /g, '\n'),
    risk: r.risk_score,
  }))

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'grounding', label: 'Grounding' },
    { id: 'mitigations', label: 'Mitigations' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
    border: '1px solid #e5e7eb',
  }

  const kpiCardStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    flex: 1,
    minWidth: 130,
    padding: 16,
  })

  const sectionHeading = { fontSize: 16, fontWeight: 700, margin: '20px 0 12px', color: '#334155' }

  const tabStyle = (active) => ({
    padding: '8px 18px',
    borderRadius: 8,
    border: 'none',
    cursor: 'pointer',
    fontWeight: active ? 700 : 500,
    fontSize: 13,
    background: active ? '#1e88e5' : '#f1f5f9',
    color: active ? '#fff' : '#475569',
    transition: 'all 0.15s',
  })

  const statusDot = (status) => ({
    display: 'inline-block',
    width: 8,
    height: 8,
    borderRadius: '50%',
    marginRight: 6,
    background: status === 'active' ? '#10b981' : status === 'partial' ? '#f59e0b' : '#94a3b8',
  })

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Hallucination Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Risk level:{' '}
          <span style={{ fontWeight: 700, color: riskColor, textTransform: 'uppercase' }}>
            {s.risk_level}
          </span>
          {' \u2014 '}{s.total_embeddings} embeddings, {s.total_analyses} analyses, {s.total_rag_queries} RAG queries
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(activeTab === t.id)} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* KPI Cards — always visible */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiCardStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: kpi.color }}>
              {fmt(kpi.value, kpi.decimals || 0)}{kpi.unit || ''}
            </div>
          </div>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {activeTab === 'overview' && (
        <>
          {/* Risk Radar + Grounding Pie side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div style={cardStyle}>
              <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Hallucination Risk by Type</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={radarData} outerRadius={100}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="type" tick={{ fontSize: 10, fill: '#475569' }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar name="Risk" dataKey="risk" stroke="#f44336" fill="#f44336" fillOpacity={0.25} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Grounding Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={groundingPieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={95}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {groundingPieData.map((_, i) => (
                      <Cell key={i} fill={groundingPieColors[i]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Risk Breakdown Table */}
          <div style={{ ...cardStyle, marginBottom: 16 }}>
            <h3 style={sectionHeading}>Risk Breakdown by Hallucination Type</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Type</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Severity</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Risk Score</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Mitigation</th>
                </tr>
              </thead>
              <tbody>
                {riskBreakdown.map(r => (
                  <tr key={r.type} style={{ borderBottom: '1px solid #e5e7eb' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 500 }}>{r.label}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <span style={{
                        padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                        background: SEVERITY_COLORS[r.severity] + '20',
                        color: SEVERITY_COLORS[r.severity],
                      }}>
                        {r.severity}
                      </span>
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{
                          width: 80, height: 6, borderRadius: 3, background: '#e5e7eb',
                          position: 'relative', overflow: 'hidden',
                        }}>
                          <div style={{
                            width: `${Math.min(r.risk_score, 100)}%`,
                            height: '100%', borderRadius: 3,
                            background: r.risk_score > 60 ? '#ef4444' : r.risk_score > 40 ? '#f59e0b' : '#10b981',
                          }} />
                        </div>
                        <span style={{ fontSize: 12, fontWeight: 600 }}>{fmt(r.risk_score, 1)}</span>
                      </div>
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{r.mitigation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Document Type Coverage + Confidence Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            {typeCoverage.length > 0 && (
              <div style={cardStyle}>
                <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Document Type Coverage</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={typeCoverage} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="type" tick={{ fontSize: 11, fill: '#475569' }} />
                    <YAxis tick={{ fontSize: 11, fill: '#475569' }} />
                    <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {typeCoverage.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div style={cardStyle}>
              <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Confidence Calibration</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                {[
                  { label: 'Average', val: confStats.avg, color: COLORS[0] },
                  { label: 'Minimum', val: confStats.min, color: COLORS[4] },
                  { label: 'Maximum', val: confStats.max, color: COLORS[2] },
                ].map(c => (
                  <div key={c.label} style={{ textAlign: 'center', padding: 16, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{c.label}</div>
                    <div style={{ fontSize: 28, fontWeight: 700, color: c.color }}>{fmt(c.val, 3)}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 16, padding: 12, background: '#f0fdf4', borderRadius: 8, fontSize: 12, color: '#166534' }}>
                Confidence scores from {s.total_analyses} AI predictions.
                Ideal calibration: predicted confidence matches actual accuracy.
              </div>
            </div>
          </div>

          {/* Interaction Faithfulness */}
          <div style={{ ...cardStyle, marginBottom: 16 }}>
            <h3 style={sectionHeading}>Interaction Faithfulness</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
              {[
                { label: 'AI Responses', val: interactionStats.total_assistant_responses, color: COLORS[0] },
                { label: 'Operator Messages', val: interactionStats.total_operator_messages, color: COLORS[1] },
                { label: 'Corrections', val: interactionStats.corrections, color: COLORS[4] },
                { label: 'Confirmations', val: interactionStats.confirmations, color: COLORS[2] },
                { label: 'Faithfulness Rate', val: `${fmt(interactionStats.faithfulness_rate, 1)}%`, color: COLORS[5], raw: true },
              ].map(c => (
                <div key={c.label} style={{ textAlign: 'center', padding: 14, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{c.label}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: c.color }}>
                    {c.raw ? c.val : fmt(c.val)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── Grounding Tab ── */}
      {activeTab === 'grounding' && (
        <>
          {/* Per-Patient Grounding */}
          {patientG.length > 0 && (
            <div style={{ ...cardStyle, marginBottom: 16 }}>
              <h3 style={sectionHeading}>Per-Patient Grounding Scores</h3>
              <ResponsiveContainer width="100%" height={Math.max(250, patientG.length * 28)}>
                <BarChart data={patientG} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11, fill: '#475569' }}
                    label={{ value: 'Grounding Score %', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                  <YAxis dataKey="patient_id" type="category" tick={{ fontSize: 10, fill: '#475569' }} width={55} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                    formatter={(val, name) => [fmt(val, 1) + '%', name]} />
                  <Bar dataKey="grounding_score" radius={[0, 4, 4, 0]} name="Grounding %">
                    {patientG.map((entry, i) => (
                      <Cell key={i} fill={entry.grounding_score >= 75 ? '#10b981' : entry.grounding_score >= 50 ? '#f59e0b' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Disease Coverage */}
          {diseaseStats.length > 0 && (
            <div style={{ ...cardStyle, marginBottom: 16 }}>
              <h3 style={sectionHeading}>Analysis Coverage by Disease</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={diseaseStats} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="disease" tick={{ fontSize: 12, fill: '#475569' }} />
                  <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
                  <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
                  <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} name="Analyses" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* HITL Verification */}
          <div style={{ ...cardStyle, marginBottom: 16 }}>
            <h3 style={sectionHeading}>Human-in-the-Loop Verification</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              {[
                { label: 'Total Reviews', val: hitlStats.total_reviews, color: COLORS[0] },
                { label: 'Approved', val: hitlStats.approved, color: '#10b981' },
                { label: 'Rejected', val: hitlStats.rejected, color: '#ef4444' },
              ].map(c => (
                <div key={c.label} style={{ textAlign: 'center', padding: 20, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{c.label}</div>
                  <div style={{ fontSize: 32, fontWeight: 700, color: c.color }}>{fmt(c.val)}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ── Mitigations Tab ── */}
      {activeTab === 'mitigations' && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Hallucination Mitigation Strategies</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Strategy</th>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Status</th>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Coverage</th>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600 }}>Effectiveness</th>
              </tr>
            </thead>
            <tbody>
              {mitigations.map(m => (
                <tr key={m.strategy} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 500 }}>{m.strategy}</td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={statusDot(m.status)} />
                    <span style={{ fontSize: 12, textTransform: 'capitalize' }}>{m.status}</span>
                  </td>
                  <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{m.coverage}</td>
                  <td style={{ padding: '10px 12px', fontSize: 12, fontWeight: 600, color: '#1e88e5' }}>{m.effectiveness}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {activeTab === 'definitions' && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Metric Definitions</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, width: '20%' }}>Metric</th>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, width: '50%' }}>Definition</th>
                <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, width: '30%' }}>Source</th>
              </tr>
            </thead>
            <tbody>
              {defsList.map(d => (
                <tr key={d.metric} style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <td style={{ padding: '10px 12px', fontWeight: 500 }}>{d.metric}</td>
                  <td style={{ padding: '10px 12px', color: '#475569' }}>{d.definition}</td>
                  <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>{d.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
