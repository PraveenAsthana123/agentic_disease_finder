import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ReferenceLine
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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

function StatusBadge({ status }) {
  const colorMap = { complete: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function RLHFTrainingDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/rlhf-training/overview`),
          axios.get(`${API_URL}/rlhf-training/breakdown`),
          axios.get(`${API_URL}/rlhf-training/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading RLHF Training data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'pairs', label: 'Preference Pairs' },
    { id: 'config', label: 'Training Config' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const ratingDist = overview?.rating_distribution
    ? Object.entries(overview.rating_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []

  const roleDist = overview?.role_distribution
    ? Object.entries(overview.role_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []

  const pipelineStages = overview?.pipeline_stages || []

  /* --- Breakdown data --- */
  const monthlyTrend = breakdown?.monthly_feedback_trend || []
  const overrideCodes = breakdown?.override_reason_codes || []
  const roleQuality = breakdown?.role_quality_metrics || []
  const rewardReadiness = breakdown?.reward_model_readiness || []

  /* --- Preference Pairs data --- */
  const prefPairs = overview?.preference_pairs || []
  const sourceDist = overview?.source_distribution
    ? Object.entries(overview.source_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []

  /* --- Training Config data --- */
  const trainingConfig = overview?.training_config || {}
  const readinessChecklist = overview?.readiness_checklist || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>RLHF Training Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Reinforcement Learning from Human Feedback — feedback collection, preference pairs, and reward model training
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Feedback" value={fmt(overview.total_feedback)} />
              <KPI label="HITL Reviews" value={fmt(overview.total_hitl_reviews)} />
              <KPI label="Expert Reviews" value={fmt(overview.total_expert_reviews)} />
              <KPI label="Preference Pairs" value={fmt(overview.preference_pairs_count)} />
              <KPI label="RLHF Readiness" value={fmtPct(overview.rlhf_readiness_score)} color={
                overview.rlhf_readiness_score >= 0.8 ? '#22c55e' : overview.rlhf_readiness_score >= 0.5 ? '#eab308' : '#ef4444'
              } />
            </div>
          </Card>

          {/* Pipeline Stages */}
          <Card title="Pipeline Stages" span={3}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, overflowX: 'auto', padding: '8px 0' }}>
              {pipelineStages.map((stage, i) => {
                const statusColor = stage.status === 'complete' ? '#22c55e' : stage.status === 'in-progress' ? '#eab308' : '#94a3b8'
                return (
                  <React.Fragment key={i}>
                    <div style={{
                      display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 120, padding: '8px 12px',
                      background: statusColor + '18', borderRadius: 10, border: `1.5px solid ${statusColor}40`
                    }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{stage.name}</div>
                      <StatusBadge status={stage.status} />
                      {stage.detail && <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>{stage.detail}</div>}
                    </div>
                    {i < pipelineStages.length - 1 && (
                      <div style={{ fontSize: 18, color: '#cbd5e1', padding: '0 4px', flexShrink: 0 }}>&#x2192;</div>
                    )}
                  </React.Fragment>
                )
              })}
            </div>
          </Card>

          {/* Rating Distribution Bar Chart */}
          <Card title="Rating Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ratingDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" name="Count" radius={[4, 4, 0, 0]}>
                  {ratingDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Role Distribution Pie */}
          <Card title="Role Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={roleDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
                  {roleDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {roleDist.map((d, i) => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: COLORS[i % COLORS.length], marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Agreement Rate */}
          {overview.agreement_rate != null && (
            <Card title="Agreement Rate" span={3}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 20, padding: '8px 0' }}>
                <div style={{ fontSize: 36, fontWeight: 700, color: overview.agreement_rate >= 0.8 ? '#22c55e' : '#eab308' }}>
                  {fmtPct(overview.agreement_rate)}
                </div>
                <div>
                  <div style={{ fontSize: 13, color: '#475569' }}>Inter-annotator agreement across all reviewers</div>
                  <div style={{
                    marginTop: 6, height: 8, width: 300, background: '#f1f5f9', borderRadius: 4, position: 'relative'
                  }}>
                    <div style={{
                      width: `${(overview.agreement_rate * 100)}%`, background: overview.agreement_rate >= 0.8 ? '#22c55e' : '#eab308',
                      borderRadius: 4, height: '100%'
                    }} />
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ======================== BREAKDOWN TAB ======================== */}
      {tab === 'breakdown' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Monthly Feedback Trend */}
          <Card title="Monthly Feedback Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={monthlyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="feedback_count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Feedback Count" />
                {monthlyTrend.length > 0 && monthlyTrend[0].pair_count != null && (
                  <Line type="monotone" dataKey="pair_count" stroke="#22c55e" strokeWidth={2} dot={{ r: 4 }} name="Pair Count" />
                )}
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Override Reason Codes */}
          <Card title="Override Reason Codes">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overrideCodes} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="reason" width={90} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" radius={[0, 4, 4, 0]}>
                  {overrideCodes.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Role Quality Metrics Table */}
          <Card title="Per-Role Quality Metrics">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Role</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Reviews</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Rating</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Agreement</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Override %</th>
                  </tr>
                </thead>
                <tbody>
                  {roleQuality.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.role}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.reviews)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.avg_rating)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(r.agreement)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(r.override_rate)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Reward Model Readiness */}
          <Card title="Reward Model Readiness: Dataset Size vs Thresholds" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={rewardReadiness}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="current" fill="#3b82f6" name="Current" radius={[4, 4, 0, 0]} />
                <Bar dataKey="threshold" fill="#e2e8f0" name="Threshold" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== PREFERENCE PAIRS TAB ======================== */}
      {tab === 'pairs' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Preference Pairs Table */}
          <Card title="Preference Pairs" span={2}>
            <div style={{ maxHeight: 420, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Chosen</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Rejected</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Source</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {prefPairs.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        <span style={{ color: '#22c55e', fontWeight: 600 }}>{p.chosen}</span>
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        <span style={{ color: '#ef4444' }}>{p.rejected}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                          background: '#3b82f622', color: '#3b82f6'
                        }}>{p.source}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{p.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Source Distribution Pie */}
          <Card title="Source Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sourceDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
                  {sourceDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {sourceDist.map((d, i) => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: COLORS[i % COLORS.length], marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ======================== TRAINING CONFIG TAB ======================== */}
      {tab === 'config' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Hyperparameters Table */}
          <Card title="Training Configuration (Hyperparameters)">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(trainingConfig).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155', whiteSpace: 'nowrap' }}>{k}</td>
                    <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Readiness Checklist */}
          <Card title="Readiness Checklist">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {readinessChecklist.map((item, i) => (
                <div key={i} style={{
                  display: 'flex', alignItems: 'center', gap: 10, padding: '8px 12px',
                  background: item.passed ? '#22c55e12' : '#ef444412', borderRadius: 8,
                  border: `1px solid ${item.passed ? '#22c55e30' : '#ef444430'}`
                }}>
                  <span style={{ fontSize: 16 }}>{item.passed ? '\u2705' : '\u274C'}</span>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{item.label}</div>
                    {item.detail && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{item.detail}</div>}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <Card title={definitions.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(definitions.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
