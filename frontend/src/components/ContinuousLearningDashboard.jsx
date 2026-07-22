import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#f97316', '#64748b']

export default function ContinuousLearningDashboard() {
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
      axios.get(`${API}/api/continuous-learning/overview`),
      axios.get(`${API}/api/continuous-learning/breakdown`),
      axios.get(`${API}/api/continuous-learning/definitions`),
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
    { id: 'feedback', label: 'Feedback & HITL' },
    { id: 'training', label: 'Training Runs' },
    { id: 'errors', label: 'Error Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Continuous Learning dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No continuous learning data available.'}</div>

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>Continuous Learning Dashboard</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Feedback collection, error analysis, retrain triggers, and model improvement lifecycle
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Training Runs" value={overview.kpis.total_training_runs} sub={`${overview.kpis.training_success_pct}% success`} color="#3b82f6" /></Card>
            <Card><KPI label="Drift Events" value={overview.kpis.drift_events} sub={overview.kpis.drift_verdict || 'monitoring'} color={overview.kpis.drift_verdict === 'DRIFT' ? '#ef4444' : '#10b981'} /></Card>
            <Card><KPI label="HITL Reviews" value={overview.kpis.total_hitl_reviews} sub={`${overview.kpis.hitl_overrides} overrides`} color="#8b5cf6" /></Card>
            <Card><KPI label="Avg Confidence" value={overview.kpis.avg_confidence ?? 'N/A'} sub={`${overview.kpis.total_analyses} analyses`} color="#f59e0b" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Feedback Count" value={overview.kpis.total_feedback} sub={overview.kpis.avg_feedback_rating ? `avg ${overview.kpis.avg_feedback_rating}/5` : 'no ratings'} color="#06b6d4" /></Card>
            <Card><KPI label="Consistency Checks" value={overview.kpis.consistency_events} sub={overview.kpis.consistency_verdict || 'stable'} color="#10b981" /></Card>
            <Card><KPI label="Fairness Tests" value={overview.kpis.fairness_events} color="#ec4899" /></Card>
            <Card><KPI label="Failed Runs" value={overview.kpis.failed_runs} sub={`of ${overview.kpis.total_training_runs}`} color={overview.kpis.failed_runs > 0 ? '#ef4444' : '#10b981'} /></Card>
          </div>

          {/* Training success rate over time */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Training Success Rate Over Time">
              {(overview.training_success_rate || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={overview.training_success_rate}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="rate_pct" stroke="#3b82f6" strokeWidth={2} name="Success %" />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No training data yet</div>}
            </Card>

            <Card title="Confidence Distribution">
              {(overview.confidence_buckets || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={overview.confidence_buckets}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence data</div>}
            </Card>
          </div>

          {/* Retrain triggers */}
          <Card title="Retrain Trigger Sources">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              {overview.retrain_triggers && Object.entries(overview.retrain_triggers).map(([key, val]) => (
                <div key={key} style={{ textAlign: 'center', padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#334155' }}>{val}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{key.replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      {/* ── Feedback & HITL Tab ── */}
      {tab === 'feedback' && breakdown && (
        <>
          {/* Expert agreement */}
          {breakdown.expert_agreement_rate !== null && (
            <div style={{ marginBottom: 16, padding: 12, background: '#f0fdf4', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: '#166534' }}>Expert Agreement Rate:</span>
              <span style={{ fontSize: 20, fontWeight: 700, color: '#15803d' }}>{breakdown.expert_agreement_rate}%</span>
            </div>
          )}

          {/* HITL reviews */}
          <Card title="Human-in-the-Loop Reviews" span={2}>
            {(breakdown.hitl_log || []).length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Decision</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>AI Prediction</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Human Decision</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Reason</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.hitl_log.map((h, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px' }}>{h.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={h.decision} color={h.decision === 'override' ? '#ef4444' : '#10b981'} />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{h.ai_prediction || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{h.human_decision || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{h.reason_code || h.reviewer_id || '-'}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(h.created_at || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No HITL reviews yet</div>}
          </Card>

          {/* Expert reviews */}
          <Card title="Expert Reviews" span={2}>
            {(breakdown.expert_reviews || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 16 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Role</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Expert</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Finding</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Agrees</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.expert_reviews.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px' }}>{e.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{e.role}</td>
                        <td style={{ padding: '6px 10px' }}>{e.expert}</td>
                        <td style={{ padding: '6px 10px' }}>{e.finding}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={e.agree_with_ai} color={e.agree_with_ai === 'agree' ? '#10b981' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(e.created_at || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No expert reviews yet</div>}
          </Card>

          {/* Feedback log */}
          <Card title="Clinician Feedback Log" span={2}>
            {(breakdown.feedback_log || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 16 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Role</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Rating</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>AI Output</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.feedback_log.map((f, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px' }}>{f.patient_id || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{f.role}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={`${f.rating}/5`} color={f.rating >= 4 ? '#10b981' : f.rating >= 3 ? '#f59e0b' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{f.ai_output || '-'}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(f.created_at || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No feedback yet</div>}
          </Card>
        </>
      )}

      {/* ── Training Runs Tab ── */}
      {tab === 'training' && breakdown && (
        <>
          {/* Daily ML events chart */}
          <Card title="Daily ML Pipeline Events" span={2}>
            {(breakdown.daily_ml_events || []).length ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={breakdown.daily_ml_events}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="training" fill="#3b82f6" name="Training" stackId="a" />
                  <Bar dataKey="drift" fill="#ef4444" name="Drift" stackId="a" />
                  <Bar dataKey="consistency" fill="#10b981" name="Consistency" stackId="a" />
                  <Bar dataKey="fairness" fill="#8b5cf6" name="Fairness" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No ML events yet</div>}
          </Card>

          {/* Training run detail */}
          <Card title="Training Run History" span={2}>
            {(breakdown.training_runs || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Action</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Detail</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Actor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.training_runs.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(t.date || '').slice(0, 16)}</td>
                        <td style={{ padding: '6px 10px' }}>{t.action}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={t.success ? 'success' : 'failed'} color={t.success ? '#10b981' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{t.detail}</td>
                        <td style={{ padding: '6px 10px' }}>{t.actor}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No training runs yet</div>}
          </Card>

          {/* Drift features */}
          {(breakdown.drift_features || []).length > 0 && (
            <Card title="Drift Feature Analysis" span={2}>
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Feature</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>P-Value</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Statistic</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Drifted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.drift_features.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.feature}</td>
                        <td style={{ padding: '6px 10px' }}>{d.p_value != null ? d.p_value.toFixed(4) : '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{d.statistic != null ? d.statistic.toFixed(4) : '-'}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={d.drifted ? 'DRIFTED' : 'stable'} color={d.drifted ? '#ef4444' : '#10b981'} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}

      {/* ── Error Analysis Tab ── */}
      {tab === 'errors' && breakdown && (
        <>
          <Card title="Per-Patient Error Analysis" span={2}>
            {(breakdown.patient_errors || []).length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Analyses</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Latest Prediction</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Signal Quality</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>HITL Overrides</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.patient_errors.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: p.has_errors ? '#fef2f2' : undefined }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{p.total_analyses}</td>
                        <td style={{ padding: '6px 10px' }}>{p.latest_prediction || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{p.latest_confidence ?? '-'}</td>
                        <td style={{ padding: '6px 10px' }}>
                          {p.latest_quality && <Badge text={p.latest_quality} color={p.latest_quality === 'Good' ? '#10b981' : '#f59e0b'} />}
                        </td>
                        <td style={{ padding: '6px 10px' }}>{p.hitl_overrides}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={p.has_errors ? 'has errors' : 'clean'} color={p.has_errors ? '#ef4444' : '#10b981'} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient analysis data yet</div>}
          </Card>

          {/* ML event timeline */}
          <Card title="ML Pipeline Event Timeline" span={2}>
            {(overview.ml_timeline || []).length ? (
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                {overview.ml_timeline.slice(-30).reverse().map((e, i) => (
                  <div key={i} style={{ display: 'flex', gap: 12, padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                    <span style={{ color: '#64748b', minWidth: 80 }}>{e.date}</span>
                    <Badge text={e.component} color={
                      e.component === 'training' ? '#3b82f6' :
                      e.component === 'drift' ? '#ef4444' :
                      e.component === 'consistency' ? '#10b981' : '#8b5cf6'
                    } />
                    <span style={{ color: '#334155' }}>{e.detail}</span>
                  </div>
                ))}
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No ML events yet</div>}
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && definitions?.sections && (
        <div style={{ display: 'grid', gap: 16 }}>
          {definitions.sections.map((sec, si) => (
            <Card key={si} title={sec.title} span={2}>
              <div style={{ display: 'grid', gap: 8 }}>
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 2 }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
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
