import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'

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

const AGREE_COLORS = { Agree: '#10b981', Disagree: '#ef4444' }
const DECISION_COLORS = { Accept: '#3b82f6', Override: '#f59e0b' }
const ROLE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#10b981']
const PIE_COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6']

export default function HumanEvaluationDashboard() {
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
      axios.get(`${API}/human-evaluation/overview`),
      axios.get(`${API}/human-evaluation/breakdown`),
      axios.get(`${API}/human-evaluation/definitions`),
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
    { id: 'hitl', label: 'HITL Reviews' },
    { id: 'expert', label: 'Expert Analysis' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Human Evaluation dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No human evaluation data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Human Evaluation AI</h2>
        <Badge
          text={`Agreement: ${overview.kpis?.agreement_rate != null ? overview.kpis.agreement_rate + '%' : 'N/A'}`}
          color={overview.kpis?.agreement_rate >= 70 ? '#10b981' : overview.kpis?.agreement_rate >= 50 ? '#f59e0b' : '#ef4444'}
        />
        <Badge
          text={`Override: ${overview.kpis?.override_rate != null ? overview.kpis.override_rate + '%' : 'N/A'}`}
          color={overview.kpis?.override_rate <= 20 ? '#10b981' : '#ef4444'}
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'hitl' && renderHITL()}
      {tab === 'expert' && renderExpert()}
      {tab === 'patients' && renderPatients()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const kpis = [
      { label: 'HITL Reviews', value: k.total_hitl_reviews, color: '#3b82f6' },
      { label: 'Expert Reviews', value: k.total_expert_reviews, color: '#8b5cf6' },
      { label: 'Clinical Decisions', value: k.total_clinical_decisions, color: '#06b6d4' },
      { label: 'Component Findings', value: k.total_component_findings, color: '#f97316' },
      { label: 'Feedback Entries', value: k.total_feedback, color: '#10b981' },
      { label: 'Agreement Rate', value: k.agreement_rate != null ? k.agreement_rate + '%' : 'N/A', color: k.agreement_rate >= 70 ? '#10b981' : '#ef4444' },
      { label: 'Override Rate', value: k.override_rate != null ? k.override_rate + '%' : 'N/A', color: k.override_rate <= 20 ? '#10b981' : '#f59e0b' },
      { label: 'Avg Feedback Rating', value: k.avg_feedback_rating != null ? k.avg_feedback_rating + '/5' : 'N/A', color: '#f59e0b' },
    ]

    const agreementData = overview.agreement_breakdown || []
    const decisionData = overview.decision_types || []
    const roleData = overview.role_distribution || []
    const timelineData = overview.review_timeline || []
    const confData = overview.confidence_vs_agreement || []
    const ratingData = overview.feedback_ratings || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Agreement Breakdown" span={2}>
          {agreementData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={agreementData} dataKey="value" nameKey="label" cx="50%" cy="50%" outerRadius={90} label={({ label, value }) => `${label}: ${value}`}>
                  {agreementData.map((d, i) => <Cell key={i} fill={AGREE_COLORS[d.label] || PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Decision Types" span={2}>
          {decisionData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={decisionData} dataKey="value" nameKey="label" cx="50%" cy="50%" outerRadius={90} label={({ label, value }) => `${label}: ${value}`}>
                  {decisionData.map((d, i) => <Cell key={i} fill={DECISION_COLORS[d.label] || PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Role Distribution" span={2}>
          {roleData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={roleData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Reviews" radius={[4, 4, 0, 0]}>
                  {roleData.map((d, i) => <Cell key={i} fill={ROLE_COLORS[i % ROLE_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Feedback Ratings" span={2}>
          {ratingData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ratingData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="rating" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Responses" radius={[4, 4, 0, 0]} fill="#f59e0b" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Review Timeline" span={2}>
          {timelineData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={timelineData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="reviews" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Confidence vs Agreement" span={2}>
          {confData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="analysis_id" tick={{ fontSize: 11 }} label={{ value: 'Analysis ID', position: 'insideBottom', offset: -5, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} label={{ value: 'Confidence', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip formatter={(v, name) => name === 'confidence' ? (typeof v === 'number' ? v.toFixed(3) : v) : v} />
                <Bar dataKey="confidence" name="Confidence" radius={[4, 4, 0, 0]}>
                  {confData.map((d, i) => <Cell key={i} fill={d.agreed ? '#10b981' : '#ef4444'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8, fontSize: 12, color: '#64748b' }}>
            <span><span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 3, background: '#10b981', marginRight: 4, verticalAlign: 'middle' }}></span>Agreed</span>
            <span><span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 3, background: '#ef4444', marginRight: 4, verticalAlign: 'middle' }}></span>Disagreed</span>
          </div>
        </Card>
      </div>
    )
  }

  function renderHITL() {
    const hitlData = breakdown?.hitl_details || []
    const decisionData = breakdown?.clinical_decision_details || []
    const findingData = breakdown?.component_finding_details || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`HITL Reviews (${hitlData.length})`}>
          {hitlData.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>ID</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>AI Prediction</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Decision</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Human Decision</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Reason</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {hitlData.map((h, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{h.id}</td>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{h.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>{h.ai_prediction || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={h.decision || 'N/A'} color={h.decision === 'accept' ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px' }}>{h.human_decision || '-'}</td>
                      <td style={{ padding: '10px 12px' }}>{h.reason_code || '-'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>{h.created_at ? h.created_at.slice(0, 10) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No HITL review data available.</p>
          )}
        </Card>

        <Card title={`Clinical Decisions (${decisionData.length})`}>
          {decisionData.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>AI Prediction</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Top Channels</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Artifact Risk</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Neuro Agreement</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Final Decision</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Reviewer</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {decisionData.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{d.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>{d.ai_prediction || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                        {d.ai_confidence != null ? (typeof d.ai_confidence === 'number' ? d.ai_confidence.toFixed(2) : d.ai_confidence) : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 11 }}>{d.top_channels || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={d.artifact_risk || 'N/A'} color={d.artifact_risk === 'Low' ? '#10b981' : d.artifact_risk === 'High' ? '#ef4444' : '#f59e0b'} />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={d.neurologist_agreement || 'N/A'} color={d.neurologist_agreement === 'Yes' ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px' }}>{d.final_decision || 'N/A'}</td>
                      <td style={{ padding: '10px 12px' }}>{d.reviewer || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{d.note || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No clinical decision data available.</p>
          )}
        </Card>

        <Card title={`Component Findings (${findingData.length})`}>
          {findingData.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Component</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Doctor Finding</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Doctor</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Agree with AI</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {findingData.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{f.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>{f.component || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{f.doctor_finding || '-'}</td>
                      <td style={{ padding: '10px 12px' }}>{f.doctor || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={f.agree_with_ai || 'N/A'} color={(f.agree_with_ai || '').toLowerCase() === 'agree' ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>{f.created_at ? f.created_at.slice(0, 10) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No component finding data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderExpert() {
    const expertData = breakdown?.expert_details || []
    const roleMatrix = breakdown?.role_agreement_matrix || []
    const feedbackData = breakdown?.feedback_details || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Expert Reviews (${expertData.length})`}>
          {expertData.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Role</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Expert</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Finding</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Agreement</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Note</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {expertData.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{e.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}><Badge text={e.role || 'N/A'} color="#8b5cf6" /></td>
                      <td style={{ padding: '10px 12px' }}>{e.expert || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12, maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.finding || '-'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={e.agree_with_ai || 'N/A'} color={(e.agree_with_ai || '').toLowerCase() === 'agree' ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>{e.note || '-'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>{e.created_at ? e.created_at.slice(0, 10) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No expert review data available.</p>
          )}
        </Card>

        <Card title="Role Agreement Matrix">
          {roleMatrix.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Role</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#10b981' }}>Agree</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#ef4444' }}>Disagree</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Agreement Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {roleMatrix.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{r.role}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace', color: '#10b981', fontWeight: 600 }}>{r.agree_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace', color: '#ef4444', fontWeight: 600 }}>{r.disagree_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                          <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(r.rate || 0, 100)}%`, height: '100%', background: r.rate >= 70 ? '#10b981' : r.rate >= 50 ? '#f59e0b' : '#ef4444', borderRadius: 4 }} />
                          </div>
                          <span style={{ fontSize: 11, fontFamily: 'monospace', fontWeight: 600 }}>{r.rate != null ? r.rate + '%' : 'N/A'}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No role agreement data available.</p>
          )}
        </Card>

        <Card title={`Clinician Feedback (${feedbackData.length})`}>
          {feedbackData.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Role</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>AI Output</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Rating</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Correction</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Reason</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {feedbackData.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{f.patient_id || 'N/A'}</td>
                      <td style={{ padding: '10px 12px' }}>{f.role || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{f.ai_output || '-'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={f.rating != null ? f.rating + '/5' : 'N/A'} color={f.rating >= 4 ? '#10b981' : f.rating >= 3 ? '#f59e0b' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{f.correction || '-'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{f.reason || '-'}</td>
                      <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>{f.created_at ? f.created_at.slice(0, 10) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No feedback data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const profiles = breakdown?.patient_evaluation_profiles || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Evaluation Profiles (${profiles.length} patients)`}>
          {profiles.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#3b82f6' }}>HITL</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#8b5cf6' }}>Expert</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#06b6d4' }}>Decisions</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#f97316' }}>Findings</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#10b981' }}>Feedback</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Agreement Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {profiles.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{p.hitl_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{p.expert_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{p.decision_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{p.finding_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{p.feedback_count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        {p.expert_count > 0 ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                            <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                              <div style={{ width: `${Math.min(p.agreement_rate || 0, 100)}%`, height: '100%', background: p.agreement_rate >= 70 ? '#10b981' : p.agreement_rate >= 50 ? '#f59e0b' : '#ef4444', borderRadius: 4 }} />
                            </div>
                            <span style={{ fontSize: 11, fontFamily: 'monospace', fontWeight: 600 }}>{p.agreement_rate}%</span>
                          </div>
                        ) : (
                          <span style={{ color: '#94a3b8', fontSize: 11 }}>N/A</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No patient evaluation data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, si) => (
          <Card key={si} title={sec.title}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
