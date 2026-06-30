import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API = '/api'
const C = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#64748b','#84cc16','#f97316']

function fmt(v) { return v == null ? '--' : typeof v === 'number' ? v.toLocaleString() : String(v) }
function pct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

export default function AgentEvaluationDashboard() {
  const [ov, setOv] = useState(null)
  const [br, setBr] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    (async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API}/agent-eval/overview`),
          axios.get(`${API}/agent-eval/breakdown`),
          axios.get(`${API}/agent-eval/definitions`)
        ])
        setOv(o.data); setBr(b.data); setDefs(d.data)
      } catch (e) { setError(e.message || 'Failed to load agent evaluation data') }
      setLoading(false)
    })()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading agent evaluation data...
    </div>
  )
  if (error) return <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>{ov?.note || 'Agent evaluation data not available.'}</div>

  const s = ov.summary || {}
  const confDist = ov.confidence_distribution || []
  const diseaseDist = ov.disease_distribution || []
  const qualDist = ov.quality_distribution || []
  const routeSummary = ov.routing_summary || []
  const expertAgree = ov.expert_agreement || {}
  const expertByRole = ov.expert_by_role || []
  const decisionDist = ov.decision_distribution || []
  const perPatient = ov.per_patient || []
  const agentActions = ov.agent_actions || []
  const dailyTrend = ov.daily_trend || []

  const expertLog = br?.expert_log || []
  const hitlLog = br?.hitl_log || []
  const decisionLog = br?.decision_log || []
  const eventLog = br?.event_log || []

  const card = { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }
  const metric = { background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }
  const th = { padding: '8px 10px', textAlign: 'left', color: '#334155', borderBottom: '1px solid #e5e7eb', fontSize: 12, fontWeight: 600 }
  const td = { padding: '6px 10px', color: '#1f2937', borderBottom: '1px solid #f1f5f9', fontSize: 12 }

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'reviews', label: 'Expert & HITL Reviews' },
    { id: 'decisions', label: 'Decisions & Routing' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const agreePie = [
    { name: 'Agree', value: expertAgree.agree || 0 },
    { name: 'Disagree', value: expertAgree.disagree || 0 }
  ]

  return (
    <div>
      <div style={{ ...card, background: 'linear-gradient(135deg, #dbeafe 0%, #ede9fe 100%)', marginBottom: 20 }}>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Agent Evaluation Dashboard</div>
        <div style={{ fontSize: 13, color: '#475569' }}>
          AI analysis confidence, expert/HITL agreement, clinical decision routing, and review coverage from clinical.db
        </div>
      </div>

      {/* tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: '1px solid #e5e7eb', borderRadius: 6, cursor: 'pointer', fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#fff', color: tab === t.id ? '#fff' : '#475569', fontWeight: tab === t.id ? 600 : 400
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && <>
        {/* KPIs */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginBottom: 16 }}>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Total Analyses</div><div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{fmt(s.total_analyses)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Avg Confidence</div><div style={{ fontSize: 22, fontWeight: 700, color: '#3b82f6' }}>{pct(s.avg_confidence)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Expert Reviews</div><div style={{ fontSize: 22, fontWeight: 700, color: '#10b981' }}>{fmt(s.total_expert_reviews)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Expert Agree Rate</div><div style={{ fontSize: 22, fontWeight: 700, color: s.expert_agree_rate >= 0.8 ? '#10b981' : '#f59e0b' }}>{pct(s.expert_agree_rate)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>HITL Reviews</div><div style={{ fontSize: 22, fontWeight: 700, color: '#8b5cf6' }}>{fmt(s.total_hitl_reviews)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>HITL Override Rate</div><div style={{ fontSize: 22, fontWeight: 700, color: s.hitl_override_rate > 0.3 ? '#ef4444' : '#10b981' }}>{pct(s.hitl_override_rate)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Clinical Decisions</div><div style={{ fontSize: 22, fontWeight: 700, color: '#ec4899' }}>{fmt(s.total_clinical_decisions)}</div></div>
          <div style={metric}><div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Review Coverage</div><div style={{ fontSize: 22, fontWeight: 700, color: s.review_coverage >= 0.5 ? '#10b981' : '#f59e0b' }}>{pct(s.review_coverage)}</div></div>
        </div>

        {/* Confidence distribution + Disease distribution */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Confidence Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={confDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Disease Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={diseaseDist} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={80} label={({ disease, count }) => `${disease}: ${count}`}>
                  {diseaseDist.map((_, i) => <Cell key={i} fill={C[i % C.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Decision routing + Expert agreement */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Decision Routing (by confidence)</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={routeSummary}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="route" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {routeSummary.map((entry, i) => (
                    <Cell key={i} fill={entry.route === 'auto' ? '#10b981' : entry.route === 'review' ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Expert Agreement</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={agreePie} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  <Cell fill="#10b981" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Daily trend + Agent actions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Daily Agent Activity</div>
            {dailyTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={dailyTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="cnt" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 20, textAlign: 'center' }}>No daily trend data</div>}
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Agent Actions by Component</div>
            {agentActions.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={agentActions} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={120} />
                  <Tooltip />
                  <Bar dataKey="cnt" fill="#06b6d4" radius={[0,4,4,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 20, textAlign: 'center' }}>No action data</div>}
          </div>
        </div>

        {/* Signal quality + Per-patient review depth */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Signal Quality Distribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={qualDist}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#84cc16" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Per-Patient Review Depth</div>
            <div style={{ maxHeight: 200, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={th}>Patient</th><th style={th}>Analyses</th><th style={th}>Expert</th><th style={th}>HITL</th><th style={th}>Decisions</th>
                </tr></thead>
                <tbody>
                  {perPatient.slice(0, 15).map((p, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={td}><code style={{ fontSize: 11 }}>{p.patient_id}</code></td>
                      <td style={td}>{p.analyses}</td>
                      <td style={td}>{p.expert_reviews}</td>
                      <td style={td}>{p.hitl_reviews}</td>
                      <td style={td}>{p.clinical_decisions}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </>}

      {/* REVIEWS TAB */}
      {tab === 'reviews' && <>
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Expert Reviews ({expertLog.length})</div>
          {expertLog.length > 0 ? (
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={th}>Patient</th><th style={th}>Role</th><th style={th}>Expert</th><th style={th}>Finding</th><th style={th}>Agree?</th><th style={th}>Note</th><th style={th}>Date</th>
                </tr></thead>
                <tbody>
                  {expertLog.map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={td}><code style={{ fontSize: 11 }}>{r.patient_id}</code></td>
                      <td style={td}>{r.role}</td>
                      <td style={td}>{r.expert}</td>
                      <td style={{ ...td, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.finding}</td>
                      <td style={td}><span style={{ color: r.agree_with_ai === 'agree' ? '#10b981' : '#ef4444', fontWeight: 600 }}>{r.agree_with_ai}</span></td>
                      <td style={{ ...td, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.note || '--'}</td>
                      <td style={td}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 16 }}>No expert reviews recorded.</div>}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Expert Reviews by Role</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={expertByRole}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Expert Agreement Breakdown</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={agreePie} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={75} label>
                  <Cell fill="#10b981" /><Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>HITL Reviews ({hitlLog.length})</div>
          {hitlLog.length > 0 ? (
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={th}>Patient</th><th style={th}>Analysis</th><th style={th}>AI Prediction</th><th style={th}>Decision</th><th style={th}>Human Decision</th><th style={th}>Date</th>
                </tr></thead>
                <tbody>
                  {hitlLog.map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={td}><code style={{ fontSize: 11 }}>{r.patient_id}</code></td>
                      <td style={td}>{r.analysis_id}</td>
                      <td style={td}>{r.ai_prediction || '--'}</td>
                      <td style={td}><span style={{ color: r.decision === 'accept' ? '#10b981' : '#ef4444', fontWeight: 600 }}>{r.decision || '--'}</span></td>
                      <td style={td}>{r.human_decision || '--'}</td>
                      <td style={td}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 16 }}>No HITL reviews recorded.</div>}
        </div>
      </>}

      {/* DECISIONS & ROUTING TAB */}
      {tab === 'decisions' && <>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Decision Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={decisionDist} dataKey="count" nameKey="decision" cx="50%" cy="50%" outerRadius={80} label={({ decision, count }) => `${decision}: ${count}`}>
                  {decisionDist.map((_, i) => <Cell key={i} fill={C[i % C.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Confidence-Based Routing</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={routeSummary}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="route" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {routeSummary.map((entry, i) => (
                    <Cell key={i} fill={entry.route === 'auto' ? '#10b981' : entry.route === 'review' ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
              Auto: confidence &ge; 0.8 &middot; Review: 0.5–0.8 &middot; Escalate: &lt; 0.5
            </div>
          </div>
        </div>

        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Clinical Decision Log ({decisionLog.length})</div>
          {decisionLog.length > 0 ? (
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={th}>Patient</th><th style={th}>AI Pred</th><th style={th}>Confidence</th><th style={th}>Decision</th><th style={th}>Reviewer</th><th style={th}>Agree?</th><th style={th}>Note</th><th style={th}>Date</th>
                </tr></thead>
                <tbody>
                  {decisionLog.map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={td}><code style={{ fontSize: 11 }}>{r.patient_id}</code></td>
                      <td style={td}>{r.ai_prediction}</td>
                      <td style={td}>{r.ai_confidence != null ? (r.ai_confidence * 100).toFixed(1) + '%' : '--'}</td>
                      <td style={td}><span style={{ fontWeight: 600, color: r.final_decision === 'Confirm' ? '#10b981' : '#ef4444' }}>{r.final_decision}</span></td>
                      <td style={td}>{r.reviewer}</td>
                      <td style={td}>{r.neurologist_agreement}</td>
                      <td style={{ ...td, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.note || '--'}</td>
                      <td style={td}>{r.created_at?.slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 16 }}>No clinical decisions recorded.</div>}
        </div>

        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>Recent Agent Events ({eventLog.length})</div>
          {eventLog.length > 0 ? (
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={th}>Patient</th><th style={th}>Component</th><th style={th}>Action</th><th style={th}>Actor</th><th style={th}>Detail</th><th style={th}>Date</th>
                </tr></thead>
                <tbody>
                  {eventLog.map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={td}><code style={{ fontSize: 11 }}>{r.patient_id || '--'}</code></td>
                      <td style={td}>{r.component}</td>
                      <td style={td}>{r.action}</td>
                      <td style={td}>{r.actor || '--'}</td>
                      <td style={{ ...td, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.detail || '--'}</td>
                      <td style={td}>{r.created_at?.slice(0, 16)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 16 }}>No agent events.</div>}
        </div>
      </>}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <div style={card}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 12 }}>{defs.title}</div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={th}>Metric</th><th style={th}>Definition</th></tr></thead>
            <tbody>
              {(defs.metrics || []).map((m, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ ...td, fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                  <td style={td}>{m.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
