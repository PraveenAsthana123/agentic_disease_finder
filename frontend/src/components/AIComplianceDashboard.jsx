import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const TIER_COLORS = { 'High-Risk': '#ef4444', 'Limited Risk': '#f59e0b', 'Minimal Risk': '#10b981' }
const CHECK_COLORS = { pass: '#10b981', fail: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AIComplianceDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/ai-compliance/overview`),
          axios.get(`${API_URL}/api/ai-compliance/breakdown`),
          axios.get(`${API_URL}/api/ai-compliance/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load compliance data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI compliance data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Compliance data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const dailyAudit = overview.daily_audit_trail || []
  const euTiers = overview.eu_ai_act_tiers || []
  const expertRoles = overview.expert_by_role || []
  const checklist = breakdown?.governance_checklist || []
  const hitlReviews = breakdown?.hitl_reviews || []
  const expertReviews = breakdown?.expert_reviews || []
  const decisions = breakdown?.clinical_decisions || []
  const componentAudit = breakdown?.component_audit || []

  const reviewPieData = [
    { name: 'HITL', value: s.total_hitl_reviews, fill: '#3b82f6' },
    { name: 'Expert', value: s.total_expert_reviews, fill: '#8b5cf6' },
    { name: 'Clinical', value: s.total_clinical_decisions, fill: '#10b981' }
  ].filter(d => d.value > 0)

  const agreePieData = [
    { name: 'Agree', value: s.expert_agree, fill: '#10b981' },
    { name: 'Disagree', value: s.expert_disagree, fill: '#ef4444' }
  ].filter(d => d.value > 0)

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#1e40af' : '#f1f5f9', color: active ? '#fff' : '#64748b',
    border: 'none', fontSize: 13, marginRight: 4
  })

  const kpiItems = [
    { label: 'Analyses', value: s.total_analyses, color: '#64748b' },
    { label: 'HITL Reviews', value: s.total_hitl_reviews, color: '#3b82f6' },
    { label: 'Override Rate', value: s.override_rate_pct != null ? `${s.override_rate_pct}%` : '--', color: s.override_rate_pct > 30 ? '#ef4444' : '#10b981' },
    { label: 'Expert Reviews', value: s.total_expert_reviews, color: '#8b5cf6' },
    { label: 'Agreement', value: s.agreement_rate_pct != null ? `${s.agreement_rate_pct}%` : '--', color: '#10b981' },
    { label: 'Review Coverage', value: s.review_coverage_pct != null ? `${s.review_coverage_pct}%` : '--', color: s.review_coverage_pct >= 80 ? '#10b981' : '#f59e0b' },
    { label: 'Accountability', value: s.accountability_pct != null ? `${s.accountability_pct}%` : '--', color: s.accountability_pct >= 95 ? '#10b981' : '#ef4444' },
    { label: 'Consent', value: s.consent_pct != null ? `${s.consent_pct}%` : '--', color: '#06b6d4' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 110
  })

  const tierBadge = (tier) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${TIER_COLORS[tier] || '#94a3b8'}22`, color: TIER_COLORS[tier] || '#64748b'
  })

  const checkBadge = (status) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${CHECK_COLORS[status] || '#94a3b8'}22`, color: CHECK_COLORS[status] || '#64748b'
  })

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'reviews', label: 'Review Audit' },
    { id: 'governance', label: 'Governance' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Compliance Dashboard</h2>
      </div>

      {/* Tab bar */}
      <div style={{ marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
                  {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Charts row: Review types + Expert agreement */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Reviews by Type</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={reviewPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                    label={({ name, value }) => `${name}: ${value}`}>
                    {reviewPieData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Expert Agreement with AI</h4>
              {agreePieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={agreePieData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                      label={({ name, value }) => `${name}: ${value}`}>
                      {agreePieData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No expert reviews yet</div>
              )}
            </div>
          </div>

          {/* Audit trail timeline */}
          {dailyAudit.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Audit Trail — Daily Operations & Actors</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyAudit}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="ops" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Operations" />
                  <Line type="monotone" dataKey="actors" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Distinct Actors" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Expert reviews by role */}
          {expertRoles.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Expert Reviews by Role</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={expertRoles}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="reviews" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Total Reviews" />
                  <Bar dataKey="agreed" fill="#10b981" radius={[4, 4, 0, 0]} name="Agreed with AI" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* EU AI Act tier mapping */}
          {euTiers.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>EU AI Act Risk Classification</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Risk Tier</th>
                      <th style={{ padding: '8px 10px' }}>Article</th>
                      <th style={{ padding: '8px 10px' }}>Rationale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {euTiers.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.component}</td>
                        <td style={{ padding: '8px 10px' }}><span style={tierBadge(t.tier)}>{t.tier}</span></td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{t.article}</td>
                        <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{t.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ═══ REVIEWS TAB ═══ */}
      {tab === 'reviews' && (
        <>
          {/* HITL reviews table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>HITL Review Log ({hitlReviews.length})</h4>
            {hitlReviews.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>AI Prediction</th>
                      <th style={{ padding: '8px 10px' }}>Human Decision</th>
                      <th style={{ padding: '8px 10px' }}>Action</th>
                      <th style={{ padding: '8px 10px' }}>Reason</th>
                      <th style={{ padding: '8px 10px' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hitlReviews.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 10px' }}>{r.ai_prediction}</td>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.human_decision}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={checkBadge(r.decision === 'override' ? 'fail' : 'pass')}>
                            {r.decision || '--'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{r.reason_code || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.created_at ? new Date(r.created_at).toLocaleDateString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', padding: 30 }}>No HITL reviews recorded yet</div>
            )}
          </div>

          {/* Expert reviews table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Expert Review Log ({expertReviews.length})</h4>
            {expertReviews.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Role</th>
                      <th style={{ padding: '8px 10px' }}>Expert</th>
                      <th style={{ padding: '8px 10px' }}>Finding</th>
                      <th style={{ padding: '8px 10px' }}>AI Agreement</th>
                      <th style={{ padding: '8px 10px' }}>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {expertReviews.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 10px' }}>{r.role}</td>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.expert}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12 }}>{r.finding}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={checkBadge(r.agree_with_ai === 'agree' ? 'pass' : 'fail')}>
                            {r.agree_with_ai || '--'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{r.note || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', padding: 30 }}>No expert reviews recorded yet</div>
            )}
          </div>

          {/* Clinical decisions table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Clinical Decision Log ({decisions.length})</h4>
            {decisions.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>AI Prediction</th>
                      <th style={{ padding: '8px 10px' }}>Confidence</th>
                      <th style={{ padding: '8px 10px' }}>Agreement</th>
                      <th style={{ padding: '8px 10px' }}>Final Decision</th>
                      <th style={{ padding: '8px 10px' }}>Reviewer</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisions.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 10px' }}>{r.ai_prediction}</td>
                        <td style={{ padding: '8px 10px' }}>{r.ai_confidence != null ? `${(r.ai_confidence * 100).toFixed(0)}%` : '--'}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={checkBadge(r.neurologist_agreement === 'Yes' ? 'pass' : 'fail')}>
                            {r.neurologist_agreement || '--'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.final_decision}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12 }}>{r.reviewer}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', padding: 30 }}>No clinical decisions recorded yet</div>
            )}
          </div>
        </>
      )}

      {/* ═══ GOVERNANCE TAB ═══ */}
      {tab === 'governance' && (
        <>
          {/* Governance checklist */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Governance Readiness Checklist</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {checklist.map((c, i) => (
                <div key={i} style={{
                  border: `1px solid ${c.status === 'pass' ? '#bbf7d0' : '#fecaca'}`,
                  borderRadius: 8, padding: 14,
                  background: c.status === 'pass' ? '#f0fdf4' : '#fef2f2'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                    <span style={{ fontSize: 16 }}>{c.status === 'pass' ? '\u2705' : '\u274C'}</span>
                    <span style={{ fontWeight: 600, color: '#1e293b', fontSize: 13 }}>{c.requirement}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{c.detail}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 14, fontSize: 12, color: '#64748b' }}>
              {checklist.filter(c => c.status === 'pass').length}/{checklist.length} checks passing
            </div>
          </div>

          {/* Component audit trail */}
          {componentAudit.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Component Audit Trail</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Action</th>
                      <th style={{ padding: '8px 10px' }}>Operations</th>
                      <th style={{ padding: '8px 10px' }}>Actors</th>
                      <th style={{ padding: '8px 10px' }}>Last Activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {componentAudit.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.component}</td>
                        <td style={{ padding: '8px 10px' }}>{r.action}</td>
                        <td style={{ padding: '8px 10px' }}>{fmt(r.ops)}</td>
                        <td style={{ padding: '8px 10px' }}>{r.actors}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.last_activity ? new Date(r.last_activity).toLocaleDateString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
          <h4 style={{ margin: '0 0 14px', color: '#1e40af' }}>Compliance Metric Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: i < defs.definitions.length - 1 ? '1px solid #dbeafe' : 'none' }}>
              <div style={{ fontWeight: 600, color: '#1e3a5f', marginBottom: 2 }}>{d.metric}</div>
              <div style={{ color: '#334155', fontSize: 13, marginBottom: 2 }}>{d.description}</div>
              <div style={{ color: '#64748b', fontSize: 11 }}>Source: {d.source}</div>
            </div>
          ))}
        </div>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: clinical.db (hitl_reviews, expert_reviews, clinical_decisions, transaction_log, patients)
      </div>
    </div>
  )
}
