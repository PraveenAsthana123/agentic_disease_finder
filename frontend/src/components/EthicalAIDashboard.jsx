import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const OK = '#10b981'
const WARN = '#f59e0b'
const ERR = '#ef4444'

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

function scoreColor(s) {
  if (s == null) return '#64748b'
  if (s >= 75) return OK
  if (s >= 50) return WARN
  return ERR
}

function statusBadge(status) {
  if (status === 'met') return <Badge text="Met" color={OK} />
  if (status === 'gap') return <Badge text="Gap" color={WARN} />
  return <Badge text={status} color="#64748b" />
}

export default function EthicalAIDashboard() {
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
      axios.get(`${API}/api/ethical-ai/overview`),
      axios.get(`${API}/api/ethical-ai/breakdown`),
      axios.get(`${API}/api/ethical-ai/definitions`),
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
    { id: 'fairness', label: 'Fairness & Bias' },
    { id: 'guardrails', label: 'Guardrails & Oversight' },
    { id: 'principles', label: 'Ethical Principles' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Ethical AI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: ERR }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const sc = overview.score_components || {}

  // Build chart data for severity by gender
  const sevData = (breakdown?.severity_by_gender || []).reduce((acc, r) => {
    const existing = acc.find(x => x.level === r.level)
    if (existing) {
      existing[r.gender] = r.count
    } else {
      acc.push({ level: r.level, [r.gender]: r.count })
    }
    return acc
  }, [])

  // Build fairness group chart
  const fairnessGroupData = Object.entries(breakdown?.fairness_by_group || {}).map(([grp, d]) => ({
    group: grp, selection_rate: +(d.selection_rate * 100).toFixed(1), count: d.count,
  }))

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Ethical AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Fairness analysis, guardrail enforcement, bias monitoring &amp; bioethics principle adherence
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Ethics Score" value={overview.composite_ethics_score} color={scoreColor(overview.composite_ethics_score)} sub={overview.score_weights} /></Card>
            <Card><KPI label="Fairness" value={sc.fairness} color={scoreColor(sc.fairness)} sub={`Gate: ${overview.fairness?.gate}`} /></Card>
            <Card><KPI label="Transparency" value={`${sc.transparency}%`} color={scoreColor(sc.transparency)} sub="XAI coverage" /></Card>
            <Card><KPI label="Oversight" value={sc.oversight} color={scoreColor(sc.oversight)} sub="Human review ratio" /></Card>
            <Card><KPI label="Guardrails" value={sc.guardrails} color={scoreColor(sc.guardrails)} sub="Enforcement score" /></Card>
            <Card><KPI label="DPD" value={overview.fairness?.dpd != null ? overview.fairness.dpd.toFixed(4) : '--'} color={overview.fairness?.dpd < 0.2 ? OK : WARN} sub={`Attr: ${overview.fairness?.protected_attribute}`} /></Card>
            <Card><KPI label="HITL Reviews" value={overview.oversight?.hitl_total || 0} color="#3b82f6" sub={`${overview.oversight?.hitl_overrides || 0} overrides`} /></Card>
            <Card><KPI label="Expert Reviews" value={overview.oversight?.expert_reviews || 0} color="#8b5cf6" sub={`${overview.oversight?.expert_agree || 0} agree`} /></Card>
          </div>

          {/* Score breakdown */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Score Components">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {Object.entries(sc).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ width: 100, fontSize: 13, fontWeight: 600, color: '#475569', textTransform: 'capitalize' }}>{k}</div>
                    <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 6, height: 20, overflow: 'hidden' }}>
                      <div style={{ width: `${Math.min(v, 100)}%`, height: '100%', background: scoreColor(v), borderRadius: 6, transition: 'width 0.5s' }} />
                    </div>
                    <div style={{ width: 50, textAlign: 'right', fontSize: 13, fontWeight: 700, color: scoreColor(v) }}>{v}</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card title="Monitoring Activity">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 0', color: '#475569' }}>Fairness runs</td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontWeight: 600 }}>{overview.fairness?.fairness_runs || 0}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 0', color: '#475569' }}>Consistency checks</td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontWeight: 600 }}>{overview.monitoring?.consistency_checks || 0}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 0', color: '#475569' }}>Drift checks</td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontWeight: 600 }}>{overview.monitoring?.drift_checks || 0}</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 0', color: '#475569' }}>Total analyses</td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontWeight: 600 }}>{overview.transparency?.total_analyses || 0}</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '8px 0', color: '#475569' }}>Clinical decisions</td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontWeight: 600 }}>{overview.oversight?.clinical_decisions || 0}</td>
                  </tr>
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}

      {/* ── FAIRNESS & BIAS TAB ── */}
      {tab === 'fairness' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Selection Rate by Group (%)">
              {fairnessGroupData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={fairnessGroupData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="group" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="selection_rate" fill="#3b82f6" radius={[4,4,0,0]} name="Selection Rate %" />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No fairness group data</div>}
              <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>
                DPD = {overview.fairness?.dpd != null ? overview.fairness.dpd.toFixed(4) : 'N/A'} | Gate: <Badge text={overview.fairness?.gate || 'N/A'} color={overview.fairness?.gate === 'PASS' ? OK : ERR} /> | Library: {overview.fairness?.library}
              </div>
            </Card>
            <Card title="Confidence by Gender">
              {(breakdown?.confidence_by_gender || []).length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={breakdown.confidence_by_gender}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="gender" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="mean_confidence" fill="#8b5cf6" radius={[4,4,0,0]} name="Mean Confidence" />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence data by gender</div>}
              <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>
                Confidence parity across genders indicates unbiased model scoring
              </div>
            </Card>
          </div>
          <Card title="Assessment Severity Distribution by Gender" span={2}>
            {sevData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={sevData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="Female" fill="#ec4899" radius={[4,4,0,0]} />
                  <Bar dataKey="Male" fill="#3b82f6" radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No severity data</div>}
          </Card>
        </>
      )}

      {/* ── GUARDRAILS & OVERSIGHT TAB ── */}
      {tab === 'guardrails' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Guardrail Event Log (Council)">
              {(breakdown?.guardrail_events || []).length > 0 ? (
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Action</th>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Detail</th>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.guardrail_events.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 4px' }}>
                          <Badge text={e.action} color={e.action === 'blocked' ? ERR : OK} />
                        </td>
                        <td style={{ padding: '6px 4px', color: '#475569', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                        <td style={{ padding: '6px 4px', color: '#94a3b8', fontSize: 11 }}>{e.ts}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No guardrail events</div>}
            </Card>
            <Card title="HITL Decision Log">
              {(breakdown?.hitl_decisions || []).length > 0 ? (
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>AI</th>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Decision</th>
                      <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Human</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.hitl_decisions.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 4px', fontWeight: 600 }}>{d.patient_id}</td>
                        <td style={{ padding: '6px 4px', color: '#475569' }}>{d.ai_prediction}</td>
                        <td style={{ padding: '6px 4px' }}>
                          <Badge text={d.decision} color={d.decision === 'accept' ? OK : WARN} />
                        </td>
                        <td style={{ padding: '6px 4px', color: '#475569' }}>{d.human_decision || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No HITL decisions</div>}
            </Card>
          </div>
          <Card title="Expert Reviews" span={2}>
            {(breakdown?.expert_reviews || []).length > 0 ? (
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Role</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Expert</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Finding</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Agree</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.expert_reviews.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 4px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 4px', color: '#475569' }}>{r.role}</td>
                      <td style={{ padding: '6px 4px', color: '#475569' }}>{r.expert}</td>
                      <td style={{ padding: '6px 4px', color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.finding}</td>
                      <td style={{ padding: '6px 4px' }}>
                        <Badge text={r.agree_with_ai} color={r.agree_with_ai === 'agree' ? OK : WARN} />
                      </td>
                      <td style={{ padding: '6px 4px', color: '#94a3b8' }}>{r.note || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No expert reviews</div>}
          </Card>
        </>
      )}

      {/* ── ETHICAL PRINCIPLES TAB ── */}
      {tab === 'principles' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {(breakdown?.ethical_principles || []).map((p, i) => (
            <Card key={i} title={p.principle}>
              <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>{p.description}</p>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Indicator</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Value</th>
                    <th style={{ textAlign: 'left', padding: '6px 4px', color: '#64748b' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {p.indicators.map((ind, j) => (
                    <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 4px', color: '#475569' }}>{ind.name}</td>
                      <td style={{ padding: '6px 4px', fontWeight: 600 }}>{ind.value}</td>
                      <td style={{ padding: '6px 4px' }}>{statusBadge(ind.status)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {(definitions.sections || []).map((sec, i) => (
            <Card key={i} title={sec.title}>
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {sec.items.map((item, j) => (
                    <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 4px', fontWeight: 600, color: '#334155', width: '30%', verticalAlign: 'top' }}>{item.term}</td>
                      <td style={{ padding: '8px 4px', color: '#475569' }}>{item.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
