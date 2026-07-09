import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  const colors = {
    red: { bg: '#fee2e2', text: '#991b1b' },
    amber: { bg: '#fef3c7', text: '#92400e' },
    green: { bg: '#dcfce7', text: '#166534' },
    blue: { bg: '#dbeafe', text: '#1e40af' },
    purple: { bg: '#f3e8ff', text: '#6b21a8' },
  }
  const c = colors[color] || colors.blue
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 9999,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.text, marginRight: 4, marginBottom: 4
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')
const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316', '#14b8a6']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'consultants', label: 'Consultant Matrix' },
  { id: 'audit', label: 'Audit Trail' },
  { id: 'health', label: 'Governance Health' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AIGovernanceDashboard() {
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
      axios.get(`${API_URL}/api/ai-governance/overview`),
      axios.get(`${API_URL}/api/ai-governance/breakdown`),
      axios.get(`${API_URL}/api/ai-governance/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI governance data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  /* ── Overview Tab ───────────────────────────────── */
  const renderOverview = () => {
    const s = overview?.summary || {}
    const eventBreakdown = overview?.governance_event_breakdown || []
    const feedbackDist = overview?.feedback_distribution || []
    const confDist = breakdown?.confidence_distribution || []

    return (
      <>
        {/* KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Total Decisions" value={fmt(s.total_decisions)} color="#3b82f6" /></Card>
          <Card><KPI label="Agreement Rate" value={pct(s.agreement_rate)} color="#10b981" /></Card>
          <Card><KPI label="Expert Reviews" value={fmt(s.expert_reviews)} sub={`${s.expert_agreement_pct}% agree`} color="#8b5cf6" /></Card>
          <Card><KPI label="HITL Reviews" value={fmt(s.hitl_reviews)} sub={`${s.hitl_overrides} overrides`} color="#f59e0b" /></Card>
          <Card><KPI label="Override Rate" value={pct(s.override_rate)} color="#ef4444" /></Card>
          <Card><KPI label="Avg Feedback" value={fmt(s.avg_feedback_rating)} sub="out of 5" color="#06b6d4" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Governance Event Breakdown */}
          <Card title="Governance Events by Action">
            {eventBreakdown.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={eventBreakdown} dataKey="cnt" nameKey="action" cx="50%" cy="50%"
                    outerRadius={90} label={({ action, cnt }) => `${action}: ${cnt}`}>
                    {eventBreakdown.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No governance events yet</div>}
          </Card>

          {/* Confidence Distribution */}
          <Card title="Decision Confidence Distribution">
            {confDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={confDist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="band" tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="cnt" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                    {confDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No confidence data</div>}
          </Card>
        </div>

        {/* Decision Trail */}
        <Card title="Recent Decision Trail" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'AI Prediction', 'Confidence', 'Agreement', 'Final Decision', 'Reviewer', 'Date'].map(h =>
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {(overview?.decision_trail || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{r.ai_prediction}</td>
                    <td style={{ padding: '8px 10px' }}>{pct(Math.round((r.confidence || 0) * 100))}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={r.agreement || '--'} color={r.agreement === 'Yes' ? 'green' : 'red'} />
                    </td>
                    <td style={{ padding: '8px 10px' }}>{r.final_decision || '--'}</td>
                    <td style={{ padding: '8px 10px' }}>{r.reviewer || '--'}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8' }}>{r.created_at?.split('T')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Consultant Matrix Tab ─────────────────────── */
  const renderConsultants = () => {
    const matrix = breakdown?.consultant_matrix || []
    const tierData = [
      { name: 'Tier 1 (Mandatory)', value: matrix.filter(c => c.tier === 1).length },
      { name: 'Tier 2 (Advisory)', value: matrix.filter(c => c.tier === 2).length },
    ]
    const radarData = matrix.map(c => ({
      role: c.name.split(' ')[0],
      tasks: c.task_count || 0,
      compliance: c.compliance_doc_count || 0,
      challenges: c.challenge_count || 0,
    }))

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Total Consultants" value={matrix.length} color="#3b82f6" /></Card>
          <Card><KPI label="Mandatory (Tier 1)" value={matrix.filter(c => c.mandatory).length} color="#ef4444" /></Card>
          <Card><KPI label="Advisory (Tier 2)" value={matrix.filter(c => !c.mandatory).length} color="#f59e0b" /></Card>
          <Card><KPI label="Engagement Model" value={breakdown?.engagement_model || '--'} color="#8b5cf6" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <Card title="Tier Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={tierData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {tierData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Role Capability Radar">
            {radarData.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="role" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis />
                  <Radar name="Tasks" dataKey="tasks" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  <Radar name="Compliance" dataKey="compliance" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                  <Radar name="Challenges" dataKey="challenges" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>
        </div>

        {/* Full consultant table */}
        <Card title="Consultant Matrix Detail" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Role', 'Tier', 'Mandatory', 'Objective', 'Tasks', 'Compliance Docs', 'Challenges'].map(h =>
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {matrix.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.name}</td>
                    <td style={{ padding: '8px 10px' }}><Badge text={`Tier ${c.tier}`} color={c.tier === 1 ? 'red' : 'amber'} /></td>
                    <td style={{ padding: '8px 10px' }}><Badge text={c.mandatory ? 'Yes' : 'No'} color={c.mandatory ? 'green' : 'blue'} /></td>
                    <td style={{ padding: '8px 10px', fontSize: 12 }}>{c.objective}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{c.task_count}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{c.compliance_doc_count}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{c.challenge_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Audit Trail Tab ───────────────────────────── */
  const renderAudit = () => {
    const reviews = overview?.review_panel || []
    const hitl = overview?.hitl_detail || []
    const roleBreak = breakdown?.role_breakdown || []

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Expert Review Panel */}
          <Card title="Expert Review Panel">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient', 'Role', 'Expert', 'Finding', 'Agrees w/ AI', 'Date'].map(h =>
                      <th key={h} style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {reviews.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{r.role}</td>
                      <td style={{ padding: '6px 8px' }}>{r.expert}</td>
                      <td style={{ padding: '6px 8px', fontSize: 12, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.finding}</td>
                      <td style={{ padding: '6px 8px' }}>
                        <Badge text={r.agree_with_ai} color={r.agree_with_ai === 'agree' ? 'green' : 'red'} />
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{r.created_at?.split('T')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* HITL Detail */}
          <Card title="Human-in-the-Loop (HITL) Log">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient', 'AI Prediction', 'Decision', 'Human Override', 'Reason', 'Date'].map(h =>
                      <th key={h} style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {hitl.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px' }}>{r.ai_prediction}</td>
                      <td style={{ padding: '6px 8px' }}>
                        <Badge text={r.decision} color={r.decision === 'accept' ? 'green' : 'amber'} />
                      </td>
                      <td style={{ padding: '6px 8px' }}>{r.human_decision || '--'}</td>
                      <td style={{ padding: '6px 8px' }}>{r.reason_code || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{r.created_at?.split('T')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Role Breakdown */}
        <Card title="Review Breakdown by Role">
          {roleBreak.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={roleBreak}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="cnt" name="Total Reviews" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="agreed" name="Agreed w/ AI" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No role data yet</div>}
        </Card>
      </>
    )
  }

  /* ── Governance Health Tab ──────────────────────── */
  const renderHealth = () => {
    const scores = breakdown?.health_scores || {}
    const healthData = Object.entries(scores).map(([k, v]) => ({
      metric: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      score: typeof v === 'number' ? v : 0,
      fullMark: 100,
    }))
    const useCases = breakdown?.use_case_register || []
    const riskDist = [
      { name: 'High Risk', value: useCases.filter(u => u.risk_class === 'high').length },
      { name: 'Medium Risk', value: useCases.filter(u => u.risk_class === 'medium').length },
      { name: 'Low Risk', value: useCases.filter(u => u.risk_class === 'low').length },
    ].filter(d => d.value > 0)

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Health Radar */}
          <Card title="Governance Health Scores">
            {healthData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={healthData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar name="Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.4} />
                  <Tooltip formatter={v => `${v}%`} />
                </RadarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No health data</div>}
          </Card>

          {/* Risk Distribution */}
          <Card title="Use-Case Risk Classification">
            {riskDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={riskDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                    <Cell fill="#ef4444" />
                    <Cell fill="#f59e0b" />
                    <Cell fill="#10b981" />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No risk data</div>}
          </Card>
        </div>

        {/* Health Score Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
          {healthData.map((h, i) => (
            <Card key={i}>
              <KPI
                label={h.metric}
                value={pct(Math.round(h.score))}
                color={h.score >= 80 ? '#10b981' : h.score >= 50 ? '#f59e0b' : '#ef4444'}
              />
            </Card>
          ))}
        </div>

        {/* Use-Case Register */}
        <Card title="Use-Case Register" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Role', 'Risk Class', 'Tier', 'Mandatory', 'Tasks', 'Compliance Docs'].map(h =>
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {useCases.map((u, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{u.role}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={u.risk_class} color={u.risk_class === 'high' ? 'red' : u.risk_class === 'medium' ? 'amber' : 'green'} />
                    </td>
                    <td style={{ padding: '8px 10px' }}><Badge text={`Tier ${u.tier}`} color={u.tier === 1 ? 'purple' : 'blue'} /></td>
                    <td style={{ padding: '8px 10px' }}>{u.mandatory ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{u.tasks}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>{u.compliance_docs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Definitions Tab ───────────────────────────── */
  const renderDefinitions = () => {
    const sections = definitions?.sections || []
    return (
      <>
        {sections.map((sec, si) => (
          <Card key={si} title={sec.title} span={2}>
            <div style={{ display: 'grid', gap: 12 }}>
              {(sec.items || []).map((item, ii) => (
                <div key={ii} style={{ borderBottom: '1px solid #f1f5f9', paddingBottom: 10 }}>
                  <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{item.definition}</div>
                  {item.clinical_relevance && (
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
                      <strong>Clinical relevance:</strong> {item.clinical_relevance}
                    </div>
                  )}
                  {item.regulatory_refs && (
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                      {item.regulatory_refs.map((r, ri) => <Badge key={ri} text={r} color="purple" />)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        ))}
      </>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>AI Governance Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Decision audit trail, expert reviews, HITL oversight, consultant matrix, and governance health
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '10px 18px', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gap: 16 }}>
        {tab === 'overview' && renderOverview()}
        {tab === 'consultants' && renderConsultants()}
        {tab === 'audit' && renderAudit()}
        {tab === 'health' && renderHealth()}
        {tab === 'definitions' && renderDefinitions()}
      </div>
    </div>
  )
}
