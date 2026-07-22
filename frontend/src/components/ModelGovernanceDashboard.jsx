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

const PIE_COLORS = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6', '#06b6d4']
const BAR_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4', '#10b981']

export default function ModelGovernanceDashboard() {
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
      axios.get(`${API}/api/model-governance/overview`),
      axios.get(`${API}/api/model-governance/breakdown`),
      axios.get(`${API}/api/model-governance/definitions`),
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
    { id: 'consultant', label: 'Consultant Matrix' },
    { id: 'signoff', label: 'Sign-Off Chain' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Model Governance dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No model governance data available.</div>

  const k = overview.kpis || {}

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Model Governance AI</h2>
        <Badge
          text={`Sign-Off: ${k.sign_off_rate != null ? k.sign_off_rate + '%' : 'N/A'}`}
          color={k.sign_off_rate >= 80 ? '#10b981' : k.sign_off_rate >= 50 ? '#f59e0b' : '#ef4444'}
        />
        <Badge
          text={`Override: ${k.override_rate != null ? k.override_rate + '%' : 'N/A'}`}
          color={k.override_rate <= 20 ? '#10b981' : k.override_rate <= 30 ? '#f59e0b' : '#ef4444'}
        />
        <Badge
          text={`Expert Agree: ${k.expert_agreement_rate != null ? k.expert_agreement_rate + '%' : 'N/A'}`}
          color={k.expert_agreement_rate >= 70 ? '#10b981' : k.expert_agreement_rate >= 50 ? '#f59e0b' : '#ef4444'}
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b', fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} k={k} />}
      {tab === 'consultant' && <ConsultantTab overview={overview} breakdown={breakdown} />}
      {tab === 'signoff' && <SignOffTab overview={overview} breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview, k }) {
  const kpis = [
    { label: 'Total Analyses', value: k.total_analyses, color: '#3b82f6' },
    { label: 'HITL Reviews', value: k.total_hitl_reviews, color: '#8b5cf6' },
    { label: 'Expert Reviews', value: k.total_expert_reviews, color: '#06b6d4' },
    { label: 'Clinical Decisions', value: k.total_clinical_decisions, color: '#10b981' },
    { label: 'Sign-Off Rate', value: k.sign_off_rate + '%', color: k.sign_off_rate >= 80 ? '#10b981' : '#f59e0b' },
    { label: 'Override Rate', value: k.override_rate + '%', color: k.override_rate <= 20 ? '#10b981' : '#ef4444' },
    { label: 'Expert Agreement', value: k.expert_agreement_rate + '%', color: k.expert_agreement_rate >= 70 ? '#10b981' : '#f59e0b' },
    { label: 'Avg Confidence', value: k.avg_confidence, color: '#3b82f6' },
  ]

  const signOffData = (overview.sign_off_chain || []).map(s => ({
    name: s.decision, value: s.count
  }))

  const lifecycleData = (overview.model_lifecycle || []).map(m => ({
    name: m.disease, count: m.analyses
  }))

  const consultantData = (overview.consultant_matrix || []).map(c => ({
    name: c.expert, agreement: c.agreement_rate, reviews: c.total_reviews
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((kp, i) => (
        <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
      ))}

      <Card title="Sign-Off Decision Distribution" span={2}>
        {signOffData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={signOffData} cx="50%" cy="50%" outerRadius={90} dataKey="value" nameKey="name" label={({ name, value }) => `${name}: ${value}`}>
                {signOffData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No sign-off data</div>}
      </Card>

      <Card title="Model Lifecycle — Analyses by Disease" span={2}>
        {lifecycleData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={lifecycleData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No lifecycle data</div>}
      </Card>

      <Card title="Consultant Agreement Rates" span={2}>
        {consultantData.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={consultantData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="agreement" fill="#10b981" radius={[4, 4, 0, 0]} name="Agreement %" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No consultant data</div>}
      </Card>

      <Card title="Governance Timeline" span={2}>
        {(overview.governance_timeline || []).length ? (
          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Date</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Type</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left' }}>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {(overview.governance_timeline || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{t.date}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <Badge
                        text={t.type.replace('_', ' ')}
                        color={t.type === 'hitl_review' ? '#8b5cf6' : t.type === 'clinical_decision' ? '#10b981' : '#3b82f6'}
                      />
                    </td>
                    <td style={{ padding: '6px 8px', fontFamily: 'monospace' }}>{t.patient_id}</td>
                    <td style={{ padding: '6px 8px' }}>
                      {t.decision || t.final_decision || (t.agree_with_ai === 'agree' ? 'Agree' : 'Disagree')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No timeline data</div>}
      </Card>
    </div>
  )
}

function ConsultantTab({ overview, breakdown }) {
  const matrix = overview.consultant_matrix || []
  const roleMatrix = breakdown?.role_agreement_matrix || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Consultant Matrix — Expert Agreement" span={2}>
        {matrix.length ? (
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px', textAlign: 'left' }}>Role</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Expert</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Reviews</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Agree</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Disagree</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Agreement Rate</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px' }}><Badge text={m.role} color="#3b82f6" /></td>
                  <td style={{ padding: '8px', fontWeight: 600 }}>{m.expert}</td>
                  <td style={{ padding: '8px', textAlign: 'center' }}>{m.total_reviews}</td>
                  <td style={{ padding: '8px', textAlign: 'center', color: '#10b981' }}>{m.agree}</td>
                  <td style={{ padding: '8px', textAlign: 'center', color: '#ef4444' }}>{m.disagree}</td>
                  <td style={{ padding: '8px', textAlign: 'center' }}>
                    <Badge
                      text={m.agreement_rate + '%'}
                      color={m.agreement_rate >= 70 ? '#10b981' : m.agreement_rate >= 50 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No consultant data</div>}
      </Card>

      <Card title="Role-Based Agreement Matrix" span={1}>
        {roleMatrix.length ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={roleMatrix}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="role" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="agreement_rate" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Agreement %" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No role data</div>}
      </Card>

      <Card title="Expert Review Detail" span={1}>
        {(breakdown?.expert_detail || []).length ? (
          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ padding: '4px 6px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left' }}>Expert</th>
                  <th style={{ padding: '4px 6px', textAlign: 'left' }}>Finding</th>
                  <th style={{ padding: '4px 6px', textAlign: 'center' }}>AI?</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.expert_detail || []).map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 6px', fontFamily: 'monospace' }}>{e.patient_id}</td>
                    <td style={{ padding: '4px 6px' }}>{e.expert}</td>
                    <td style={{ padding: '4px 6px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.finding}</td>
                    <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                      <Badge text={e.agree_with_ai} color={e.agree_with_ai === 'agree' ? '#10b981' : '#ef4444'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No expert reviews</div>}
      </Card>
    </div>
  )
}

function SignOffTab({ overview, breakdown }) {
  const hitl = breakdown?.hitl_detail || []
  const decisions = breakdown?.decision_chain || []
  const findings = breakdown?.component_findings || []
  const feedbackLog = breakdown?.feedback_log || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="HITL Sign-Off Reviews" span={2}>
        {hitl.length ? (
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px', textAlign: 'left' }}>ID</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>AI Prediction</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Decision</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Human Decision</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Reason</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {hitl.map((h, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px', fontFamily: 'monospace' }}>{h.id}</td>
                  <td style={{ padding: '8px', fontFamily: 'monospace' }}>{h.patient_id}</td>
                  <td style={{ padding: '8px' }}>{h.ai_prediction || '-'}</td>
                  <td style={{ padding: '8px' }}>
                    <Badge text={h.decision} color={h.decision === 'accept' ? '#10b981' : '#f59e0b'} />
                  </td>
                  <td style={{ padding: '8px' }}>{h.human_decision || '-'}</td>
                  <td style={{ padding: '8px' }}>{h.reason_code || '-'}</td>
                  <td style={{ padding: '8px', fontFamily: 'monospace', fontSize: 11 }}>{(h.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No HITL reviews</div>}
      </Card>

      <Card title="Clinical Decision Chain" span={2}>
        {decisions.length ? (
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>AI Prediction</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Confidence</th>
                <th style={{ padding: '8px', textAlign: 'center' }}>Neuro Agree</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Final Decision</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Reviewer</th>
                <th style={{ padding: '8px', textAlign: 'left' }}>Note</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px', fontFamily: 'monospace' }}>{d.patient_id}</td>
                  <td style={{ padding: '8px' }}>{d.ai_prediction}</td>
                  <td style={{ padding: '8px', textAlign: 'center' }}>{d.ai_confidence}</td>
                  <td style={{ padding: '8px', textAlign: 'center' }}>
                    <Badge text={d.neurologist_agreement} color={d.neurologist_agreement === 'Yes' ? '#10b981' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '8px', fontWeight: 600 }}>{d.final_decision}</td>
                  <td style={{ padding: '8px' }}>{d.reviewer}</td>
                  <td style={{ padding: '8px', fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No clinical decisions</div>}
      </Card>

      <Card title="Component Findings" span={1}>
        {findings.length ? (
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Component</th>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Doctor</th>
                <th style={{ padding: '4px 6px', textAlign: 'center' }}>AI?</th>
              </tr>
            </thead>
            <tbody>
              {findings.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 6px', fontFamily: 'monospace' }}>{f.patient_id}</td>
                  <td style={{ padding: '4px 6px' }}>{f.component}</td>
                  <td style={{ padding: '4px 6px' }}>{f.doctor || '-'}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'center' }}>
                    <Badge text={f.agree_with_ai} color={f.agree_with_ai === 'agree' ? '#10b981' : '#ef4444'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No findings</div>}
      </Card>

      <Card title="Clinician Feedback Log" span={1}>
        {feedbackLog.length ? (
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Role</th>
                <th style={{ padding: '4px 6px', textAlign: 'center' }}>Rating</th>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Correction</th>
                <th style={{ padding: '4px 6px', textAlign: 'left' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {feedbackLog.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 6px' }}>{f.role}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'center', fontWeight: 700,
                    color: f.rating >= 4 ? '#10b981' : f.rating >= 3 ? '#f59e0b' : '#ef4444' }}>{f.rating}/5</td>
                  <td style={{ padding: '4px 6px' }}>{f.correction || '-'}</td>
                  <td style={{ padding: '4px 6px', fontFamily: 'monospace' }}>{(f.created_at || '').slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No feedback</div>}
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const profiles = breakdown?.patient_profiles || []
  const governed = profiles.filter(p => p.governance_complete)
  const ungoverned = profiles.filter(p => !p.governance_complete)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title={`Governance Coverage: ${governed.length}/${profiles.length} patients`} span={2}>
        {profiles.length ? (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Governed', value: governed.length },
                  { name: 'Ungoverned', value: ungoverned.length },
                ]}
                cx="50%" cy="50%" outerRadius={70} dataKey="value" nameKey="name"
                label={({ name, value }) => `${name}: ${value}`}
              >
                <Cell fill="#10b981" />
                <Cell fill="#94a3b8" />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No patient data</div>}
      </Card>

      <Card title="Patient Governance Profiles" span={2}>
        {profiles.length ? (
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px', textAlign: 'left' }}>Patient ID</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Analyses</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>HITL</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Expert</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Decisions</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Findings</th>
                  <th style={{ padding: '8px', textAlign: 'left' }}>Diseases</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Governed</th>
                </tr>
              </thead>
              <tbody>
                {profiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontFamily: 'monospace', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{p.analyses}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{p.hitl_reviews}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{p.expert_reviews}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{p.clinical_decisions}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>{p.component_findings}</td>
                    <td style={{ padding: '8px', fontSize: 11 }}>{(p.diseases || []).join(', ') || '-'}</td>
                    <td style={{ padding: '8px', textAlign: 'center' }}>
                      <Badge text={p.governance_complete ? 'Yes' : 'No'} color={p.governance_complete ? '#10b981' : '#94a3b8'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No patient profiles</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  const sections = definitions?.sections || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((s, si) => (
        <Card key={si} title={s.title}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <tbody>
              {(s.items || []).map((item, ii) => (
                <tr key={ii} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, verticalAlign: 'top', whiteSpace: 'nowrap', width: '20%' }}>{item.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569', lineHeight: 1.5 }}>{item.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </div>
  )
}
