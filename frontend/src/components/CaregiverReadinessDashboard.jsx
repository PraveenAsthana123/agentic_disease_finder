import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const READINESS_COLORS = {
  high: '#10b981',
  moderate: '#f59e0b',
  low: '#ef4444',
  ready: '#10b981',
  'not ready': '#ef4444',
}

const BURNOUT_COLORS = {
  low: '#10b981',
  moderate: '#f59e0b',
  high: '#ef4444',
  severe: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'profiles', label: 'Caregiver Profiles' },
  { id: 'matrix', label: 'Readiness Matrix' },
  { id: 'burnout', label: 'Burnout Alerts' },
  { id: 'training', label: 'Training Gaps' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CaregiverReadinessDashboard() {
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
      axios.get(`${API_URL}/api/caregiver-readiness/overview`),
      axios.get(`${API_URL}/api/caregiver-readiness/breakdown`),
      axios.get(`${API_URL}/api/caregiver-readiness/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Caregiver Readiness data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No caregiver readiness data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Caregiver Readiness Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Caregiver preparedness, burnout risk, training gaps, and readiness assessments
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'profiles' && <ProfilesTab breakdown={breakdown} />}
      {tab === 'matrix' && <MatrixTab breakdown={breakdown} />}
      {tab === 'burnout' && <BurnoutTab breakdown={breakdown} />}
      {tab === 'training' && <TrainingTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const readinessDist = overview.readiness_distribution || []
  const burnoutDist = overview.burnout_distribution || []
  const roleDist = overview.role_distribution || []
  const trainingCoverage = overview.training_topic_coverage || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Readiness Distribution Pie */}
      <Card title="Readiness Distribution" span={2}>
        {readinessDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={readinessDist} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={100}
                label={({ level, count }) => `${level} (${count})`}>
                {readinessDist.map((entry, i) => <Cell key={i} fill={entry.color || COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No readiness data</div>}
      </Card>

      {/* Burnout Distribution Bar */}
      <Card title="Burnout Distribution" span={2}>
        {burnoutDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={burnoutDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {burnoutDist.map((entry, i) => (
                  <Cell key={i} fill={entry.color || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No burnout data</div>}
      </Card>

      {/* Role Distribution Pie */}
      <Card title="Role Distribution" span={2}>
        {roleDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={roleDist} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={100}
                label={({ role, count }) => `${role} (${count})`}>
                {roleDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No role data</div>}
      </Card>

      {/* Training Topic Coverage Bar */}
      <Card title="Training Topic Coverage" span={2}>
        {trainingCoverage.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={trainingCoverage} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="topic" type="category" tick={{ fontSize: 10 }} width={160} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No training coverage data</div>}
      </Card>
    </div>
  )
}

function ProfilesTab({ breakdown }) {
  const profiles = breakdown?.caregiver_profiles || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Caregiver Profiles (${profiles.length})`}>
        {profiles.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Patient ID', 'Caregiver', 'Role', 'Experience (yrs)', 'Training', 'First Aid', 'Rescue Med', 'Confidence', 'Burnout', 'Readiness Level'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><Badge text={p.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.caregiver_name || p.caregiver}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{p.role}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.experience_years ?? p.experience}</td>
                    <td style={{ padding: '6px 10px' }}>
                      {p.training_completed ? <span style={{ color: '#10b981' }}>Yes</span> : <span style={{ color: '#ef4444' }}>No</span>}
                    </td>
                    <td style={{ padding: '6px 10px' }}>
                      {p.first_aid_certified ? <span style={{ color: '#10b981' }}>Yes</span> : <span style={{ color: '#ef4444' }}>No</span>}
                    </td>
                    <td style={{ padding: '6px 10px' }}>
                      {p.rescue_med_trained ? <span style={{ color: '#10b981' }}>Yes</span> : <span style={{ color: '#ef4444' }}>No</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.confidence_score ?? p.confidence}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={p.burnout_level || p.burnout || 'N/A'} color={BURNOUT_COLORS[(p.burnout_level || p.burnout || '').toLowerCase()] || '#6b7280'} />
                    </td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={p.readiness_level || 'N/A'} color={READINESS_COLORS[(p.readiness_level || '').toLowerCase()] || '#6b7280'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 20, textAlign: 'center' }}>No caregiver profiles available</div>}
      </Card>
    </div>
  )
}

function MatrixTab({ breakdown }) {
  const matrix = breakdown?.readiness_matrix || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Readiness Matrix (${matrix.length})`}>
        {matrix.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Caregiver', 'Training', 'First Aid', 'Rescue Med', 'Safety Plan', 'Action Plan', 'Overall'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Caregiver' ? 'left' : 'center', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.caregiver_name || m.caregiver}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      {m.training ? <span style={{ color: '#10b981', fontWeight: 700 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontWeight: 700 }}>&#10007;</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      {m.first_aid ? <span style={{ color: '#10b981', fontWeight: 700 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontWeight: 700 }}>&#10007;</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      {m.rescue_med ? <span style={{ color: '#10b981', fontWeight: 700 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontWeight: 700 }}>&#10007;</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      {m.safety_plan ? <span style={{ color: '#10b981', fontWeight: 700 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontWeight: 700 }}>&#10007;</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      {m.action_plan ? <span style={{ color: '#10b981', fontWeight: 700 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontWeight: 700 }}>&#10007;</span>}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      <Badge text={m.overall_readiness || m.overall || 'N/A'} color={READINESS_COLORS[(m.overall_readiness || m.overall || '').toLowerCase()] || '#6b7280'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13, padding: 20, textAlign: 'center' }}>No readiness matrix data available</div>}
      </Card>
    </div>
  )
}

function BurnoutTab({ breakdown }) {
  const alerts = breakdown?.burnout_risk_alerts || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Burnout Risk Alerts (${alerts.length})`}>
        {alerts.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#fef2f2' }}>
                  {['Caregiver', 'Patient ID', 'Burnout Score', 'Stress', 'Sleep Quality', 'Risk Factors'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Burnout Score' ? 'right' : 'left', borderBottom: '1px solid #fecaca', color: '#991b1b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {alerts.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fef2f2' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.caregiver_name || a.caregiver}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={a.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600, color: '#ef4444' }}>{a.burnout_score}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={a.stress_level || a.stress || 'N/A'} color={a.stress_level === 'high' || a.stress === 'high' ? '#ef4444' : '#f59e0b'} />
                    </td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{a.sleep_quality || 'N/A'}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(a.risk_factors || []).map((rf, j) => (
                          <Badge key={j} text={rf} color="#ef4444" />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#10b981', fontSize: 13, padding: 20, textAlign: 'center' }}>No burnout risk alerts — all caregivers within safe range</div>}
      </Card>
    </div>
  )
}

function TrainingTab({ breakdown }) {
  const gaps = breakdown?.training_gaps || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Training Gaps (${gaps.length})`}>
        {gaps.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Caregiver', 'Missing Topics'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {gaps.map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.caregiver_name || g.caregiver}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(g.missing_topics || []).map((topic, j) => (
                          <Badge key={j} text={topic} color="#f59e0b" />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#10b981', fontSize: 13, padding: 20, textAlign: 'center' }}>No training gaps identified — all caregivers fully trained</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const concepts = definitions.concepts || []
  const qualityMetrics = definitions.quality_metrics || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Concepts */}
      <Card title={`Caregiver Readiness Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${qualityMetrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {qualityMetrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Standard</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.ref}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Strategy</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.strategy}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
