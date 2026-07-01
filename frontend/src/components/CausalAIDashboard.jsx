import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line, ScatterChart, Scatter, ZAxis
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

const STRENGTH_COLORS = {
  strong: '#ef4444', moderate: '#f59e0b', weak: '#94a3b8',
  high: '#ef4444', medium: '#f59e0b', low: '#10b981'
}

const CAT_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899', '#64748b']

const SEV_COLORS = { Mild: '#f59e0b', Moderate: '#f97316', Severe: '#ef4444', Unknown: '#94a3b8' }

const EFF_COLORS = { good: '#10b981', moderate: '#f59e0b', poor: '#ef4444' }

export default function CausalAIDashboard() {
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
      axios.get(`${API}/causal-ai/overview`),
      axios.get(`${API}/causal-ai/breakdown`),
      axios.get(`${API}/causal-ai/definitions`),
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
    { id: 'pathways', label: 'Causal Pathways' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'interventions', label: 'Interventions' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Causal AI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const k = overview.kpis || {}

  /* ── Overview Tab ──────────────────────────────────────────────────── */
  const renderOverview = () => {
    const medData = (overview.medication_seizure_links || []).map(m => ({
      name: m.drug, patients: m.patients, seizures: m.total_seizures,
      avg: m.avg_seizures_per_patient, strength: m.causal_strength
    }))

    const triggerData = (overview.trigger_chains || []).map(t => ({
      name: t.trigger, count: t.count, risk: t.risk_level,
      avg_duration: t.avg_duration_sec
    }))

    const ageSevData = (overview.age_severity || []).map(a => ({
      name: a.group, seizures: a.seizures, duration: a.avg_duration
    }))

    const genderData = (overview.gender_analysis || []).filter(g => g.patients > 0).map(g => ({
      name: g.gender || 'Unknown', value: g.patients, seizures: g.seizures, rate: g.seizure_rate
    }))

    const graphNodes = overview.causal_graph?.nodes || []
    const nodeTypes = {}
    graphNodes.forEach(n => { nodeTypes[n.type] = (nodeTypes[n.type] || 0) + 1 })
    const nodeTypeData = Object.entries(nodeTypes).map(([k, v]) => ({ name: k, value: v }))

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Total Patients" value={k.total_patients} color="#3b82f6" /></Card>
          <Card><KPI label="Patients with Seizures" value={k.patients_with_seizures} sub={`${Math.round(k.patients_with_seizures / k.total_patients * 100)}% of cohort`} color="#8b5cf6" /></Card>
          <Card><KPI label="Seizure Events" value={k.total_seizure_events} color="#ef4444" /></Card>
          <Card><KPI label="Causal Pathways" value={k.total_causal_pathways} sub={`${k.strong_pathways} strong`} color="#10b981" /></Card>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Medications Analysed" value={k.medications_analysed} color="#06b6d4" /></Card>
          <Card><KPI label="Triggers Identified" value={k.triggers_identified} color="#f59e0b" /></Card>
          <Card><KPI label="Strong Pathways" value={k.strong_pathways} color="#ef4444" /></Card>
          <Card><KPI label="Graph Nodes" value={k.causal_graph_nodes} color="#64748b" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <Card title="Medication → Seizure Links" span={1}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={medData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="patients" fill="#3b82f6" name="Patients" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Seizure Triggers" span={1}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={triggerData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" name="Occurrences" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Age Group → Seizure Burden">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ageSevData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="seizures" fill="#8b5cf6" name="Seizures" radius={[4, 4, 0, 0]} />
                <Bar dataKey="duration" fill="#06b6d4" name="Avg Duration (s)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Gender Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={genderData} cx="50%" cy="50%" outerRadius={80} dataKey="value"
                  label={({ name, value }) => `${name} (${value})`}>
                  {genderData.map((_, i) => <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(v, n, p) => [v, n === 'value' ? 'Patients' : n]} />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </>
    )
  }

  /* ── Causal Pathways Tab ───────────────────────────────────────────── */
  const renderPathways = () => {
    const medLinks = overview.medication_seizure_links || []
    const triggers = overview.trigger_chains || []
    const graph = overview.causal_graph || {}

    const medStrengthData = medLinks.map(m => ({
      name: m.drug, patients: m.patients,
      fill: STRENGTH_COLORS[m.causal_strength] || '#94a3b8'
    }))

    return (
      <>
        <Card title="Medication → Seizure Causal Pathways" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Drug', 'Patients', 'Total Seizures', 'Avg Seizures/Patient', 'Avg Dose (mg)', 'Strength'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {medLinks.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{m.drug}</td>
                    <td style={{ padding: '8px 12px' }}>{m.patients}</td>
                    <td style={{ padding: '8px 12px' }}>{m.total_seizures}</td>
                    <td style={{ padding: '8px 12px' }}>{m.avg_seizures_per_patient}</td>
                    <td style={{ padding: '8px 12px' }}>{m.avg_dose_mg || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={m.causal_strength} color={STRENGTH_COLORS[m.causal_strength]} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
          <Card title="Medication Causal Strength">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={medStrengthData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="patients" name="Patients" radius={[0, 4, 4, 0]}>
                  {medStrengthData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Trigger → Seizure Chains">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Trigger', 'Count', 'Avg Duration', 'Risk', 'Severity'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {triggers.map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.trigger}</td>
                      <td style={{ padding: '8px 10px' }}>{t.count}</td>
                      <td style={{ padding: '8px 10px' }}>{t.avg_duration_sec}s</td>
                      <td style={{ padding: '8px 10px' }}>
                        <Badge text={t.risk_level} color={STRENGTH_COLORS[t.risk_level]} />
                      </td>
                      <td style={{ padding: '8px 10px' }}>
                        {Object.entries(t.severity_distribution || {}).map(([k, v]) => (
                          <span key={k} style={{ marginRight: 6 }}>
                            <Badge text={`${k}: ${v}`} color={SEV_COLORS[k] || '#94a3b8'} />
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        <Card title="Causal Graph Summary" span={2}>
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', gap: 24, marginBottom: 12, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 13, color: '#475569' }}>Nodes: <strong>{(graph.nodes || []).length}</strong></span>
              <span style={{ fontSize: 13, color: '#475569' }}>Edges: <strong>{(graph.edges || []).length}</strong></span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8 }}>
              {(graph.edges || []).map((e, i) => {
                const src = (graph.nodes || []).find(n => n.id === e.source)
                const tgt = (graph.nodes || []).find(n => n.id === e.target)
                return (
                  <div key={i} style={{
                    padding: '8px 12px', borderRadius: 8, fontSize: 12,
                    background: STRENGTH_COLORS[e.strength] + '12',
                    border: `1px solid ${STRENGTH_COLORS[e.strength] || '#e2e8f0'}40`
                  }}>
                    <strong>{src?.label || '?'}</strong>
                    <span style={{ margin: '0 6px', color: '#94a3b8' }}>{e.relation}</span>
                    <strong>{tgt?.label || '?'}</strong>
                    <div style={{ marginTop: 4 }}>
                      <Badge text={e.strength} color={STRENGTH_COLORS[e.strength]} />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </Card>
      </>
    )
  }

  /* ── Patient Profiles Tab ──────────────────────────────────────────── */
  const renderPatients = () => {
    const profiles = breakdown?.patient_profiles || []
    const mriCorr = breakdown?.mri_correlations || []
    const assessLinks = breakdown?.assessment_links || []
    const timeline = breakdown?.seizure_timeline || []

    const timelineByDate = {}
    timeline.forEach(t => {
      const d = t.date || 'Unknown'
      if (!timelineByDate[d]) timelineByDate[d] = { date: d, count: 0, severe: 0, mild: 0 }
      timelineByDate[d].count++
      if (t.severity === 'Severe') timelineByDate[d].severe++
      if (t.severity === 'Mild') timelineByDate[d].mild++
    })
    const timelineData = Object.values(timelineByDate).sort((a, b) => a.date.localeCompare(b.date))

    return (
      <>
        <Card title="Seizure Event Timeline" span={2}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} name="Total" dot={{ r: 4 }} />
              <Line type="monotone" dataKey="severe" stroke="#ef4444" strokeWidth={2} name="Severe" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Patient Causal Profiles" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Patient', 'Age', 'Gender', 'Seizures', 'Medications', 'Triggers', 'Avg Duration', 'AI Conf', 'Risk'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{p.age || '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{p.gender || '—'}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{ fontWeight: 700, color: p.seizure_count >= 3 ? '#ef4444' : p.seizure_count >= 1 ? '#f59e0b' : '#10b981' }}>
                        {p.seizure_count}
                      </span>
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      {(p.medications || []).map((m, j) => <Badge key={j} text={m} color="#3b82f6" />)}
                      {(p.medications || []).length === 0 && <span style={{ color: '#94a3b8' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 10px' }}>
                      {(p.triggers || []).map((t, j) => <Badge key={j} text={t} color="#f59e0b" />)}
                      {(p.triggers || []).length === 0 && <span style={{ color: '#94a3b8' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 10px' }}>{p.avg_duration_sec ? `${p.avg_duration_sec}s` : '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{p.avg_ai_confidence ? `${p.avg_ai_confidence}%` : '—'}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={p.risk_level} color={STRENGTH_COLORS[p.risk_level]} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
          <Card title="MRI Finding → Seizure Correlation">
            {mriCorr.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={mriCorr} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="finding" width={150} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="patients" fill="#8b5cf6" name="Patients" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="seizures" fill="#ef4444" name="Seizures" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No MRI correlation data</div>}
          </Card>

          <Card title="Assessment → Seizure Association">
            {assessLinks.length > 0 ? (
              <div style={{ overflowY: 'auto', maxHeight: 200 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      {['Instrument', 'Samples', 'Avg Score', 'Direction'].map(h => (
                        <th key={h} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {assessLinks.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{a.instrument}</td>
                        <td style={{ padding: '6px 8px' }}>{a.samples}</td>
                        <td style={{ padding: '6px 8px' }}>{a.avg_score}</td>
                        <td style={{ padding: '6px 8px' }}>
                          <Badge text={a.direction}
                            color={a.direction === 'positive' ? '#ef4444' : a.direction === 'negative' ? '#10b981' : '#94a3b8'} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No assessment data</div>}
          </Card>
        </div>
      </>
    )
  }

  /* ── Interventions Tab ─────────────────────────────────────────────── */
  const renderInterventions = () => {
    const interventions = breakdown?.interventions || []

    const effData = interventions.map(iv => ({
      name: iv.drug, patients: iv.patients,
      avg_seizures: iv.avg_seizures, avg_duration: iv.avg_duration_sec,
      effectiveness: iv.effectiveness
    }))

    const radarData = interventions.map(iv => ({
      drug: iv.drug,
      patients: iv.patients * 10,
      seizures: Math.max(1, 10 - iv.avg_seizures),
      duration: Math.max(1, Math.round(300 - iv.avg_duration_sec) / 30),
      effectiveness: iv.effectiveness === 'good' ? 10 : iv.effectiveness === 'moderate' ? 6 : 3,
    }))

    return (
      <>
        <Card title="Intervention Effectiveness Comparison" span={2}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={effData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="patients" fill="#3b82f6" name="Patients" radius={[4, 4, 0, 0]} />
              <Bar dataKey="avg_seizures" fill="#ef4444" name="Avg Seizures" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Intervention Detail" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Drug', 'Patients', 'Avg Seizures', 'Avg Duration (s)', 'Severity Breakdown', 'Effectiveness'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {interventions.map((iv, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{iv.drug}</td>
                    <td style={{ padding: '8px 12px' }}>{iv.patients}</td>
                    <td style={{ padding: '8px 12px' }}>{iv.avg_seizures}</td>
                    <td style={{ padding: '8px 12px' }}>{iv.avg_duration_sec || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>
                      {Object.entries(iv.severity_dist || {}).map(([k, v]) => (
                        <span key={k} style={{ marginRight: 4 }}>
                          <Badge text={`${k}: ${v}`} color={SEV_COLORS[k] || '#94a3b8'} />
                        </span>
                      ))}
                      {Object.keys(iv.severity_dist || {}).length === 0 && <span style={{ color: '#94a3b8' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <Badge text={iv.effectiveness} color={EFF_COLORS[iv.effectiveness]} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Definitions Tab ───────────────────────────────────────────────── */
  const renderDefinitions = () => {
    const sections = definitions?.sections || []
    return (
      <>
        {sections.map((sec, si) => (
          <Card key={si} title={sec.title} span={2}>
            <div style={{ display: 'grid', gap: 10 }}>
              {(sec.items || []).map((item, ii) => (
                <div key={ii} style={{
                  padding: '10px 14px', borderRadius: 8, background: '#f8fafc',
                  borderLeft: '3px solid #3b82f6'
                }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.definition}</div>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </>
    )
  }

  return (
    <div style={{ padding: '24px 20px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Causal AI Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Causal inference and relationship analysis — medication pathways, trigger chains, demographic associations
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        {tab === 'overview' && renderOverview()}
        {tab === 'pathways' && renderPathways()}
        {tab === 'patients' && renderPatients()}
        {tab === 'interventions' && renderInterventions()}
        {tab === 'definitions' && renderDefinitions()}
      </div>
    </div>
  )
}
