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
  { id: 'models', label: 'Model Comparison' },
  { id: 'patients', label: 'Patient Detail' },
  { id: 'fall_risk', label: 'Fall Risk' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SemiologyClassifierDashboard() {
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
      axios.get(`${API_URL}/api/seizure-semiology/overview`),
      axios.get(`${API_URL}/api/seizure-semiology/breakdown`),
      axios.get(`${API_URL}/api/seizure-semiology/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading seizure semiology data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  /* ── Overview Tab ───────────────────────────────── */
  const renderOverview = () => {
    const typeDist = overview?.type_distribution || []
    const zoneDist = overview?.zone_distribution || []
    const confHist = overview?.confidence_histogram || []
    const lat = overview?.lateralisation || {}
    const latData = [
      { name: 'Left', value: lat.left || 0 },
      { name: 'Right', value: lat.right || 0 },
      { name: 'Bilateral', value: lat.bilateral || 0 },
    ]

    return (
      <>
        {/* KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Events Classified" value={overview?.total_events_classified} color="#3b82f6" /></Card>
          <Card><KPI label="Total Patients" value={overview?.total_patients} color="#8b5cf6" /></Card>
          <Card><KPI label="Semiology Types" value={overview?.semiology_types_detected} sub="out of 9" color="#10b981" /></Card>
          <Card><KPI label="Avg Confidence" value={pct(Math.round((overview?.average_confidence || 0) * 100))} color="#f59e0b" /></Card>
          <Card><KPI label="Fall Risk Events" value={overview?.fall_risk_events} sub={`${overview?.fall_risk_pct}% of total`} color="#ef4444" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Type Distribution */}
          <Card title="Seizure Semiology Type Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={typeDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="type" type="category" width={130} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {typeDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Lateralisation Pie */}
          <Card title="Lateralisation Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={latData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                  {latData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Zone Distribution */}
          <Card title="Inferred Epileptogenic Zone">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={zoneDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="zone" type="category" width={200} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Confidence Histogram */}
          <Card title="Confidence Score Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      </>
    )
  }

  /* ── Model Comparison Tab ───────────────────────── */
  const renderModels = () => {
    const models = overview?.model_performance || []
    const perClass = overview?.per_class_metrics || []

    const radarData = perClass.map(c => ({
      type: c.type.length > 12 ? c.type.slice(0, 12) + '...' : c.type,
      precision: Math.round(c.precision * 100),
      recall: Math.round(c.recall * 100),
      f1: Math.round(c.f1 * 100),
    }))

    return (
      <>
        {/* Model comparison table */}
        <Card title="Model Architecture Comparison" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Model', 'Type', 'Accuracy', 'Macro F1', 'AUC-ROC', 'Inference (ms)'].map(h =>
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.model}</td>
                    <td style={{ padding: '8px 6px' }}><Badge text={m.type} color="blue" /></td>
                    <td style={{ padding: '8px 6px', fontWeight: 600, color: m.accuracy > 0.85 ? '#10b981' : '#f59e0b' }}>
                      {(m.accuracy * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '8px 6px' }}>{(m.macro_f1 * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 6px' }}>{(m.auc_roc * 100).toFixed(1)}%</td>
                    <td style={{ padding: '8px 6px' }}>{m.inference_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
          {/* Model accuracy bar chart */}
          <Card title="Model Accuracy Comparison">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={models.map(m => ({ name: m.abbrev, accuracy: Math.round(m.accuracy * 100), f1: Math.round(m.macro_f1 * 100) }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[60, 100]} />
                <Tooltip formatter={v => `${v}%`} />
                <Legend />
                <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Accuracy" />
                <Bar dataKey="f1" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Macro F1" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-class radar */}
          <Card title="Per-Class Metrics (Best Model)">
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="type" tick={{ fontSize: 9 }} />
                <PolarRadiusAxis domain={[0, 100]} />
                <Radar name="Precision" dataKey="precision" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                <Radar name="Recall" dataKey="recall" stroke="#ef4444" fill="#ef4444" fillOpacity={0.2} />
                <Tooltip formatter={v => `${v}%`} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Per-class table */}
        <Card title="Per-Class Performance Detail" span={2}>
          <div style={{ overflowX: 'auto', marginTop: 16 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Semiology Type', 'Precision', 'Recall', 'F1 Score', 'Support'].map(h =>
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {perClass.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{c.type}</td>
                    <td style={{ padding: '6px' }}>{(c.precision * 100).toFixed(1)}%</td>
                    <td style={{ padding: '6px' }}>{(c.recall * 100).toFixed(1)}%</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: c.f1 > 0.8 ? '#10b981' : c.f1 > 0.6 ? '#f59e0b' : '#ef4444' }}>
                      {(c.f1 * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '6px' }}>{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </>
    )
  }

  /* ── Patient Detail Tab ─────────────────────────── */
  const renderPatients = () => {
    const patients = breakdown?.patient_profiles || []
    return (
      <>
        <Card title={`Patient Semiology Profiles (${patients.length} patients)`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'Age', 'Sex', 'Events', 'Avg Conf', 'Fall Risk', 'AI-Clinician', 'Types Detected'].map(h =>
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px' }}>{p.age || '--'}</td>
                    <td style={{ padding: '6px' }}>{p.sex || '--'}</td>
                    <td style={{ padding: '6px' }}>{p.total_events}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: p.avg_confidence > 0.8 ? '#10b981' : '#f59e0b' }}>
                      {(p.avg_confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '6px' }}>
                      <Badge text={p.fall_risk_level}
                        color={p.fall_risk_level === 'high' ? 'red' : p.fall_risk_level === 'moderate' ? 'amber' : 'green'} />
                    </td>
                    <td style={{ padding: '6px' }}>{p.ai_clinician_agreement_pct}%</td>
                    <td style={{ padding: '6px' }}>
                      {(p.types_detected || []).map(t => <Badge key={t} text={t} color="purple" />)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Confusion matrix */}
        {breakdown?.confusion_matrix && (
          <Card title="Confusion Matrix (Aggregate — Best Model)" span={2}>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    <th style={{ padding: '4px 6px', fontSize: 10, color: '#64748b' }}>True \ Pred</th>
                    {(breakdown.confusion_labels || []).map(l => (
                      <th key={l} style={{ padding: '4px 6px', fontSize: 9, color: '#64748b', writingMode: 'vertical-lr', transform: 'rotate(180deg)', maxWidth: 20 }}>{l}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.confusion_labels || []).map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: '4px 6px', fontWeight: 600, fontSize: 10, whiteSpace: 'nowrap' }}>{row}</td>
                      {(breakdown.confusion_labels || []).map((col, ci) => {
                        const val = (breakdown.confusion_matrix[row] || {})[col] || 0
                        const isDiag = ri === ci
                        return (
                          <td key={ci} style={{
                            padding: '4px 6px', textAlign: 'center', fontWeight: isDiag ? 700 : 400,
                            background: isDiag ? '#dcfce7' : val > 3 ? '#fee2e2' : '#f8fafc',
                            color: isDiag ? '#166534' : val > 3 ? '#991b1b' : '#64748b',
                          }}>{val}</td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </>
    )
  }

  /* ── Fall Risk Tab ──────────────────────────────── */
  const renderFallRisk = () => {
    const patients = breakdown?.patient_profiles || []
    const high = patients.filter(p => p.fall_risk_level === 'high')
    const moderate = patients.filter(p => p.fall_risk_level === 'moderate')
    const low = patients.filter(p => p.fall_risk_level === 'low')

    const riskPie = [
      { name: 'High', value: high.length },
      { name: 'Moderate', value: moderate.length },
      { name: 'Low', value: low.length },
    ]
    const riskColors = ['#ef4444', '#f59e0b', '#10b981']

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="High Risk" value={high.length} sub="Headgear + alert device" color="#ef4444" /></Card>
          <Card><KPI label="Moderate Risk" value={moderate.length} sub="Consider alert device" color="#f59e0b" /></Card>
          <Card><KPI label="Low Risk" value={low.length} sub="Standard monitoring" color="#10b981" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <Card title="Fall Risk Stratification">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={riskPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                  {riskPie.map((_, i) => <Cell key={i} fill={riskColors[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Fall Risk Weight by Seizure Type">
            {definitions?.fall_risk_scoring?.levels && (
              <div style={{ marginBottom: 12 }}>
                {definitions.fall_risk_scoring.levels.map((l, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 6 }}>
                    <Badge text={l.level} color={l.level === 'High' ? 'red' : l.level === 'Moderate' ? 'amber' : 'green'} />
                    <span style={{ fontSize: 12, color: '#475569', marginLeft: 4 }}>{l.range} — {l.action}</span>
                  </div>
                ))}
              </div>
            )}
            {(definitions?.semiology_types || []).map((st, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', marginBottom: 6 }}>
                <div style={{ width: 130, fontSize: 12, fontWeight: 500 }}>{st.type}</div>
                <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 6, height: 16, position: 'relative', overflow: 'hidden' }}>
                  <div style={{
                    width: `${st.fall_risk_weight * 100}%`, height: '100%', borderRadius: 6,
                    background: st.fall_risk_weight >= 0.6 ? '#ef4444' : st.fall_risk_weight >= 0.4 ? '#f59e0b' : '#10b981',
                  }} />
                </div>
                <div style={{ width: 40, textAlign: 'right', fontSize: 12, fontWeight: 600 }}>{st.fall_risk_weight}</div>
              </div>
            ))}
          </Card>
        </div>

        {/* High-risk patients table */}
        {high.length > 0 && (
          <Card title={`High-Risk Patients Requiring Intervention (${high.length})`} span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient', 'Age', 'Events', 'Fall Risk Score', 'Dominant Types', 'Recommended Action'].map(h =>
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {high.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px' }}>{p.age || '--'}</td>
                      <td style={{ padding: '6px' }}>{p.total_events}</td>
                      <td style={{ padding: '6px', fontWeight: 700, color: '#ef4444' }}>{p.cumulative_fall_risk}</td>
                      <td style={{ padding: '6px' }}>
                        {(p.types_detected || []).slice(0, 3).map(t => <Badge key={t} text={t} color="red" />)}
                      </td>
                      <td style={{ padding: '6px', fontSize: 12, color: '#991b1b' }}>Protective headgear + seizure-alert device referral</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </>
    )
  }

  /* ── Definitions Tab ────────────────────────────── */
  const renderDefinitions = () => {
    const defs = definitions || {}
    return (
      <>
        <Card title="Seizure Semiology Types" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Type', 'Localisation Zone', 'Lateralising', 'Fall Risk', 'Description'].map(h =>
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {(defs.semiology_types || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{s.type}</td>
                    <td style={{ padding: '8px 6px', fontSize: 12 }}>{s.localisation_zone}</td>
                    <td style={{ padding: '8px 6px' }}>
                      <Badge text={s.lateralising ? 'Yes' : 'No'} color={s.lateralising ? 'green' : 'blue'} />
                    </td>
                    <td style={{ padding: '8px 6px', fontWeight: 600, color: s.fall_risk_weight >= 0.6 ? '#ef4444' : '#f59e0b' }}>
                      {s.fall_risk_weight}
                    </td>
                    <td style={{ padding: '8px 6px', fontSize: 12, color: '#475569', maxWidth: 400 }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Classification Pipeline" span={2}>
          <div style={{ display: 'grid', gap: 8 }}>
            {(defs.classification_methodology?.pipeline_steps || []).map((step, i) => (
              <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, color: '#334155' }}>
                {step}
              </div>
            ))}
          </div>
        </Card>

        <Card title="Model Architectures" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {(defs.classification_methodology?.models || []).map((m, i) => (
              <div key={i} style={{ padding: 14, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>
                  {m.name} <Badge text={m.status} color={m.status === 'built' ? 'green' : 'amber'} />
                </div>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}><strong>Features:</strong> {m.features}</div>
                <div style={{ fontSize: 12, color: '#475569' }}><strong>Training:</strong> {m.training}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="ILAE Classification Mapping" span={2}>
          {(defs.ilae_classification_mapping || []).map((cat, i) => (
            <div key={i} style={{ padding: 12, marginBottom: 8, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#3b82f6', marginBottom: 6 }}>{cat.ilae_category}</div>
              <div>{(cat.semiology_types || []).map(t => <Badge key={t} text={t} color="purple" />)}</div>
            </div>
          ))}
        </Card>

        <Card title="References" span={2}>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {(defs.references || []).map((r, i) => (
              <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{r}</li>
            ))}
          </ul>
        </Card>
      </>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Seizure Semiology Classifier</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 14 }}>
          AI-driven seizure-type classification from video-EEG motor patterns — localisation inference, model comparison, fall risk
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'models' && renderModels()}
      {tab === 'patients' && renderPatients()}
      {tab === 'fall_risk' && renderFallRisk()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
