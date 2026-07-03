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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
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

const fmt = v => (v != null ? v : '--')

const LESION_TYPE_COLORS = {
  HS: '#e74c3c', FCD: '#e67e22', TUM: '#9b59b6', CAV: '#3498db',
  AVM: '#e91e63', ENC: '#95a5a6', NRM: '#2ecc71'
}

const CLASSIFICATION_COLORS = {
  LESIONAL: '#e74c3c', NON_LESIONAL: '#2ecc71'
}

const CONFIDENCE_COLORS = {
  High: '#2ecc71', Moderate: '#f39c12', Low: '#e74c3c'
}

const QUALITY_COLORS = {
  Diagnostic: '#2ecc71', Adequate: '#3498db', Suboptimal: '#f39c12', 'Non-diagnostic': '#e74c3c'
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

function LesionBadge({ type }) {
  return <Badge text={type || '--'} color={LESION_TYPE_COLORS[type] || '#64748b'} />
}

function ClassificationBadge({ classification }) {
  const c = classification?.toUpperCase()?.replace(/[- ]/g, '_')
  return <Badge text={classification || '--'} color={CLASSIFICATION_COLORS[c] || '#64748b'} />
}

function ConfidenceBadge({ level }) {
  return <Badge text={level || '--'} color={CONFIDENCE_COLORS[level] || '#64748b'} />
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'lesion', label: 'Lesion Analysis' },
  { id: 'patients', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RadiologistDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/radiologist/overview`),
      axios.get(`${API_URL}/api/radiologist/breakdown`),
      axios.get(`${API_URL}/api/radiologist/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const toggleExpand = id => setExpanded(prev => ({ ...prev, [id]: !prev[id] }))

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Radiologist Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Radiologist Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        MRI lesion analysis, neuroradiology quality metrics, structural imaging correlation with EEG findings
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total MRI Scans"><KPI value={ov.total_mri_scans} label="Scans" color="#3b82f6" /></Card>
          <Card title="Lesion-Positive Rate"><KPI value={ov.lesion_positive_rate != null ? `${ov.lesion_positive_rate}%` : '--'} label="Lesion-positive" color="#e74c3c" /></Card>
          <Card title="MRI Coverage"><KPI value={ov.mri_coverage != null ? `${ov.mri_coverage}%` : '--'} label="Of patients" color="#8b5cf6" /></Card>
          <Card title="Avg Confidence"><KPI value={ov.avg_radiologist_confidence} label="Radiologist confidence" color="#2ecc71" /></Card>

          {/* KPI Row 2 */}
          <Card title="Hippocampal Sclerosis"><KPI value={ov.hippocampal_sclerosis_count} label="Cases" color="#e67e22" /></Card>
          <Card title="Lesional Count"><KPI value={ov.lesion_positive_count} label="Lesion-positive" color="#e74c3c" /></Card>
          <Card title="Non-Lesional Count"><KPI value={ov.non_lesional_count} label="Non-lesional" color="#2ecc71" /></Card>
          <Card title="Enhancing Lesions"><KPI value={ov.enhancing_count} label="Enhancing" color="#9b59b6" /></Card>

          {/* Lesion Type Distribution */}
          <Card title="Lesion Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.lesion_type_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} />
                <YAxis dataKey="label" type="category" fontSize={11} width={100} />
                <Tooltip />
                <Bar dataKey="count" name="Count">
                  {(ov.lesion_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={LESION_TYPE_COLORS[e.lesion_type] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Classification Distribution (Lesional vs Non-lesional) */}
          <Card title="Classification (Lesional vs Non-lesional)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.classification_distribution || []} dataKey="count" nameKey="classification" cx="50%" cy="50%" outerRadius={80} label={e => `${e.classification}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.classification_distribution || []).map((e, i) => {
                    const key = e.classification?.toUpperCase()?.replace(/[- ]/g, '_')
                    return <Cell key={i} fill={CLASSIFICATION_COLORS[key] || COLORS[i % COLORS.length]} />
                  })}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Location Distribution */}
          <Card title="Lesion Location Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.location_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="location" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3498db" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Laterality Distribution */}
          <Card title="Laterality Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.laterality_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="laterality" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* LESION ANALYSIS TAB */}
      {tab === 'lesion' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Quality Distribution */}
          <Card title="MRI Quality Distribution">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Quality', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.quality_distribution || []).map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px' }}><Badge text={q.quality} color={QUALITY_COLORS[q.quality] || '#64748b'} /></td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{q.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Lesion Type Breakdown */}
          <Card title="Lesion Type Breakdown">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Lesion Type', 'Label', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.lesion_type_distribution || []).map((lt, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px' }}><LesionBadge type={lt.lesion_type} /></td>
                      <td style={{ padding: '6px' }}>{lt.label}</td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{lt.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* T2/FLAIR Signal Distribution */}
          <Card title="T2/FLAIR Signal Distribution">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Signal', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.t2_flair_distribution || []).map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px' }}>{t.signal}</td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{t.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Enhancing vs Non-Enhancing */}
          <Card title="Enhancing vs Non-Enhancing Lesions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#9b59b618' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#9b59b6' }}>{fmt(ov.enhancing_count)}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Enhancing</div>
              </div>
              <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#3498db18' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#3498db' }}>{fmt(ov.non_enhancing_count)}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Non-Enhancing</div>
              </div>
            </div>
          </Card>

          {/* Hippocampal Volume Asymmetry Stats */}
          <Card title="Hippocampal Volume Asymmetry Statistics">
            {ov.hippocampal_volume_asymmetry_stats ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f1f5f9' }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{fmt(ov.hippocampal_volume_asymmetry_stats.mean)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Mean Asymmetry</div>
                </div>
                <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f1f5f9' }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#e74c3c' }}>{fmt(ov.hippocampal_volume_asymmetry_stats.max)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Max Asymmetry</div>
                </div>
                <div style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#f1f5f9' }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#2ecc71' }}>{fmt(ov.hippocampal_volume_asymmetry_stats.min)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Min Asymmetry</div>
                </div>
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: 13 }}>No hippocampal volume asymmetry data available.</p>
            )}
          </Card>

          {/* Confidence Distribution */}
          <Card title="Radiologist Confidence Distribution">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Confidence Level', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.confidence_distribution || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px' }}><ConfidenceBadge level={c.level} /></td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{c.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patients' && (
        <Card title="Per-Patient MRI and EEG Detail" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['', 'Patient', 'Age', 'Gender', 'Disease', 'Lesion Type', 'Classification', 'Confidence'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => {
                  const mri = p.mri || {}
                  const eeg = p.eeg_analysis || {}
                  const isExp = expanded[p.patient_id]
                  return (
                    <React.Fragment key={i}>
                      <tr
                        style={{ borderBottom: '1px solid #e2e8f0', cursor: 'pointer' }}
                        onClick={() => toggleExpand(p.patient_id)}
                      >
                        <td style={{ padding: '6px', fontSize: 14, transition: 'transform 0.2s', transform: isExp ? 'rotate(90deg)' : 'rotate(0deg)', display: 'inline-block', width: 20 }}>&#9654;</td>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px' }}>{fmt(p.age)}</td>
                        <td style={{ padding: '6px' }}>{fmt(p.gender)}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>{fmt(p.disease)}</td>
                        <td style={{ padding: '6px' }}><LesionBadge type={mri.lesion_type} /></td>
                        <td style={{ padding: '6px' }}><ClassificationBadge classification={mri.classification} /></td>
                        <td style={{ padding: '6px' }}><ConfidenceBadge level={mri.confidence} /></td>
                      </tr>
                      {isExp && (
                        <tr>
                          <td colSpan={8} style={{ padding: '12px 20px', background: '#f8fafc' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                              {/* MRI Findings */}
                              <div>
                                <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>MRI Findings</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                                  <div><span style={{ color: '#64748b' }}>Lesion Type:</span> <LesionBadge type={mri.lesion_type} /></div>
                                  <div><span style={{ color: '#64748b' }}>Label:</span> {fmt(mri.lesion_label)}</div>
                                  <div><span style={{ color: '#64748b' }}>Location:</span> {fmt(mri.location)}</div>
                                  <div><span style={{ color: '#64748b' }}>Laterality:</span> {fmt(mri.laterality)}</div>
                                  <div><span style={{ color: '#64748b' }}>Hippocampal Sclerosis:</span> {mri.hippocampal_sclerosis ? 'Yes' : 'No'}</div>
                                  <div><span style={{ color: '#64748b' }}>Volume Asymmetry:</span> {fmt(mri.hippocampal_volume_asymmetry)}</div>
                                  <div><span style={{ color: '#64748b' }}>T2/FLAIR Signal:</span> {fmt(mri.t2_flair_signal)}</div>
                                  <div><span style={{ color: '#64748b' }}>Quality:</span> {mri.quality ? <Badge text={mri.quality} color={QUALITY_COLORS[mri.quality] || '#64748b'} /> : '--'}</div>
                                  <div><span style={{ color: '#64748b' }}>Classification:</span> <ClassificationBadge classification={mri.classification} /></div>
                                  <div><span style={{ color: '#64748b' }}>Confidence:</span> <ConfidenceBadge level={mri.confidence} /></div>
                                </div>
                              </div>
                              {/* Linked EEG Analysis */}
                              <div>
                                <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Linked EEG Analysis</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                                  <div><span style={{ color: '#64748b' }}>Predicted Label:</span> {fmt(eeg.predicted_label)}</div>
                                  <div><span style={{ color: '#64748b' }}>Confidence:</span> {fmt(eeg.confidence)}</div>
                                  <div><span style={{ color: '#64748b' }}>Signal Quality:</span> {fmt(eeg.signal_quality)}</div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Neuroradiology Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
