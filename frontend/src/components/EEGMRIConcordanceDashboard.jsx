import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const CONC_COLORS = {
  concordant: '#16a34a',
  partially_concordant: '#3b82f6',
  discordant: '#ef4444',
  non_lesional: '#94a3b8',
  insufficient: '#cbd5e1',
}
const PIE_COLORS = ['#16a34a', '#3b82f6', '#ef4444', '#94a3b8', '#cbd5e1']
const CANDIDACY_COLORS = {
  'Strong candidate': '#16a34a',
  'Moderate candidate': '#3b82f6',
  'Needs further workup': '#f59e0b',
  'Phase II / invasive': '#ef4444',
  'Cannot assess': '#94a3b8',
}
const LESION_COLORS = {
  HS: '#8b5cf6', FCD: '#3b82f6', TUM: '#ef4444', CAV: '#f59e0b',
  AVM: '#ec4899', ENC: '#14b8a6', NRM: '#94a3b8', NL: '#cbd5e1',
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

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

function ConcordanceBadge({ concordance }) {
  const color = CONC_COLORS[concordance] || '#94a3b8'
  const label = concordance ? concordance.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function CandidacyBadge({ tier }) {
  const color = CANDIDACY_COLORS[tier] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{tier || 'Unknown'}</span>
  )
}

export default function EEGMRIConcordanceDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedPatient, setExpandedPatient] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/eeg-mri-concordance/overview`),
          axios.get(`${API_URL}/api/eeg-mri-concordance/breakdown`),
          axios.get(`${API_URL}/api/eeg-mri-concordance/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG-MRI Concordance data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'mri', label: 'MRI Analysis' },
    { id: 'concordance', label: 'Concordance Map' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const concDist = overview?.concordance_distribution || {}
  const lesionCounts = overview?.lesion_type_counts || {}
  const locationCounts = overview?.location_counts || {}
  const lateralityCounts = overview?.laterality_counts || {}
  const candidacyDist = overview?.candidacy_distribution || {}
  const lobeMatrix = overview?.lobe_match_matrix || {}
  const patients = breakdown?.patients || []

  const concPieData = Object.entries(concDist).map(([k, v]) => ({ name: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()), value: v }))
  const lesionBarData = Object.entries(lesionCounts).map(([k, v]) => ({ name: k, count: v }))
  const locationBarData = Object.entries(locationCounts).filter(([k]) => k && k !== 'null').map(([k, v]) => ({ name: k, count: v }))
  const candidacyBarData = Object.entries(candidacyDist).map(([k, v]) => ({ name: k, count: v }))
  const lobeMatrixData = Object.entries(lobeMatrix).map(([k, v]) => ({ name: k, count: v }))

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>EEG-MRI Concordance Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Structural MRI lesion vs EEG seizure focus correlation for pre-surgical epilepsy evaluation
        &mdash; {overview?.total_patients || 0} patients
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '2px solid #e2e8f0', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'transparent', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ─────────────────────────────────── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Total Patients"><KPI value={overview?.total_patients || 0} label="with MRI data" /></Card>
          <Card title="Concordance Rate"><KPI value={`${kpis.concordance_rate || 0}%`} label="lesional patients" color="#16a34a" /></Card>
          <Card title="Lesional"><KPI value={kpis.lesional_count || 0} label="structural lesion" color="#8b5cf6" /></Card>
          <Card title="Non-lesional"><KPI value={kpis.non_lesional_count || 0} label="normal MRI" color="#94a3b8" /></Card>

          <Card title="Strong Surgical Candidates"><KPI value={kpis.strong_surgical_candidates || 0} label="concordant" color="#16a34a" /></Card>
          <Card title="Needs Further Workup"><KPI value={kpis.needs_further_workup || 0} label="discordant + non-lesional" color="#f59e0b" /></Card>
          <Card title="Mean EEG Confidence"><KPI value={fmt(kpis.mean_eeg_confidence)} label="model confidence" color="#3b82f6" /></Card>
          <Card title="Hippocampal Sclerosis"><KPI value={kpis.hippocampal_sclerosis_count || 0} label="HS on MRI" color="#8b5cf6" /></Card>

          {/* Concordance pie */}
          <Card title="Concordance Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={concPieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                     outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {concPieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Surgical candidacy bar */}
          <Card title="Surgical Candidacy Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={candidacyBarData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6">
                  {candidacyBarData.map((entry, i) => (
                    <Cell key={i} fill={CANDIDACY_COLORS[entry.name] || '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── MRI Analysis Tab ─────────────────────────────── */}
      {tab === 'mri' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Lesion type distribution */}
          <Card title="Lesion Type Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={lesionBarData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count">
                  {lesionBarData.map((entry, i) => (
                    <Cell key={i} fill={LESION_COLORS[entry.name] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Location distribution */}
          <Card title="MRI Lesion Location">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={locationBarData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Full MRI patient table */}
          <Card title="MRI Findings per Patient" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Lesion Type</th>
                    <th style={thStyle}>Location</th>
                    <th style={thStyle}>Laterality</th>
                    <th style={thStyle}>Classification</th>
                    <th style={thStyle}>HS</th>
                    <th style={thStyle}>Quality</th>
                    <th style={thStyle}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.patient_id}>
                      <td style={tdStyle}>{p.patient_id}</td>
                      <td style={tdStyle}>
                        <span style={{ color: LESION_COLORS[p.mri_lesion_type] || '#64748b', fontWeight: 600 }}>
                          {p.mri_lesion_type}
                        </span>
                        <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 4 }}>{p.mri_lesion_label}</span>
                      </td>
                      <td style={tdStyle}>{p.mri_location || '--'}</td>
                      <td style={tdStyle}>{p.mri_laterality || '--'}</td>
                      <td style={tdStyle}>{p.mri_classification}</td>
                      <td style={tdStyle}>{p.hippocampal_sclerosis}</td>
                      <td style={tdStyle}>{p.mri_quality}</td>
                      <td style={tdStyle}>{p.mri_radiologist_confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Concordance Map Tab ──────────────────────────── */}
      {tab === 'concordance' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Lobe match matrix */}
          <Card title="Lobe Match Matrix (MRI Location \u2192 EEG Focus)" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(300, lobeMatrixData.length * 28)}>
              <BarChart data={lobeMatrixData} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Concordance by lesion type */}
          <Card title="Concordance by Lesion Type" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>MRI (Lesion / Location / Lat)</th>
                    <th style={thStyle}>EEG (Focus / Lat)</th>
                    <th style={thStyle}>Pattern</th>
                    <th style={thStyle}>Concordance</th>
                    <th style={thStyle}>Surgical Candidacy</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.patient_id} style={{ background: p.concordance === 'concordant' ? '#f0fdf4' : p.concordance === 'discordant' ? '#fef2f2' : undefined }}>
                      <td style={tdStyle}>{p.patient_id}</td>
                      <td style={tdStyle}>
                        <span style={{ color: LESION_COLORS[p.mri_lesion_type] || '#64748b', fontWeight: 600 }}>{p.mri_lesion_type}</span>
                        {' / '}{p.mri_location || '--'}{' / '}{p.mri_laterality || '--'}
                      </td>
                      <td style={tdStyle}>{p.eeg_focus_lobe}{' / '}{p.eeg_focus_laterality}</td>
                      <td style={tdStyle}><span style={{ fontSize: 11 }}>{p.eeg_pattern}</span></td>
                      <td style={tdStyle}><ConcordanceBadge concordance={p.concordance} /></td>
                      <td style={tdStyle}><CandidacyBadge tier={p.surgical_candidacy} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patient Detail Tab ───────────────────────────── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
          {patients.map(p => {
            const isExpanded = expandedPatient === p.patient_id
            return (
              <Card key={p.patient_id}>
                <div onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                     style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{
                    display: 'inline-block', width: 20, textAlign: 'center', fontSize: 14,
                    transform: isExpanded ? 'rotate(90deg)' : 'none', transition: 'transform .15s'
                  }}>&#9654;</span>
                  <span style={{ fontWeight: 600, minWidth: 80 }}>{p.patient_id}</span>
                  <span style={{ fontSize: 12, color: '#64748b' }}>Age {p.age} / {p.sex}</span>
                  <ConcordanceBadge concordance={p.concordance} />
                  <CandidacyBadge tier={p.surgical_candidacy} />
                  <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>
                    Engel I: {p.engel_I_rate}
                  </span>
                </div>
                {isExpanded && (
                  <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                    {/* MRI detail */}
                    <div>
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>MRI Findings</h4>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <tbody>
                          {[
                            ['Lesion Type', `${p.mri_lesion_type} — ${p.mri_lesion_label}`],
                            ['Location', p.mri_location || '--'],
                            ['Laterality', p.mri_laterality || '--'],
                            ['Classification', p.mri_classification],
                            ['Hippocampal Sclerosis', p.hippocampal_sclerosis],
                            ['Volume Asymmetry', fmt(p.hippocampal_volume_asymmetry)],
                            ['T2/FLAIR Signal', p.enhancing ? 'Enhancing' : 'Non-enhancing'],
                            ['Quality', p.mri_quality],
                            ['Protocol', p.mri_protocol],
                            ['Radiologist Confidence', p.mri_radiologist_confidence],
                          ].map(([label, val], i) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, color: '#64748b', fontSize: 12, width: '45%' }}>{label}</td>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>{val}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {/* EEG detail */}
                    <div>
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>EEG Focus</h4>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <tbody>
                          {[
                            ['Focus Lobe', p.eeg_focus_lobe],
                            ['Focus Laterality', p.eeg_focus_laterality],
                            ['EEG Pattern', p.eeg_pattern],
                            ['EEG Confidence', fmt(p.eeg_confidence)],
                            ['Model Prediction', p.eeg_predicted_label || '--'],
                            ['Model Confidence', p.eeg_model_confidence != null ? fmt(p.eeg_model_confidence) : '--'],
                            ['Signal Quality', p.eeg_signal_quality || '--'],
                          ].map(([label, val], i) => (
                            <tr key={i}>
                              <td style={{ ...tdStyle, color: '#64748b', fontSize: 12, width: '45%' }}>{label}</td>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>{val}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {/* Concordance assessment */}
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '16px 0 8px' }}>Concordance Assessment</h4>
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <tbody>
                          <tr>
                            <td style={{ ...tdStyle, color: '#64748b', fontSize: 12, width: '45%' }}>Concordance</td>
                            <td style={tdStyle}><ConcordanceBadge concordance={p.concordance} /></td>
                          </tr>
                          <tr>
                            <td style={{ ...tdStyle, color: '#64748b', fontSize: 12 }}>Surgical Candidacy</td>
                            <td style={tdStyle}><CandidacyBadge tier={p.surgical_candidacy} /></td>
                          </tr>
                          <tr>
                            <td style={{ ...tdStyle, color: '#64748b', fontSize: 12 }}>Engel I Rate</td>
                            <td style={{ ...tdStyle, fontWeight: 600, color: '#16a34a' }}>{p.engel_I_rate}</td>
                          </tr>
                          {p.additional_workup && p.additional_workup.length > 0 && (
                            <tr>
                              <td style={{ ...tdStyle, color: '#64748b', fontSize: 12 }}>Additional Workup</td>
                              <td style={tdStyle}>
                                {p.additional_workup.map((w, wi) => (
                                  <span key={wi} style={{
                                    display: 'inline-block', padding: '2px 6px', borderRadius: 4,
                                    fontSize: 10, background: '#fef3c7', color: '#92400e', marginRight: 4, marginBottom: 2
                                  }}>{w}</span>
                                ))}
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(defs.sections || []).map((sec, si) => (
            <Card key={si} title={sec.heading}>
              {sec.content && <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{sec.content}</p>}
              {sec.items && (
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
                  <tbody>
                    {sec.items.map((item, ii) => (
                      <tr key={ii}>
                        <td style={{ ...tdStyle, fontWeight: 600, color: '#334155', width: '25%' }}>{item.term}</td>
                        <td style={{ ...tdStyle, color: '#475569' }}>{item.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
