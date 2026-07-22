import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#16a34a', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
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

function CompBadge({ valA, valB, higher_is_better }) {
  if (valA == null || valB == null) return null
  const diff = valA - valB
  if (Math.abs(diff) < 0.01) return <span style={{ fontSize: 11, color: '#94a3b8' }}>≈ equal</span>
  const aWins = higher_is_better ? diff > 0 : diff < 0
  return (
    <span style={{
      fontSize: 11, fontWeight: 600,
      color: aWins ? '#3b82f6' : '#16a34a'
    }}>
      {aWins ? '← A' : 'B →'}
    </span>
  )
}

export default function PatientComparisonDashboard() {
  const [overview, setOverview] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('compare')
  const [patientA, setPatientA] = useState('EPAT001')
  const [patientB, setPatientB] = useState('EPAT002')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, df] = await Promise.all([
          axios.get(`${API_URL}/api/patient-comparison/overview`),
          axios.get(`${API_URL}/api/patient-comparison/definitions`)
        ])
        setOverview(ov.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  useEffect(() => {
    const loadComparison = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/patient-comparison/compare?a=${patientA}&b=${patientB}`)
        setComparison(res.data)
      } catch (e) {
        setError(e.message)
      }
    }
    if (patientA && patientB) loadComparison()
  }, [patientA, patientB])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Patient Comparison…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No data available</div>

  const tabs = ['compare', 'details', 'definitions']
  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Patient Comparison</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Side-by-side comparison of two patients across demographics, seizures, assessments, cognition, and medication adherence
      </p>

      {/* KPI row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16, marginBottom: 24 }}>
        <Card><KPI label="Total Patients" value={fmt(k.total_patients)} /></Card>
        <Card><KPI label="Seizure Events" value={fmt(k.total_seizure_events)} /></Card>
        <Card><KPI label="Assessments" value={fmt(k.total_assessments)} /></Card>
        <Card><KPI label="Cognitive Tests" value={fmt(k.total_cognitive_tests)} /></Card>
        <Card><KPI label="Med Records" value={fmt(k.total_med_records)} /></Card>
        <Card><KPI label="EEG Analyses" value={fmt(k.total_analyses)} /></Card>
      </div>

      {/* Patient selectors */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 20, alignItems: 'center' }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>Patient A:</label>
        <select
          value={patientA}
          onChange={e => setPatientA(e.target.value)}
          style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 13 }}
        >
          {(overview.patients || []).map(p => (
            <option key={p.patient_id} value={p.patient_id}>
              {p.patient_id} — {p.full_name} ({p.age}{p.sex ? `, ${p.sex}` : ''})
            </option>
          ))}
        </select>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>Patient B:</label>
        <select
          value={patientB}
          onChange={e => setPatientB(e.target.value)}
          style={{ padding: '6px 12px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 13 }}
        >
          {(overview.patients || []).map(p => (
            <option key={p.patient_id} value={p.patient_id}>
              {p.patient_id} — {p.full_name} ({p.age}{p.sex ? `, ${p.sex}` : ''})
            </option>
          ))}
        </select>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'compare' && comparison && <CompareTab data={comparison} />}
      {tab === 'details' && comparison && <DetailsTab data={comparison} />}
      {tab === 'definitions' && defs && <DefinitionsTab data={defs} />}
    </div>
  )
}

function CompareTab({ data }) {
  const a = data.patient_a || {}
  const b = data.patient_b || {}
  const radar = data.radar_comparison || []

  const demoA = a.demographics || {}
  const demoB = b.demographics || {}

  const demoFields = [
    { label: 'Full Name', key: 'full_name' },
    { label: 'Age', key: 'age' },
    { label: 'Sex', key: 'sex' },
    { label: 'Epilepsy Type', key: 'epilepsy_type' },
    { label: 'Onset Age', key: 'epilepsy_onset_age' },
    { label: 'Years with Epilepsy', key: 'years_with_epilepsy' },
    { label: 'BMI', key: 'bmi' },
    { label: 'Blood Type', key: 'blood_type' },
    { label: 'Education', key: 'education_level' },
    { label: 'Insurance', key: 'insurance_type' },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      {/* Radar chart */}
      <Card title="Comparison Radar" span={2}>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radar}>
            <PolarGrid />
            <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Radar name={`A: ${a.patient_id}`} dataKey="patient_a" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
            <Radar name={`B: ${b.patient_id}`} dataKey="patient_b" stroke="#16a34a" fill="#16a34a" fillOpacity={0.2} />
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      {/* Demographics comparison table */}
      <Card title="Demographics Comparison" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Field</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#3b82f6' }}>Patient A ({a.patient_id})</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#16a34a' }}>Patient B ({b.patient_id})</th>
            </tr>
          </thead>
          <tbody>
            {demoFields.map(f => (
              <tr key={f.key}>
                <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{f.label}</td>
                <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{fmt(demoA[f.key])}</td>
                <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{fmt(demoB[f.key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Seizure comparison */}
      <Card title={`Seizures — A: ${a.patient_id}`}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Total Events</span><div style={{ fontWeight: 700 }}>{fmt(a.seizure_summary?.total_events)}</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Avg Duration</span><div style={{ fontWeight: 700 }}>{fmt(a.seizure_summary?.avg_duration_sec)}s</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>ER Visits</span><div style={{ fontWeight: 700 }}>{fmt(a.seizure_summary?.er_visits)}</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Injuries</span><div style={{ fontWeight: 700 }}>{fmt(a.seizure_summary?.injuries)}</div></div>
        </div>
        {(a.seizure_triggers || []).length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Top Triggers</div>
            {a.seizure_triggers.map((t, i) => (
              <div key={i} style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>{t.trigger}</span><span style={{ fontWeight: 600 }}>{t.count}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card title={`Seizures — B: ${b.patient_id}`}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Total Events</span><div style={{ fontWeight: 700 }}>{fmt(b.seizure_summary?.total_events)}</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Avg Duration</span><div style={{ fontWeight: 700 }}>{fmt(b.seizure_summary?.avg_duration_sec)}s</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>ER Visits</span><div style={{ fontWeight: 700 }}>{fmt(b.seizure_summary?.er_visits)}</div></div>
          <div><span style={{ fontSize: 11, color: '#64748b' }}>Injuries</span><div style={{ fontWeight: 700 }}>{fmt(b.seizure_summary?.injuries)}</div></div>
        </div>
        {(b.seizure_triggers || []).length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Top Triggers</div>
            {b.seizure_triggers.map((t, i) => (
              <div key={i} style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span>{t.trigger}</span><span style={{ fontWeight: 600 }}>{t.count}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Medication adherence comparison */}
      <Card title="Medication Adherence" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div>
            <div style={{ fontSize: 12, color: '#3b82f6', fontWeight: 600, marginBottom: 4 }}>Patient A ({a.patient_id})</div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>{fmt(a.medication_adherence?.avg_adherence)}%</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>Taken: {fmt(a.medication_adherence?.doses_taken)} / Missed: {fmt(a.medication_adherence?.doses_missed)}</div>
            <div style={{ marginTop: 6, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${a.medication_adherence?.avg_adherence || 0}%`, background: '#3b82f6', borderRadius: 4 }} />
            </div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#16a34a', fontWeight: 600, marginBottom: 4 }}>Patient B ({b.patient_id})</div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>{fmt(b.medication_adherence?.avg_adherence)}%</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>Taken: {fmt(b.medication_adherence?.doses_taken)} / Missed: {fmt(b.medication_adherence?.doses_missed)}</div>
            <div style={{ marginTop: 6, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${b.medication_adherence?.avg_adherence || 0}%`, background: '#16a34a', borderRadius: 4 }} />
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

function DetailsTab({ data }) {
  const a = data.patient_a || {}
  const b = data.patient_b || {}

  // Cognitive domain comparison bar chart
  const cogDomains = new Set([
    ...(a.cognitive_domains || []).map(d => d.domain),
    ...(b.cognitive_domains || []).map(d => d.domain)
  ])
  const cogData = [...cogDomains].map(domain => {
    const aEntry = (a.cognitive_domains || []).find(d => d.domain === domain) || {}
    const bEntry = (b.cognitive_domains || []).find(d => d.domain === domain) || {}
    return {
      domain,
      patient_a: aEntry.avg_accuracy || 0,
      patient_b: bEntry.avg_accuracy || 0,
    }
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Cognitive domains bar chart */}
      {cogData.length > 0 && (
        <Card title="Cognitive Domain Accuracy (%)">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={cogData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
              <YAxis dataKey="domain" type="category" width={100} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="patient_a" name={`A: ${a.patient_id}`} fill="#3b82f6" radius={[0, 4, 4, 0]} />
              <Bar dataKey="patient_b" name={`B: ${b.patient_id}`} fill="#16a34a" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Assessments comparison */}
      <Card title="Assessment Scores">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#3b82f6', fontWeight: 600, marginBottom: 8 }}>Patient A ({a.patient_id})</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Instrument</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Score</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Level</th>
                </tr>
              </thead>
              <tbody>
                {(a.assessments || []).slice(0, 10).map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9' }}>{row.instrument}</td>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9', textAlign: 'right' }}>{row.score}/{row.max_score}</td>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                        background: row.alert ? '#fee2e2' : '#f1f5f9', color: row.alert ? '#991b1b' : '#475569'
                      }}>{row.level || row.interpretation || '--'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#16a34a', fontWeight: 600, marginBottom: 8 }}>Patient B ({b.patient_id})</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Instrument</th>
                  <th style={{ padding: '6px 8px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Score</th>
                  <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Level</th>
                </tr>
              </thead>
              <tbody>
                {(b.assessments || []).slice(0, 10).map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9' }}>{row.instrument}</td>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9', textAlign: 'right' }}>{row.score}/{row.max_score}</td>
                    <td style={{ padding: '4px 8px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                        background: row.alert ? '#fee2e2' : '#f1f5f9', color: row.alert ? '#991b1b' : '#475569'
                      }}>{row.level || row.interpretation || '--'}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      {/* EEG Analysis comparison */}
      <Card title="EEG Analysis Results">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#3b82f6', fontWeight: 600, marginBottom: 8 }}>Patient A ({a.patient_id})</div>
            {(a.analyses || []).length === 0 ? <div style={{ fontSize: 12, color: '#94a3b8' }}>No EEG analyses</div> :
              (a.analyses || []).map((r, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                  <span style={{ fontWeight: 600 }}>{r.predicted_label}</span>
                  <span style={{ color: '#64748b', marginLeft: 8 }}>conf: {fmt(r.avg_confidence)}</span>
                  <span style={{ color: '#94a3b8', marginLeft: 8 }}>({r.total_analyses} analyses)</span>
                </div>
              ))
            }
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#16a34a', fontWeight: 600, marginBottom: 8 }}>Patient B ({b.patient_id})</div>
            {(b.analyses || []).length === 0 ? <div style={{ fontSize: 12, color: '#94a3b8' }}>No EEG analyses</div> :
              (b.analyses || []).map((r, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                  <span style={{ fontWeight: 600 }}>{r.predicted_label}</span>
                  <span style={{ color: '#64748b', marginLeft: 8 }}>conf: {fmt(r.avg_confidence)}</span>
                  <span style={{ color: '#94a3b8', marginLeft: 8 }}>({r.total_analyses} analyses)</span>
                </div>
              ))
            }
          </div>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Comparison Dimensions" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          {(data.comparison_dimensions || []).map((d, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{d.dimension}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{d.description}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Glossary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <tbody>
            {(data.glossary || []).map((g, i) => (
              <tr key={i}>
                <td style={{ padding: '6px 8px', borderBottom: '1px solid #f1f5f9', fontWeight: 600, color: '#334155', width: 120 }}>{g.term}</td>
                <td style={{ padding: '6px 8px', borderBottom: '1px solid #f1f5f9', color: '#64748b' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Clinical Notes">
        <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: '#64748b', lineHeight: 1.8 }}>
          {(data.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      </Card>
    </div>
  )
}
