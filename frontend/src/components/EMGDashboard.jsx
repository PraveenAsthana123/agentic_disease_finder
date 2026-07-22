import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PATTERN_COLORS = { normal: '#16a34a', neuropathic: '#8b5cf6', myopathic: '#f59e0b', mixed: '#ef4444', nmj: '#06b6d4' }
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444']

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

function SeverityBadge({ severity }) {
  const color = SEV_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{severity || 'Unknown'}</span>
  )
}

function PatternBadge({ pattern }) {
  const color = PATTERN_COLORS[pattern] || '#94a3b8'
  const labels = { normal: 'Normal', neuropathic: 'Neuropathic', myopathic: 'Myopathic', mixed: 'Mixed', nmj: 'NMJ' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{labels[pattern] || pattern || 'Unknown'}</span>
  )
}

function RangeIndicator({ value, lower, upper, unit }) {
  if (value == null) return <span>--</span>
  const abnormal = (lower != null && value < lower) || (upper != null && value > upper)
  const refStr = lower != null && upper != null ? `${lower}-${upper}` : lower != null ? `>${lower}` : `<${upper}`
  return (
    <span style={{ color: abnormal ? '#ef4444' : '#16a34a', fontWeight: abnormal ? 600 : 400 }}>
      {fmt(value)} <span style={{ fontSize: 10, color: '#94a3b8' }}>({refStr}{unit ? ` ${unit}` : ''})</span>
    </span>
  )
}

export default function EMGDashboard() {
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
          axios.get(`${API_URL}/api/emg/overview`),
          axios.get(`${API_URL}/api/emg/breakdown`),
          axios.get(`${API_URL}/api/emg/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EMG data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'MUAP Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.diagnostic_pattern_distribution || []
  const muscleRates = overview?.muscle_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const muapSummary = breakdown?.muap_summary || []
  const recruitDist = breakdown?.recruitment_distribution || []
  const spontDist = breakdown?.spontaneous_activity_distribution || []
  const durHist = breakdown?.duration_histogram || []
  const ampHist = breakdown?.amplitude_histogram || []
  const limbComp = breakdown?.limb_comparison || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>Electromyography (EMG)</h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        MUAP classification &amp; neuromuscular disorder detection — needle EMG analysis
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Studies" value={kpis.total_studies} />
              <KPI label="Abnormal" value={kpis.abnormal_count} color="#ef4444" />
              <KPI label="Abnormal Rate" value={`${kpis.abnormal_rate_pct}%`} color={kpis.abnormal_rate_pct > 40 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean Duration" value={`${kpis.mean_muap_duration_ms} ms`} sub="MUAP" />
              <KPI label="Mean Amplitude" value={`${kpis.mean_muap_amplitude_uv} uV`} sub="MUAP" />
              <KPI label="Muscles/Study" value={kpis.muscles_tested_per_study} sub="8 muscles" />
            </div>
          </Card>

          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sevDist} dataKey="count" nameKey="severity" cx="50%" cy="50%"
                     outerRadius={80} label={({ severity, count }) => `${severity}: ${count}`}>
                  {sevDist.map((d, i) => <Cell key={i} fill={SEV_COLORS[d.severity] || PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Diagnostic Pattern">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={patternDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={100} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.map((d, i) => <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Muscle Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={muscleRates} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="muscle" width={180} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="rate_pct" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Patient Summary" span={3}>
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Age</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Severity</th>
                    <th style={thStyle}>Pattern</th>
                    <th style={thStyle}>Abnormal Muscles</th>
                  </tr>
                </thead>
                <tbody>
                  {patientSummary.map((p, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{p.name || p.patient_id}</td>
                      <td style={tdStyle}>{p.age}</td>
                      <td style={tdStyle}>{p.disease}</td>
                      <td style={tdStyle}><SeverityBadge severity={p.overall_severity} /></td>
                      <td style={tdStyle}><PatternBadge pattern={p.diagnostic_pattern} /></td>
                      <td style={tdStyle}>{p.abnormal_muscles} / {p.total_muscles}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── MUAP Analysis Tab ─── */}
      {tab === 'analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="MUAP Parameter Summary" span={2}>
            <div style={{ overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Muscle</th>
                    <th style={thStyle}>Limb</th>
                    <th style={thStyle}>Mean Duration (ms)</th>
                    <th style={thStyle}>Mean Amplitude (uV)</th>
                    <th style={thStyle}>Mean Phases</th>
                    <th style={thStyle}>Polyphasic %</th>
                    <th style={thStyle}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {muapSummary.map((m, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, fontSize: 12 }}>{m.muscle}</td>
                      <td style={tdStyle}>{m.limb === 'upper' ? 'Upper' : 'Lower'}</td>
                      <td style={tdStyle}>
                        <RangeIndicator value={m.mean_duration_ms} lower={m.duration_ref_lower} upper={m.duration_ref_upper} />
                      </td>
                      <td style={tdStyle}>
                        <RangeIndicator value={m.mean_amplitude_uv} lower={m.amplitude_ref_lower} upper={m.amplitude_ref_upper} />
                      </td>
                      <td style={tdStyle}>{fmt(m.mean_phases)}</td>
                      <td style={tdStyle}>
                        <span style={{ color: m.polyphasic_pct > 20 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {m.polyphasic_pct}%
                        </span>
                      </td>
                      <td style={tdStyle}>
                        <span style={{ color: m.abnormal_pct > 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {m.abnormal_pct}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recruitment Pattern Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={recruitDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Spontaneous Activity Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={spontDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-15} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="MUAP Duration Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={durHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="MUAP Amplitude Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ampHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Upper vs Lower Limb Abnormality" span={2}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={limbComp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="limb" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="normal" stackId="a" fill="#16a34a" name="Normal" />
                <Bar dataKey="abnormal" stackId="a" fill="#f59e0b" name="Abnormal" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gap: 12 }}>
          {patientDetails.map((p, i) => {
            const isExpanded = expandedPatient === p.patient_id
            return (
              <Card key={i}>
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
                  onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                >
                  <span style={{ fontSize: 18 }}>{isExpanded ? '\u25BC' : '\u25B6'}</span>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{p.name || p.patient_id}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 10 }}>
                      Age {p.age} | {p.disease}
                    </span>
                  </div>
                  <SeverityBadge severity={p.overall_severity} />
                  <PatternBadge pattern={p.diagnostic_pattern} />
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Muscle</th>
                          <th style={thStyle}>Limb</th>
                          <th style={thStyle}>Duration (ms)</th>
                          <th style={thStyle}>Amplitude (uV)</th>
                          <th style={thStyle}>Phases</th>
                          <th style={thStyle}>Recruitment</th>
                          <th style={thStyle}>Spontaneous</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(p.muscles || []).map((r, j) => (
                          <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                            <td style={{ ...tdStyle, fontWeight: 500, fontSize: 12 }}>{r.muscle}</td>
                            <td style={tdStyle}>{r.limb === 'upper' ? 'Upper' : 'Lower'}</td>
                            <td style={tdStyle}>
                              <RangeIndicator value={r.muap_duration_ms} lower={r.duration_ref_lower} upper={r.duration_ref_upper} />
                            </td>
                            <td style={tdStyle}>
                              <RangeIndicator value={r.muap_amplitude_uv} lower={r.amplitude_ref_lower} upper={r.amplitude_ref_upper} />
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: r.polyphasic ? '#ef4444' : '#16a34a', fontWeight: r.polyphasic ? 600 : 400 }}>
                                {r.muap_phases} {r.polyphasic && <span style={{ fontSize: 10 }}>(poly)</span>}
                              </span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: r.recruitment !== 'Normal' ? '#ef4444' : '#16a34a', fontSize: 12 }}>
                                {r.recruitment}
                              </span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: r.spontaneous_activity !== 'None' ? '#ef4444' : '#16a34a', fontSize: 12 }}>
                                {r.spontaneous_activity}
                              </span>
                            </td>
                            <td style={tdStyle}><SeverityBadge severity={r.severity} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="EMG Study Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Muscles Tested</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
              <thead>
                <tr>
                  <th style={thStyle}>Muscle</th>
                  <th style={thStyle}>Innervation</th>
                  <th style={thStyle}>Limb</th>
                </tr>
              </thead>
              <tbody>
                {(defs.protocol?.innervation_map || []).map((m, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{m.muscle}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{m.innervation}</td>
                    <td style={tdStyle}>{m.limb === 'upper' ? 'Upper' : 'Lower'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 12, color: '#94a3b8' }}>Standard: {defs.protocol?.standard}</p>
            <h4 style={{ fontSize: 13, margin: '12px 0 6px', color: '#334155' }}>Indications</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
              {(defs.protocol?.indications || []).map((ind, i) => <li key={i}>{ind}</li>)}
            </ul>
          </Card>

          <Card title="EMG Parameters">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Unit</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.parameters || []).map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name}</td>
                    <td style={tdStyle}>{p.unit}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{p.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="MUAP Reference Ranges">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Normal Range</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Duration</td>
                  <td style={tdStyle}>{defs.reference_ranges?.muap?.duration_ms?.normal_range}</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Amplitude</td>
                  <td style={tdStyle}>{defs.reference_ranges?.muap?.amplitude_uv?.normal_range}</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Phases</td>
                  <td style={tdStyle}>{defs.reference_ranges?.muap?.phases?.polyphasic_threshold} = polyphasic</td>
                </tr>
              </tbody>
            </table>
          </Card>

          <Card title="Diagnostic Patterns">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Pattern</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.diagnostic_patterns || []).map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><PatternBadge pattern={p.pattern} /></td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{p.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Severity Levels">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Level</th>
                  <th style={thStyle}>Criteria</th>
                </tr>
              </thead>
              <tbody>
                {(defs.severity_levels || []).map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><SeverityBadge severity={s.level} /></td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{s.criteria}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Significance">
            <p style={{ fontSize: 13, color: '#475569', margin: 0, lineHeight: 1.6 }}>{defs.clinical_significance}</p>
            <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 12 }}>Reference: {defs.reference}</p>
          </Card>
        </div>
      )}
    </div>
  )
}
