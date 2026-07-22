import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PATTERN_COLORS = {
  normal: '#16a34a', peripheral_hearing_loss: '#f59e0b', auditory_neuropathy: '#8b5cf6',
  brainstem_lesion: '#ef4444', acoustic_neuroma: '#ec4899', central_auditory_dysfunction: '#dc2626'
}
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444']

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
  const labels = {
    normal: 'Normal', peripheral_hearing_loss: 'Peripheral HL',
    auditory_neuropathy: 'Auditory Neuropathy', brainstem_lesion: 'Brainstem Lesion',
    acoustic_neuroma: 'Acoustic Neuroma', central_auditory_dysfunction: 'Central Dysfunction'
  }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{labels[pattern] || pattern || 'Unknown'}</span>
  )
}

function RefIndicator({ value, ref_val, unit, mode }) {
  if (value == null) return <span>--</span>
  const abnormal = mode === 'lower' ? value < ref_val : value > ref_val
  return (
    <span style={{ color: abnormal ? '#ef4444' : '#16a34a', fontWeight: abnormal ? 600 : 400 }}>
      {fmt(value)} {unit} <span style={{ fontSize: 10, color: '#94a3b8' }}>
        ({mode === 'lower' ? '\u2265' : '\u2264'}{ref_val})
      </span>
    </span>
  )
}

export default function BeraDashboard() {
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
          axios.get(`${API_URL}/api/bera/overview`),
          axios.get(`${API_URL}/api/bera/breakdown`),
          axios.get(`${API_URL}/api/bera/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading BERA data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'BERA Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const earRates = overview?.ear_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const leftSummary = breakdown?.left_summary || {}
  const rightSummary = breakdown?.right_summary || {}
  const waveVHist = breakdown?.wave_v_latency_histogram || []
  const ampHist = breakdown?.wave_v_amplitude_histogram || []
  const iplHist = breakdown?.ipl_i_v_histogram || []
  const earComp = breakdown?.ear_comparison || []
  const interAuralHist = breakdown?.inter_aural_histogram || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Brainstem Evoked Response Audiometry (BERA)
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Wave I-V peak identification &amp; inter-peak latency analysis — brainstem auditory pathway integrity scoring
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

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Studies" value={kpis.total_studies} />
              <KPI label="Abnormal" value={kpis.abnormal_count} color="#ef4444" />
              <KPI label="Abnormal Rate" value={`${kpis.abnormal_rate_pct}%`} color={kpis.abnormal_rate_pct > 40 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean Wave V" value={`${kpis.mean_wave_v_latency_ms} ms`} sub={`ref \u22646.00`} color={kpis.mean_wave_v_latency_ms > 6.0 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean V Amp" value={`${kpis.mean_wave_v_amplitude_uv} \u00b5V`} sub={`ref \u22650.25`} />
              <KPI label="Mean I-V IPL" value={`${kpis.mean_ipl_i_v_ms} ms`} sub={`ref \u22644.50`} color={kpis.mean_ipl_i_v_ms > 4.5 ? '#ef4444' : '#16a34a'} />
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
                <YAxis type="category" dataKey="label" width={160} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.map((d, i) => <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Ear Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={earRates} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="ear" width={80} tick={{ fontSize: 12 }} />
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
                    <th style={thStyle}>Inter-Aural Diff</th>
                    <th style={thStyle}>Abnormal Ears</th>
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
                      <td style={tdStyle}>
                        <span style={{ color: p.inter_aural_diff_ms > 0.3 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {fmt(p.inter_aural_diff_ms)} ms
                        </span>
                      </td>
                      <td style={tdStyle}>{p.abnormal_ears} / {p.total_ears}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* BERA Analysis Tab */}
      {tab === 'analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Left Ear Summary */}
          <Card title={leftSummary.ear || 'Left Ear'}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr><td style={tdStyle}>Wave I</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_wave_i_ms} ref_val={1.80} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Wave III</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_wave_iii_ms} ref_val={4.00} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Wave V</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_wave_v_ms} ref_val={6.00} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>V Amplitude</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_wave_v_amp_uv} ref_val={0.25} unit={'\u00b5V'} mode="lower" /></td></tr>
                <tr><td style={tdStyle}>I-III IPL</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_ipl_i_iii_ms} ref_val={2.50} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>III-V IPL</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_ipl_iii_v_ms} ref_val={2.20} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>I-V IPL</td><td style={tdStyle}><RefIndicator value={leftSummary.mean_ipl_i_v_ms} ref_val={4.50} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Abnormal</td><td style={tdStyle}><span style={{ fontWeight: 600, color: leftSummary.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{leftSummary.abnormal_pct}%</span></td></tr>
              </tbody>
            </table>
          </Card>

          {/* Right Ear Summary */}
          <Card title={rightSummary.ear || 'Right Ear'}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr><td style={tdStyle}>Wave I</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_wave_i_ms} ref_val={1.80} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Wave III</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_wave_iii_ms} ref_val={4.00} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Wave V</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_wave_v_ms} ref_val={6.00} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>V Amplitude</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_wave_v_amp_uv} ref_val={0.25} unit={'\u00b5V'} mode="lower" /></td></tr>
                <tr><td style={tdStyle}>I-III IPL</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_ipl_i_iii_ms} ref_val={2.50} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>III-V IPL</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_ipl_iii_v_ms} ref_val={2.20} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>I-V IPL</td><td style={tdStyle}><RefIndicator value={rightSummary.mean_ipl_i_v_ms} ref_val={4.50} unit="ms" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Abnormal</td><td style={tdStyle}><span style={{ fontWeight: 600, color: rightSummary.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{rightSummary.abnormal_pct}%</span></td></tr>
              </tbody>
            </table>
          </Card>

          <Card title="Wave V Latency Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={waveVHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {waveVHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = abnormal (&gt; 6.0 ms)
            </p>
          </Card>

          <Card title="Wave V Amplitude Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ampHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {ampHist.map((d, i) => <Cell key={i} fill={d.low_range ? '#ef4444' : '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = low amplitude (&lt; 0.30 {'\u00b5V'})
            </p>
          </Card>

          <Card title="I-V IPL Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={iplHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {iplHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = prolonged I-V IPL (&gt; 4.5 ms)
            </p>
          </Card>

          <Card title="Inter-Aural Wave V Difference">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={interAuralHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {interAuralHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = asymmetry (&gt; 0.3 ms)
            </p>
          </Card>

          <Card title="Left vs Right Ear Comparison" span={2}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={earComp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ear" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="abnormal" fill="#ef4444" name="Abnormal" radius={[4, 4, 0, 0]} />
                <Bar dataKey="total" fill="#e2e8f0" name="Total" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Patient Detail Tab */}
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
                  {p.inter_aural_abnormal && (
                    <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: '#fef2f2', color: '#ef4444', fontWeight: 600 }}>
                      Asymmetry
                    </span>
                  )}
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {['left', 'right'].map(side => {
                      const ear = p[side]
                      if (!ear) return null
                      return (
                        <div key={side} style={{ marginBottom: 16 }}>
                          <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                            {ear.ear} Ear ({side === 'left' ? 'AS' : 'AD'})
                            <SeverityBadge severity={ear.severity} />
                          </h4>
                          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                              <tr>
                                <th style={thStyle}>Parameter</th>
                                <th style={thStyle}>Value</th>
                                <th style={thStyle}>Reference</th>
                                <th style={thStyle}>Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr style={{ background: '#fff' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Wave I</td>
                                <td style={tdStyle}>{fmt(ear.wave_i_latency_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.wave_i_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.wave_i_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.wave_i_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Wave III</td>
                                <td style={tdStyle}>{fmt(ear.wave_iii_latency_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.wave_iii_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.wave_iii_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.wave_iii_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#fff' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Wave V</td>
                                <td style={tdStyle}>{fmt(ear.wave_v_latency_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.wave_v_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.wave_v_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.wave_v_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>V Amplitude</td>
                                <td style={tdStyle}>{fmt(ear.wave_v_amplitude_uv)} {'\u00b5V'}</td>
                                <td style={tdStyle}>{'\u2265'}{ear.wave_v_amp_ref} {'\u00b5V'}</td>
                                <td style={tdStyle}><span style={{ color: ear.wave_v_amp_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.wave_v_amp_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#fff' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>I-III IPL</td>
                                <td style={tdStyle}>{fmt(ear.ipl_i_iii_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.ipl_i_iii_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.ipl_i_iii_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.ipl_i_iii_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>III-V IPL</td>
                                <td style={tdStyle}>{fmt(ear.ipl_iii_v_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.ipl_iii_v_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.ipl_iii_v_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.ipl_iii_v_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#fff' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>I-V IPL</td>
                                <td style={tdStyle}>{fmt(ear.ipl_i_v_ms)} ms</td>
                                <td style={tdStyle}>{'\u2264'}{ear.ipl_i_v_ref} ms</td>
                                <td style={tdStyle}><span style={{ color: ear.ipl_i_v_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{ear.ipl_i_v_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      )
                    })}
                    <div style={{ padding: '8px 10px', background: p.inter_aural_abnormal ? '#fef2f2' : '#f0fdf4', borderRadius: 6, fontSize: 12, fontWeight: 600 }}>
                      Inter-Aural Wave V Difference: {fmt(p.inter_aural_diff_ms)} ms
                      <span style={{ marginLeft: 8, color: p.inter_aural_abnormal ? '#ef4444' : '#16a34a' }}>
                        {p.inter_aural_abnormal ? 'ASYMMETRIC (>0.30 ms)' : 'Symmetric'}
                      </span>
                    </div>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="BERA Study Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Stimulus</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
              <tbody>
                {defs.protocol?.stimulus && Object.entries(defs.protocol.stimulus).map(([k, v], i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize', width: 140 }}>{k.replace(/_/g, ' ')}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Recording</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
              <tbody>
                {defs.protocol?.recording && Object.entries(defs.protocol.recording).map(([k, v], i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize', width: 140 }}>{k.replace(/_/g, ' ')}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{v}</td>
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

          <Card title="BERA Parameters">
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

          <Card title="Reference Ranges">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Upper Limit / Lower Limit</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Wave I Latency</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.wave_i_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Wave III Latency</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.wave_iii_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Wave V Latency</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.wave_v_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Wave V Amplitude</td>
                  <td style={tdStyle}>{'\u2265'} {defs.reference_ranges?.wave_v_amplitude_lower_uv} {'\u00b5V'}</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>I-III IPL</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.ipl_i_iii_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>III-V IPL</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.ipl_iii_v_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>I-V IPL</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.ipl_i_v_upper_ms} ms</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Inter-Aural V Diff</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.inter_aural_v_upper_ms} ms</td>
                </tr>
              </tbody>
            </table>
            <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>{defs.reference_ranges?.notes}</p>
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
