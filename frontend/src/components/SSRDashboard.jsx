import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PATTERN_COLORS = {
  normal: '#16a34a', peripheral_neuropathy: '#f59e0b', small_fiber_neuropathy: '#8b5cf6',
  postganglionic_lesion: '#ef4444', preganglionic_lesion: '#ec4899',
  generalized_dysautonomia: '#dc2626'
}
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444']
const SCORE_COLORS = { normal: '#16a34a', mild: '#3b82f6', moderate: '#eab308', severe: '#ef4444', very_severe: '#991b1b' }

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
    normal: 'Normal', peripheral_neuropathy: 'Peripheral Neuropathy',
    small_fiber_neuropathy: 'Small Fiber Neuropathy',
    postganglionic_lesion: 'Postganglionic', preganglionic_lesion: 'Preganglionic',
    generalized_dysautonomia: 'Gen. Dysautonomia'
  }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{labels[pattern] || pattern || 'Unknown'}</span>
  )
}

function RefIndicator({ value, ref_val, unit, mode, absent }) {
  if (absent) return <span style={{ color: '#ef4444', fontWeight: 700 }}>ABSENT</span>
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

export default function SSRDashboard() {
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
          axios.get(`${API_URL}/ssr/overview`),
          axios.get(`${API_URL}/ssr/breakdown`),
          axios.get(`${API_URL}/ssr/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading SSR data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'SSR Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const siteRates = overview?.site_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const handSummary = breakdown?.hand_summary || {}
  const footSummary = breakdown?.foot_summary || {}
  const handLatHist = breakdown?.hand_latency_histogram || []
  const footLatHist = breakdown?.foot_latency_histogram || []
  const handAmpHist = breakdown?.hand_amplitude_histogram || []
  const scoreHist = breakdown?.score_histogram || []
  const siteComp = breakdown?.site_comparison || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Sympathetic Skin Response (SSR)
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Latency/amplitude auto-measurement &amp; dysautonomia screening — sympathetic sudomotor pathway integrity
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
              <KPI label="Mean Hand Lat" value={`${kpis.mean_hand_latency_s} s`} sub={`ref \u22641.50`} color={kpis.mean_hand_latency_s > 1.5 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean Foot Lat" value={`${kpis.mean_foot_latency_s} s`} sub={`ref \u22642.20`} color={kpis.mean_foot_latency_s > 2.2 ? '#ef4444' : '#16a34a'} />
              <KPI label="Dysautonomia Score" value={kpis.mean_dysautonomia_score} sub="0-100 scale" color={kpis.mean_dysautonomia_score > 25 ? '#ef4444' : '#16a34a'} />
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

          <Card title="Per-Site Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={siteRates} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="site" width={80} tick={{ fontSize: 12 }} />
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
                    <th style={thStyle}>Dysautonomia</th>
                    <th style={thStyle}>Abnormal Sites</th>
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
                        <span style={{ fontWeight: 600, color: p.dysautonomia_score > 25 ? '#ef4444' : '#16a34a' }}>
                          {fmt(p.dysautonomia_score)}
                        </span>
                      </td>
                      <td style={tdStyle}>
                        {p.abnormal_sites} / {p.total_sites}
                        {p.any_absent && <span style={{ marginLeft: 6, fontSize: 10, padding: '1px 4px', borderRadius: 3, background: '#fef2f2', color: '#ef4444', fontWeight: 600 }}>ABSENT</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* SSR Analysis Tab */}
      {tab === 'analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Hand Summary */}
          <Card title={handSummary.site || 'Hand (Palmar)'}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr><td style={tdStyle}>Latency</td><td style={tdStyle}><RefIndicator value={handSummary.mean_latency_s} ref_val={1.50} unit="s" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Amplitude</td><td style={tdStyle}><RefIndicator value={handSummary.mean_amplitude_mv} ref_val={0.50} unit="mV" mode="lower" /></td></tr>
                <tr><td style={tdStyle}>Habituation</td><td style={tdStyle}><RefIndicator value={handSummary.mean_habituation_pct} ref_val={50} unit="%" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Present</td><td style={tdStyle}><span style={{ fontWeight: 600 }}>{handSummary.present_count}</span> / {handSummary.count}</td></tr>
                <tr><td style={tdStyle}>Absent</td><td style={tdStyle}><span style={{ fontWeight: 600, color: handSummary.absent_count > 0 ? '#ef4444' : '#16a34a' }}>{handSummary.absent_count}</span></td></tr>
                <tr><td style={tdStyle}>Abnormal</td><td style={tdStyle}><span style={{ fontWeight: 600, color: handSummary.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{handSummary.abnormal_pct}%</span></td></tr>
              </tbody>
            </table>
          </Card>

          {/* Foot Summary */}
          <Card title={footSummary.site || 'Foot (Plantar)'}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <tbody>
                <tr><td style={tdStyle}>Latency</td><td style={tdStyle}><RefIndicator value={footSummary.mean_latency_s} ref_val={2.20} unit="s" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Amplitude</td><td style={tdStyle}><RefIndicator value={footSummary.mean_amplitude_mv} ref_val={0.20} unit="mV" mode="lower" /></td></tr>
                <tr><td style={tdStyle}>Habituation</td><td style={tdStyle}><RefIndicator value={footSummary.mean_habituation_pct} ref_val={50} unit="%" mode="upper" /></td></tr>
                <tr><td style={tdStyle}>Present</td><td style={tdStyle}><span style={{ fontWeight: 600 }}>{footSummary.present_count}</span> / {footSummary.count}</td></tr>
                <tr><td style={tdStyle}>Absent</td><td style={tdStyle}><span style={{ fontWeight: 600, color: footSummary.absent_count > 0 ? '#ef4444' : '#16a34a' }}>{footSummary.absent_count}</span></td></tr>
                <tr><td style={tdStyle}>Abnormal</td><td style={tdStyle}><span style={{ fontWeight: 600, color: footSummary.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{footSummary.abnormal_pct}%</span></td></tr>
              </tbody>
            </table>
          </Card>

          <Card title="Hand Latency Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={handLatHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {handLatHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = abnormal (&gt; 1.50 s)
            </p>
          </Card>

          <Card title="Foot Latency Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={footLatHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {footLatHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = abnormal (&gt; 2.20 s)
            </p>
          </Card>

          <Card title="Hand Amplitude Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={handAmpHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {handAmpHist.map((d, i) => <Cell key={i} fill={d.low_range ? '#ef4444' : '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = low amplitude (&lt; 0.50 mV)
            </p>
          </Card>

          <Card title="Dysautonomia Score Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={scoreHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {scoreHist.map((d, i) => <Cell key={i} fill={SCORE_COLORS[d.grade] || '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Color by severity grade (0-100 scale)
            </p>
          </Card>

          <Card title="Hand vs Foot Comparison" span={2}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={siteComp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="site" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="abnormal" fill="#ef4444" name="Abnormal" radius={[4, 4, 0, 0]} />
                <Bar dataKey="absent" fill="#991b1b" name="Absent" radius={[4, 4, 0, 0]} />
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
                  <span style={{ fontSize: 11, color: p.dysautonomia_score > 25 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                    Score: {fmt(p.dysautonomia_score)}
                  </span>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {['hand', 'foot'].map(siteKey => {
                      const site = p[siteKey]
                      if (!site) return null
                      return (
                        <div key={siteKey} style={{ marginBottom: 16 }}>
                          <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                            {site.site} ({site.recording}){' '}
                            <SeverityBadge severity={site.severity} />
                            {site.absent && (
                              <span style={{ marginLeft: 8, fontSize: 10, padding: '2px 6px', borderRadius: 4, background: '#fef2f2', color: '#ef4444', fontWeight: 700 }}>
                                ABSENT
                              </span>
                            )}
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
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Onset Latency</td>
                                <td style={tdStyle}>{site.absent ? 'N/A' : `${fmt(site.onset_latency_s)} s`}</td>
                                <td style={tdStyle}>{'\u2264'}{site.latency_ref_s} s</td>
                                <td style={tdStyle}><span style={{ color: site.absent || site.latency_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{site.absent ? 'ABSENT' : site.latency_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Amplitude</td>
                                <td style={tdStyle}>{site.absent ? 'N/A' : `${fmt(site.amplitude_mv)} mV`}</td>
                                <td style={tdStyle}>{'\u2265'}{site.amplitude_ref_mv} mV</td>
                                <td style={tdStyle}><span style={{ color: site.amplitude_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{site.absent ? 'ABSENT' : site.amplitude_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                              <tr style={{ background: '#fff' }}>
                                <td style={{ ...tdStyle, fontWeight: 600 }}>Habituation</td>
                                <td style={tdStyle}>{site.absent ? 'N/A' : `${fmt(site.habituation_pct)}%`}</td>
                                <td style={tdStyle}>&lt;{site.habituation_ref_pct}%</td>
                                <td style={tdStyle}><span style={{ color: site.habituation_abnormal ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{site.absent ? 'N/A' : site.habituation_abnormal ? 'ABNORMAL' : 'Normal'}</span></td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      )
                    })}
                    <div style={{ padding: '8px 10px', background: p.dysautonomia_score > 25 ? '#fef2f2' : '#f0fdf4', borderRadius: 6, fontSize: 12, fontWeight: 600 }}>
                      Dysautonomia Score: {fmt(p.dysautonomia_score)} / 100
                      <span style={{ marginLeft: 8, color: p.dysautonomia_score > 50 ? '#ef4444' : p.dysautonomia_score > 25 ? '#eab308' : '#16a34a' }}>
                        {p.dysautonomia_score > 50 ? 'SEVERE' : p.dysautonomia_score > 25 ? 'MODERATE' : p.dysautonomia_score > 10 ? 'MILD' : 'NORMAL'}
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
          <Card title="SSR Study Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Stimulus</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
              <tbody>
                {defs.protocol?.stimulus && Object.entries(defs.protocol.stimulus).map(([k, v], i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize', width: 180 }}>{k.replace(/_/g, ' ')}</td>
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
                    <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize', width: 180 }}>{k.replace(/_/g, ' ')}</td>
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

          <Card title="SSR Parameters">
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
                  <th style={thStyle}>Limit</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Hand Onset Latency</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.hand_latency_upper_s} s</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Hand Amplitude</td>
                  <td style={tdStyle}>{'\u2265'} {defs.reference_ranges?.hand_amplitude_lower_mv} mV</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Foot Onset Latency</td>
                  <td style={tdStyle}>{'\u2264'} {defs.reference_ranges?.foot_latency_upper_s} s</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Foot Amplitude</td>
                  <td style={tdStyle}>{'\u2265'} {defs.reference_ranges?.foot_amplitude_lower_mv} mV</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Habituation</td>
                  <td style={tdStyle}>&lt; {defs.reference_ranges?.habituation_upper_pct}%</td>
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
