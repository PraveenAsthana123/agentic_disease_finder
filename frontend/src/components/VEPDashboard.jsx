import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444']
const PATTERN_COLORS = {
  normal: '#16a34a', optic_neuritis: '#3b82f6', optic_neuropathy: '#8b5cf6',
  chiasmal_lesion: '#f59e0b', retrochiasmal_lesion: '#ef4444', cortical_dysfunction: '#dc2626'
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
  const label = pattern ? pattern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function RefIndicator({ value, refVal, higher_is_bad }) {
  if (value == null || refVal == null) return <span>{fmt(value)}</span>
  const abnormal = higher_is_bad ? value > refVal : value < refVal
  return (
    <span style={{ color: abnormal ? '#ef4444' : '#16a34a', fontWeight: abnormal ? 600 : 400 }}>
      {fmt(value)} <span style={{ fontSize: 10, color: '#94a3b8' }}>({higher_is_bad ? '<' : '>'}{fmt(refVal)})</span>
    </span>
  )
}

export default function VEPDashboard() {
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
          axios.get(`${API_URL}/api/vep/overview`),
          axios.get(`${API_URL}/api/vep/breakdown`),
          axios.get(`${API_URL}/api/vep/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading VEP data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'eyes', label: 'Eye Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const eyeRates = overview?.eye_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const leftSummary = breakdown?.left_summary || {}
  const rightSummary = breakdown?.right_summary || {}
  const p100LatHist = breakdown?.p100_latency_histogram || []
  const p100AmpHist = breakdown?.p100_amplitude_histogram || []
  const eyeComp = breakdown?.eye_comparison || []
  const interEyeHist = breakdown?.inter_eye_histogram || []
  const patientDetails = breakdown?.patient_details || []

  const protocol = defs?.protocol || {}
  const parameters = defs?.parameters || []
  const refRanges = defs?.reference_ranges || {}
  const diagPatterns = defs?.diagnostic_patterns || []
  const sevLevels = defs?.severity_levels || []
  const clinSig = defs?.clinical_significance || []
  const reference = defs?.reference || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Visual Evoked Potentials (VEP)
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Visual pathway assessment: N75/P100/N145 waveform analysis with per-eye severity grading and inter-eye comparison
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
              <KPI label="Abnormal Rate" value={`${kpis.abnormal_rate_pct}%`}
                   color={kpis.abnormal_rate_pct > 40 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean P100 Latency" value={`${fmt(kpis.mean_p100_latency_ms)} ms`} sub="Cortical peak" />
              <KPI label="Mean P100 Amp" value={`${fmt(kpis.mean_p100_amplitude_uv)} uV`} sub="Peak amplitude" />
              <KPI label="Eyes/Study" value={kpis.eyes_per_study} sub="Left + Right" />
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
              <BarChart data={patternDist.filter(d => d.count > 0)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Eye Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={eyeRates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="eye" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="rate_pct" fill="#f59e0b" radius={[4, 4, 0, 0]} />
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
                    <th style={thStyle}>Inter-eye Diff</th>
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
                        <span style={{ color: p.inter_eye_abnormal ? '#ef4444' : '#16a34a', fontWeight: p.inter_eye_abnormal ? 600 : 400 }}>
                          {fmt(p.inter_eye_diff_ms)} ms
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Eye Analysis Tab ─── */}
      {tab === 'eyes' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Left Eye Summary (OS)">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
              <KPI label="Mean N75" value={`${fmt(leftSummary.mean_n75_ms)} ms`} sub="Initial negative" />
              <KPI label="Mean P100" value={`${fmt(leftSummary.mean_p100_ms)} ms`} sub="Major positive" />
              <KPI label="Mean N145" value={`${fmt(leftSummary.mean_n145_ms)} ms`} sub="Late negative" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <KPI label="P100 Amplitude" value={`${fmt(leftSummary.mean_p100_amp_uv)} uV`} />
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontWeight: 600, color: leftSummary.abnormal_pct > 20 ? '#ef4444' : '#16a34a', fontSize: 18 }}>
                  {leftSummary.abnormal_pct}% abnormal
                </span>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>({leftSummary.count} studies)</div>
              </div>
            </div>
          </Card>

          <Card title="Right Eye Summary (OD)">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
              <KPI label="Mean N75" value={`${fmt(rightSummary.mean_n75_ms)} ms`} sub="Initial negative" />
              <KPI label="Mean P100" value={`${fmt(rightSummary.mean_p100_ms)} ms`} sub="Major positive" />
              <KPI label="Mean N145" value={`${fmt(rightSummary.mean_n145_ms)} ms`} sub="Late negative" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <KPI label="P100 Amplitude" value={`${fmt(rightSummary.mean_p100_amp_uv)} uV`} />
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontWeight: 600, color: rightSummary.abnormal_pct > 20 ? '#ef4444' : '#16a34a', fontSize: 18 }}>
                  {rightSummary.abnormal_pct}% abnormal
                </span>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>({rightSummary.count} studies)</div>
              </div>
            </div>
          </Card>

          <Card title="P100 Latency Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={p100LatHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="P100 Amplitude Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={p100AmpHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} label={{ value: 'uV', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Left vs Right Eye Comparison" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Eye</th>
                  <th style={thStyle}>Total Studies</th>
                  <th style={thStyle}>Abnormal</th>
                  <th style={thStyle}>Abnormal %</th>
                  <th style={thStyle}>Mean P100 Latency</th>
                  <th style={thStyle}>Mean P100 Amp</th>
                </tr>
              </thead>
              <tbody>
                {eyeComp.map((e, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{e.eye}</td>
                    <td style={tdStyle}>{e.total}</td>
                    <td style={tdStyle}>{e.abnormal}</td>
                    <td style={tdStyle}>
                      <span style={{ color: e.abnormal_pct > 20 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                        {e.abnormal_pct}%
                      </span>
                    </td>
                    <td style={tdStyle}>{fmt(e.mean_p100_ms)} ms</td>
                    <td style={tdStyle}>{fmt(e.mean_p100_amp_uv)} uV</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Inter-eye P100 Difference Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={interEyeHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Abnormal threshold: &gt;8.0 ms inter-eye difference
            </div>
          </Card>

          {/* Left eye severity distribution */}
          {leftSummary.severity_dist && (
            <Card title="Left Eye Severity">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={Object.entries(leftSummary.severity_dist).map(([s, c]) => ({ severity: s, count: c }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {Object.entries(leftSummary.severity_dist).map(([s], i) => (
                      <Cell key={i} fill={SEV_COLORS[s] || '#6366f1'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
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
                  {p.inter_eye_abnormal && (
                    <span style={{ fontSize: 11, color: '#ef4444', fontWeight: 600 }}>
                      Inter-eye: {fmt(p.inter_eye_diff_ms)} ms
                    </span>
                  )}
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Left Eye */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Left Eye (OS)</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Peak</th>
                          <th style={thStyle}>Value</th>
                          <th style={thStyle}>Reference</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N75 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.left.n75_latency_ms} refVal={p.left.n75_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.left.n75_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.left.n75_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P100 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.left.p100_latency_ms} refVal={p.left.p100_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.left.p100_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.left.p100_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N145 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.left.n145_latency_ms} refVal={p.left.n145_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.left.n145_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.left.n145_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P100 Amplitude</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.left.p100_amplitude_uv} refVal={p.left.p100_amp_ref} higher_is_bad={false} />
                          </td>
                          <td style={tdStyle}>{fmt(p.left.p100_amp_ref)} uV</td>
                          <td style={tdStyle}><SeverityBadge severity={p.left.p100_amp_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                      </tbody>
                    </table>

                    {/* Right Eye */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Right Eye (OD)</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Peak</th>
                          <th style={thStyle}>Value</th>
                          <th style={thStyle}>Reference</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N75 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.right.n75_latency_ms} refVal={p.right.n75_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.right.n75_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.right.n75_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P100 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.right.p100_latency_ms} refVal={p.right.p100_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.right.p100_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.right.p100_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N145 Latency</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.right.n145_latency_ms} refVal={p.right.n145_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.right.n145_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.right.n145_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P100 Amplitude</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.right.p100_amplitude_uv} refVal={p.right.p100_amp_ref} higher_is_bad={false} />
                          </td>
                          <td style={tdStyle}>{fmt(p.right.p100_amp_ref)} uV</td>
                          <td style={tdStyle}><SeverityBadge severity={p.right.p100_amp_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                      </tbody>
                    </table>

                    {/* Inter-eye comparison */}
                    <div style={{ fontSize: 12, color: '#64748b', padding: '8px 10px', background: '#f1f5f9', borderRadius: 6 }}>
                      Inter-eye P100 difference:{' '}
                      <span style={{ fontWeight: 600, color: p.inter_eye_abnormal ? '#ef4444' : '#16a34a' }}>
                        {fmt(p.inter_eye_diff_ms)} ms
                      </span>
                      <span style={{ color: '#94a3b8', marginLeft: 8 }}>(threshold: 8.0 ms)</span>
                      &nbsp;|&nbsp;
                      <SeverityBadge severity={p.left.severity} /> (Left) &nbsp;|&nbsp; <SeverityBadge severity={p.right.severity} /> (Right)
                    </div>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {/* Protocol */}
          <Card title="Protocol">
            <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6 }}>{protocol.description}</p>
            {protocol.stimulus && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 6px' }}>Stimulus Parameters</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(protocol.stimulus).map(([k, v], i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, width: '30%' }}>{k.replace(/_/g, ' ')}</td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {protocol.recording && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 6px' }}>Recording Setup</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(protocol.recording).map(([k, v], i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, width: '30%' }}>{k.replace(/_/g, ' ')}</td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {protocol.indications && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 6px' }}>Clinical Indications</h4>
                <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                  {protocol.indications.map((ind, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>{ind}</li>
                  ))}
                </ul>
              </div>
            )}
          </Card>

          {/* Parameters */}
          <Card title="VEP Parameters">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Unit</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {parameters.map((param, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{param.name}</td>
                    <td style={tdStyle}>{param.unit}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{param.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Reference Ranges */}
          <Card title="Reference Ranges">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Threshold</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(refRanges).filter(([k]) => k !== 'notes').map(([k, v], i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{k.replace(/_/g, ' ')}</td>
                    <td style={tdStyle}>{typeof v === 'number' ? `${v} ${k.includes('amplitude') ? 'uV' : 'ms'}` : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {refRanges.notes && (
              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>{refRanges.notes}</p>
            )}
          </Card>

          {/* Diagnostic Patterns */}
          <Card title="Diagnostic Patterns">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Pattern</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {diagPatterns.map((dp, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}><PatternBadge pattern={dp.pattern} /></td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{dp.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Severity Levels */}
          <Card title="Severity Levels">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Level</th>
                  <th style={thStyle}>Criteria</th>
                </tr>
              </thead>
              <tbody>
                {sevLevels.map((sl, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><SeverityBadge severity={sl.level} /></td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{sl.criteria}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical Significance */}
          {clinSig.length > 0 && (
            <Card title="Clinical Significance">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {clinSig.map((s, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>
                    {typeof s === 'string' ? s : s.description || JSON.stringify(s)}
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* References */}
          {reference.length > 0 && (
            <Card title="References">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {reference.map((r, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{r}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      <div style={{ marginTop: 24, padding: 16, background: '#f1f5f9', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
        VEP Dashboard — Real clinical.db data ({kpis.total_studies} studies, {patientDetails.length} patients) |
        Visual pathway analysis with automated severity grading and inter-eye comparison
      </div>
    </div>
  )
}
