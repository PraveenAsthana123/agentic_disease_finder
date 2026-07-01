import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PATTERN_COLORS = {
  normal: '#16a34a', trigeminal_neuropathy: '#8b5cf6', facial_neuropathy: '#f59e0b',
  pontine_lesion: '#ef4444', medullary_lesion: '#ec4899'
}
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

export default function BlinkReflexDashboard() {
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
          axios.get(`${API_URL}/blink-reflex/overview`),
          axios.get(`${API_URL}/blink-reflex/breakdown`),
          axios.get(`${API_URL}/blink-reflex/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Blink Reflex data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'responses', label: 'Response Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const sideRates = overview?.side_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const sideSummary = breakdown?.side_summary || []
  const r1Hist = breakdown?.r1_latency_histogram || []
  const ipsiVsContra = breakdown?.ipsi_vs_contra_r2 || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>Blink Reflex</h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Trigeminal-facial reflex arc: R1/R2 latency analysis with brainstem dysfunction screening
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
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Studies" value={kpis.total_studies} />
              <KPI label="Abnormal" value={kpis.abnormal_count} color="#ef4444" />
              <KPI label="Abnormal Rate" value={`${kpis.abnormal_rate_pct}%`} color={kpis.abnormal_rate_pct > 40 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean R1" value={`${kpis.mean_r1_latency_ms} ms`} sub="Ipsilateral" />
              <KPI label="Mean R2" value={`${kpis.mean_r2_latency_ms} ms`} sub="Ipsilateral" />
              <KPI label="Sides/Study" value={kpis.sides_per_study} sub="L + R" />
            </div>
          </Card>

          {/* Severity Distribution */}
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

          {/* Diagnostic Pattern Distribution */}
          <Card title="Diagnostic Pattern">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={patternDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.map((d, i) => <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Side Abnormality Rate */}
          <Card title="Per-Side Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sideRates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="side" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="rate_pct" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Patient Summary Table */}
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
                    <th style={thStyle}>Abnormal Params</th>
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
                      <td style={tdStyle}>{p.abnormal_params} / {p.total_params}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Response Analysis Tab ─── */}
      {tab === 'responses' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Side-by-Side Response Summary */}
          <Card title="Response Summary by Stimulus Side" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Stim Side</th>
                  <th style={thStyle}>Mean R1 (ms)</th>
                  <th style={thStyle}>Mean R2 Ipsi (ms)</th>
                  <th style={thStyle}>Mean R2 Contra (ms)</th>
                  <th style={thStyle}>Mean R1 Amp (mV)</th>
                  <th style={thStyle}>Mean R2 Amp (mV)</th>
                  <th style={thStyle}>Abnormal %</th>
                </tr>
              </thead>
              <tbody>
                {sideSummary.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.side}</td>
                    <td style={tdStyle}><RefIndicator value={s.mean_r1_latency_ms} refVal={13} higher_is_bad={true} /></td>
                    <td style={tdStyle}><RefIndicator value={s.mean_r2_ipsi_latency_ms} refVal={40} higher_is_bad={true} /></td>
                    <td style={tdStyle}><RefIndicator value={s.mean_r2_contra_latency_ms} refVal={42} higher_is_bad={true} /></td>
                    <td style={tdStyle}><RefIndicator value={s.mean_r1_amplitude_mv} refVal={0.1} higher_is_bad={false} /></td>
                    <td style={tdStyle}><RefIndicator value={s.mean_r2_amplitude_mv} refVal={0.05} higher_is_bad={false} /></td>
                    <td style={tdStyle}>
                      <span style={{ color: s.abnormal_pct > 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                        {s.abnormal_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* R1 Latency Distribution */}
          <Card title="R1 Latency Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={r1Hist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Ipsi vs Contra R2 Comparison */}
          <Card title="R2 Ipsilateral vs Contralateral">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ipsiVsContra.map(d => ({
                component: d.component,
                normal: d.total - d.abnormal_count,
                abnormal: d.abnormal_count,
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="normal" stackId="a" fill="#16a34a" name="Normal" />
                <Bar dataKey="abnormal" stackId="a" fill="#ef4444" name="Abnormal" radius={[4, 4, 0, 0]} />
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
                    {[p.left, p.right].filter(Boolean).map((side, si) => (
                      <div key={si} style={{ marginBottom: 16 }}>
                        <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                          {side.side} Stimulation
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
                              <td style={{ ...tdStyle, fontWeight: 500 }}>R1 Latency (ms)</td>
                              <td style={tdStyle}><RefIndicator value={side.r1_latency_ms} refVal={13} higher_is_bad={true} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#94a3b8' }}>&le; 13 ms</td>
                              <td style={tdStyle}><SeverityBadge severity={side.r1_latency_ms > 13 ? 'Abnormal' : 'Normal'} /></td>
                            </tr>
                            <tr style={{ background: '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>R2 Ipsilateral (ms)</td>
                              <td style={tdStyle}><RefIndicator value={side.r2_ipsi_latency_ms} refVal={40} higher_is_bad={true} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#94a3b8' }}>&le; 40 ms</td>
                              <td style={tdStyle}><SeverityBadge severity={side.r2_ipsi_latency_ms > 40 ? 'Abnormal' : 'Normal'} /></td>
                            </tr>
                            <tr style={{ background: '#fff' }}>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>R2 Contralateral (ms)</td>
                              <td style={tdStyle}><RefIndicator value={side.r2_contra_latency_ms} refVal={42} higher_is_bad={true} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#94a3b8' }}>&le; 42 ms</td>
                              <td style={tdStyle}><SeverityBadge severity={side.r2_contra_latency_ms > 42 ? 'Abnormal' : 'Normal'} /></td>
                            </tr>
                            <tr style={{ background: '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>R1 Amplitude (mV)</td>
                              <td style={tdStyle}><RefIndicator value={side.r1_amplitude_mv} refVal={0.1} higher_is_bad={false} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#94a3b8' }}>&ge; 0.1 mV</td>
                              <td style={tdStyle}><SeverityBadge severity={side.r1_amplitude_mv < 0.1 ? 'Abnormal' : 'Normal'} /></td>
                            </tr>
                            <tr style={{ background: '#fff' }}>
                              <td style={{ ...tdStyle, fontWeight: 500 }}>R2 Amplitude (mV)</td>
                              <td style={tdStyle}><RefIndicator value={side.r2_amplitude_mv} refVal={0.05} higher_is_bad={false} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#94a3b8' }}>&ge; 0.05 mV</td>
                              <td style={tdStyle}><SeverityBadge severity={side.r2_amplitude_mv < 0.05 ? 'Abnormal' : 'Normal'} /></td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    ))}
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
          <Card title="Blink Reflex Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Stimulus</h4>
                <p style={{ fontSize: 13, color: '#475569', margin: 0 }}>{defs.protocol?.stimulus}</p>
              </div>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Recording</h4>
                <p style={{ fontSize: 13, color: '#475569', margin: 0 }}>{defs.protocol?.recording}</p>
              </div>
            </div>
            <h4 style={{ fontSize: 13, margin: '12px 0 6px', color: '#334155' }}>Indications</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
              {(defs.protocol?.indications || []).map((ind, i) => <li key={i}>{ind}</li>)}
            </ul>
            <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 10 }}>Standard: {defs.protocol?.standard}</p>
          </Card>

          <Card title="Blink Reflex Parameters">
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
                  <th style={thStyle}>Normal Limit</th>
                  <th style={thStyle}>Direction</th>
                </tr>
              </thead>
              <tbody>
                {(defs.reference_ranges || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.parameter}</td>
                    <td style={tdStyle}>{r.normal_limit}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{r.direction}</td>
                  </tr>
                ))}
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
                {(defs.diagnostic_patterns || []).map((t, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><PatternBadge pattern={t.pattern} /></td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{t.description}</td>
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
