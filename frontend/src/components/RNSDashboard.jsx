import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PATTERN_COLORS = { normal: '#16a34a', postsynaptic_nmj: '#8b5cf6', presynaptic_nmj: '#f59e0b', mixed_nmj: '#ef4444' }
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
  const labels = { normal: 'Normal', postsynaptic_nmj: 'Postsynaptic (MG)', presynaptic_nmj: 'Presynaptic (LEMS)', mixed_nmj: 'Mixed NMJ' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{labels[pattern] || pattern || 'Unknown'}</span>
  )
}

function DecrementIndicator({ value, threshold }) {
  if (value == null) return <span>--</span>
  const abnormal = value >= threshold
  return (
    <span style={{ color: abnormal ? '#ef4444' : '#16a34a', fontWeight: abnormal ? 600 : 400 }}>
      {fmt(value)}% <span style={{ fontSize: 10, color: '#94a3b8' }}>(threshold {threshold}%)</span>
    </span>
  )
}

export default function RNSDashboard() {
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
          axios.get(`${API_URL}/api/rns/overview`),
          axios.get(`${API_URL}/api/rns/breakdown`),
          axios.get(`${API_URL}/api/rns/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading RNS data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'RNS Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.diagnostic_pattern_distribution || []
  const siteRates = overview?.site_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const rnsSummary = breakdown?.rns_summary || []
  const decHist = breakdown?.decrement_histogram || []
  const facHist = breakdown?.facilitation_histogram || []
  const siteComp = breakdown?.site_comparison || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>Repetitive Nerve Stimulation (RNS)</h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Decrement/increment auto-quantification &amp; NMJ disorder screening — myasthenia gravis &amp; LEMS detection
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
              <KPI label="Mean Decrement" value={`${kpis.mean_decrement_pct}%`} sub="threshold 10%" color={kpis.mean_decrement_pct > 10 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean Facilitation" value={`${kpis.mean_facilitation_pct}%`} sub="LEMS > 100%" />
              <KPI label="Mean CMAP" value={`${kpis.mean_baseline_cmap_mv} mV`} sub="baseline" />
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
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11 }} />
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
                <YAxis type="category" dataKey="site" width={180} tick={{ fontSize: 10 }} />
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
                      <td style={tdStyle}>{p.abnormal_sites} / {p.total_sites}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* RNS Analysis Tab */}
      {tab === 'analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="RNS Site Summary" span={2}>
            <div style={{ overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Nerve / Muscle</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Mean Decrement %</th>
                    <th style={thStyle}>Mean Facilitation %</th>
                    <th style={thStyle}>Mean CMAP (mV)</th>
                    <th style={thStyle}>Mean Exhaustion %</th>
                    <th style={thStyle}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {rnsSummary.map((s, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, fontSize: 12 }}>{s.site}</td>
                      <td style={tdStyle}>
                        <span style={{
                          padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600,
                          background: s.type === 'proximal' ? '#dbeafe' : '#fef3c7',
                          color: s.type === 'proximal' ? '#2563eb' : '#d97706'
                        }}>{s.type === 'proximal' ? 'Proximal' : 'Distal'}</span>
                      </td>
                      <td style={tdStyle}>
                        <DecrementIndicator value={s.mean_decrement_pct} threshold={s.decrement_threshold} />
                      </td>
                      <td style={tdStyle}>
                        <span style={{ color: s.mean_facilitation_pct > 100 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {fmt(s.mean_facilitation_pct)}%
                        </span>
                      </td>
                      <td style={tdStyle}>{fmt(s.mean_cmap_mv)}</td>
                      <td style={tdStyle}>{fmt(s.mean_exhaustion_pct)}%</td>
                      <td style={tdStyle}>
                        <span style={{ color: s.abnormal_pct > 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                          {s.abnormal_pct}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Decrement % Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={decHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {decHist.map((d, i) => <Cell key={i} fill={d.abnormal ? '#ef4444' : '#16a34a'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Red = abnormal (&ge; 10% decrement)
            </p>
          </Card>

          <Card title="Post-Exercise Facilitation Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={facHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {facHist.map((d, i) => <Cell key={i} fill={d.lems_range ? '#f59e0b' : '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 10, color: '#94a3b8', textAlign: 'center', margin: '4px 0 0' }}>
              Orange = LEMS range (&ge; 100% facilitation)
            </p>
          </Card>

          <Card title="Proximal vs Distal Abnormality" span={2}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={siteComp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 12 }} />
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
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Nerve / Muscle</th>
                          <th style={thStyle}>Type</th>
                          <th style={thStyle}>CMAP (mV)</th>
                          <th style={thStyle}>Decrement %</th>
                          <th style={thStyle}>Facilitation %</th>
                          <th style={thStyle}>Exhaustion %</th>
                          <th style={thStyle}>Repair</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(p.nerves || []).map((r, j) => (
                          <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                            <td style={{ ...tdStyle, fontWeight: 500, fontSize: 12 }}>
                              {r.nerve} &rarr; {r.muscle}
                            </td>
                            <td style={tdStyle}>
                              <span style={{
                                padding: '1px 6px', borderRadius: 4, fontSize: 10,
                                background: r.type === 'proximal' ? '#dbeafe' : '#fef3c7',
                                color: r.type === 'proximal' ? '#2563eb' : '#d97706'
                              }}>{r.type === 'proximal' ? 'Prox' : 'Dist'}</span>
                            </td>
                            <td style={tdStyle}>
                              <span style={{
                                color: (r.baseline_cmap_mv < r.cmap_ref_lower) ? '#ef4444' : '#16a34a',
                                fontWeight: (r.baseline_cmap_mv < r.cmap_ref_lower) ? 600 : 400
                              }}>
                                {fmt(r.baseline_cmap_mv)}
                                <span style={{ fontSize: 10, color: '#94a3b8' }}> ({r.cmap_ref_lower}-{r.cmap_ref_upper})</span>
                              </span>
                            </td>
                            <td style={tdStyle}>
                              <DecrementIndicator value={r.decrement_pct} threshold={r.decrement_threshold} />
                            </td>
                            <td style={tdStyle}>
                              <span style={{ color: r.facilitation_pct > 100 ? '#f59e0b' : '#16a34a', fontWeight: 600 }}>
                                {fmt(r.facilitation_pct)}%
                              </span>
                            </td>
                            <td style={tdStyle}>{fmt(r.post_exercise_exhaustion_pct)}%</td>
                            <td style={{ ...tdStyle, fontSize: 11 }}>{r.repair_of_decrement}</td>
                            <td style={tdStyle}><SeverityBadge severity={r.severity} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    {/* CMAP Train Waveform */}
                    {(p.nerves || []).filter(r => r.severity !== 'Normal').slice(0, 3).map((r, j) => {
                      const trainData = (r.cmap_train || []).map((amp, idx) => ({
                        stimulus: `S${idx + 1}`,
                        amplitude: amp
                      }))
                      return (
                        <div key={j} style={{ marginBottom: 12 }}>
                          <p style={{ fontSize: 12, color: '#475569', fontWeight: 600, margin: '0 0 4px' }}>
                            CMAP Train: {r.nerve} &rarr; {r.muscle} ({r.stim_frequency}, decrement {fmt(r.decrement_pct)}%)
                          </p>
                          <ResponsiveContainer width="100%" height={120}>
                            <LineChart data={trainData}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="stimulus" tick={{ fontSize: 10 }} />
                              <YAxis unit=" mV" tick={{ fontSize: 10 }} />
                              <Tooltip formatter={(v) => `${v} mV`} />
                              <Line type="monotone" dataKey="amplitude" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      )
                    })}
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
          <Card title="RNS Study Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Nerve-Muscle Pairs Tested</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
              <thead>
                <tr>
                  <th style={thStyle}>Nerve</th>
                  <th style={thStyle}>Muscle</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Innervation</th>
                </tr>
              </thead>
              <tbody>
                {(defs.protocol?.nerve_muscle_pairs || []).map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.nerve}</td>
                    <td style={tdStyle}>{p.muscle}</td>
                    <td style={tdStyle}>
                      <span style={{
                        padding: '1px 6px', borderRadius: 4, fontSize: 10,
                        background: p.type === 'proximal' ? '#dbeafe' : '#fef3c7',
                        color: p.type === 'proximal' ? '#2563eb' : '#d97706'
                      }}>{p.type === 'proximal' ? 'Proximal' : 'Distal'}</span>
                    </td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{p.innervation}</td>
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

          <Card title="RNS Parameters">
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
                  <th style={thStyle}>Normal</th>
                  <th style={thStyle}>Abnormal</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Decrement</td>
                  <td style={tdStyle}>{defs.reference_ranges?.decrement?.normal}</td>
                  <td style={{ ...tdStyle, color: '#ef4444' }}>{defs.reference_ranges?.decrement?.abnormal}</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Baseline CMAP</td>
                  <td style={tdStyle}>{defs.reference_ranges?.baseline_cmap?.normal_range}</td>
                  <td style={tdStyle}>--</td>
                </tr>
                <tr style={{ background: '#fff' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Facilitation</td>
                  <td style={tdStyle}>{defs.reference_ranges?.facilitation?.normal}</td>
                  <td style={{ ...tdStyle, color: '#f59e0b' }}>LEMS: {defs.reference_ranges?.facilitation?.lems_threshold}</td>
                </tr>
                <tr style={{ background: '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>Stimulation</td>
                  <td style={tdStyle}>Low-freq: {defs.reference_ranges?.stimulation?.low_frequency}, Train: {defs.reference_ranges?.stimulation?.train_length}</td>
                  <td style={tdStyle}>High-freq: {defs.reference_ranges?.stimulation?.high_frequency}</td>
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
