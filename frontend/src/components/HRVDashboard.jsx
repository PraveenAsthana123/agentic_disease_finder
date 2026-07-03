import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const PIE_COLORS = ['#16a34a', '#3b82f6', '#eab308', '#ef4444']
const PATTERN_COLORS = {
  normal: '#16a34a',
  sympathetic_dominance: '#f59e0b',
  parasympathetic_dominance: '#3b82f6',
  autonomic_neuropathy: '#dc2626',
  cardiac_risk: '#ef4444',
  chronotropic_incompetence: '#8b5cf6'
}
const FLAG_COLORS = { normal: '#16a34a', low: '#3b82f6', high: '#ef4444', info: '#94a3b8' }

const PARAM_LABELS = {
  mean_rr_ms: 'Mean RR Interval',
  sdnn_ms: 'SDNN',
  rmssd_ms: 'RMSSD',
  pnn50_pct: 'pNN50',
  hr_mean_bpm: 'Mean Heart Rate',
  hr_min_bpm: 'HR Min',
  hr_max_bpm: 'HR Max',
  vlf_power_ms2: 'VLF Power',
  lf_power_ms2: 'LF Power',
  hf_power_ms2: 'HF Power',
  lf_hf_ratio: 'LF/HF Ratio',
  total_power_ms2: 'Total Power'
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

function PatternBadge({ pattern, label }) {
  const color = PATTERN_COLORS[pattern] || '#94a3b8'
  const displayLabel = label || (pattern ? pattern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown')
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{displayLabel}</span>
  )
}

function FlagBadge({ flag }) {
  const color = FLAG_COLORS[flag] || '#94a3b8'
  const label = flag ? flag.charAt(0).toUpperCase() + flag.slice(1) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function HRVDashboard() {
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
          axios.get(`${API_URL}/hrv/overview`),
          axios.get(`${API_URL}/hrv/breakdown`),
          axios.get(`${API_URL}/hrv/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading HRV data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'time_domain', label: 'Time Domain Analysis' },
    { id: 'freq_domain', label: 'Frequency Domain Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}

  // Severity distribution: convert object to array for PieChart
  const sevDistRaw = overview?.severity_distribution || {}
  const sevDist = Object.entries(sevDistRaw).map(([severity, count]) => ({ severity, count }))

  // Pattern distribution: convert object to array for BarChart
  const patternDistRaw = overview?.pattern_distribution || {}
  const patternDist = Object.entries(patternDistRaw).map(([pattern, count]) => ({
    pattern,
    label: pattern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    count
  }))

  const patientSummary = overview?.patient_summary || []

  const timeDomainSummary = breakdown?.time_domain_summary || []
  const freqDomainSummary = breakdown?.freq_domain_summary || []
  const sdnnHistogram = breakdown?.sdnn_histogram || []
  const rmssdHistogram = breakdown?.rmssd_histogram || []
  const lfHfHistogram = breakdown?.lf_hf_ratio_histogram || []
  const autonomicScoreHistogram = breakdown?.autonomic_score_histogram || []
  const patientDetailCards = breakdown?.patient_detail_cards || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  // Helper to format histogram bin labels
  const histLabel = (d) => `${fmt(d.bin_start)}–${fmt(d.bin_end)}`

  // Prepare histogram data with labels
  const sdnnHistData = sdnnHistogram.map(d => ({ label: histLabel(d), count: d.count }))
  const rmssdHistData = rmssdHistogram.map(d => ({ label: histLabel(d), count: d.count }))
  const lfHfHistData = lfHfHistogram.map(d => ({ label: histLabel(d), count: d.count }))
  const autoScoreHistData = autonomicScoreHistogram.map(d => ({ label: histLabel(d), count: d.count }))

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        HRV / RR Variation Analysis
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Heart Rate Variability analysis: time domain + frequency domain parameters with autonomic scoring and diagnostic pattern classification
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
          {/* 6 KPI cards in 3x2 grid */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Studies" value={fmt(kpis.total_studies)} />
              <KPI label="Abnormal Count" value={fmt(kpis.abnormal_count)}
                   color={kpis.abnormal_count > 0 ? '#ef4444' : '#16a34a'} />
              <KPI label="Abnormal Rate" value={`${fmt(kpis.abnormal_rate_pct)}%`}
                   color={kpis.abnormal_rate_pct > 50 ? '#ef4444' : '#eab308'} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Mean SDNN" value={`${fmt(kpis.mean_sdnn_ms)} ms`} sub="Normal > 50 ms" />
              <KPI label="Mean RMSSD" value={`${fmt(kpis.mean_rmssd_ms)} ms`} sub="Normal > 20 ms" />
              <KPI label="Mean LF/HF Ratio" value={fmt(kpis.mean_lf_hf_ratio)} sub="Sympathovagal balance" />
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
          <Card title="Diagnostic Pattern Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={patternDist.filter(d => d.count > 0)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={200} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Patient Summary Table */}
          <Card title="Patient Summary" span={3}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Age</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Severity</th>
                    <th style={thStyle}>Pattern</th>
                    <th style={thStyle}>Autonomic Score</th>
                    <th style={thStyle}>SDNN (ms)</th>
                    <th style={thStyle}>RMSSD (ms)</th>
                    <th style={thStyle}>LF/HF</th>
                    <th style={thStyle}>Recording Type</th>
                  </tr>
                </thead>
                <tbody>
                  {patientSummary.map((p, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || p.patient_id}</td>
                      <td style={tdStyle}>{p.age}</td>
                      <td style={tdStyle}>{p.disease}</td>
                      <td style={tdStyle}><SeverityBadge severity={p.severity} /></td>
                      <td style={tdStyle}><PatternBadge pattern={p.diagnostic_pattern} label={p.pattern_label} /></td>
                      <td style={tdStyle}>
                        <span style={{ fontWeight: 600, color: p.is_abnormal ? '#ef4444' : '#16a34a' }}>
                          {fmt(p.autonomic_score)}
                        </span>
                      </td>
                      <td style={tdStyle}>{fmt(p.sdnn_ms)}</td>
                      <td style={tdStyle}>{fmt(p.rmssd_ms)}</td>
                      <td style={tdStyle}>{fmt(p.lf_hf_ratio)}</td>
                      <td style={tdStyle}>{p.recording_type || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Time Domain Analysis Tab ─── */}
      {tab === 'time_domain' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Parameter summary table */}
          <Card title="Time Domain Parameter Summary" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Mean</th>
                    <th style={thStyle}>Min</th>
                    <th style={thStyle}>Max</th>
                    <th style={thStyle}>Unit</th>
                    <th style={thStyle}>Ref Range</th>
                    <th style={thStyle}>Abnormal N</th>
                    <th style={thStyle}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {timeDomainSummary.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>
                        {PARAM_LABELS[row.param_key] || row.parameter}
                      </td>
                      <td style={tdStyle}>{fmt(row.mean)}</td>
                      <td style={tdStyle}>{fmt(row.min)}</td>
                      <td style={tdStyle}>{fmt(row.max)}</td>
                      <td style={tdStyle}>{row.unit || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                        {row.ref_low != null && row.ref_high != null
                          ? `${row.ref_low}–${row.ref_high}`
                          : row.ref_low != null ? `≥ ${row.ref_low}`
                          : row.ref_high != null ? `≤ ${row.ref_high}`
                          : '--'}
                      </td>
                      <td style={tdStyle}>{fmt(row.abnormal_n)}</td>
                      <td style={tdStyle}>
                        <span style={{ color: (row.abnormal_pct || 0) > 50 ? '#ef4444' : (row.abnormal_pct || 0) > 20 ? '#eab308' : '#16a34a', fontWeight: 600 }}>
                          {fmt(row.abnormal_pct)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* SDNN Distribution */}
          <Card title="SDNN Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sdnnHistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Normal threshold: SDNN &gt; 50 ms (Task Force ESC/NASPE 1996)
            </div>
          </Card>

          {/* RMSSD Distribution */}
          <Card title="RMSSD Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={rmssdHistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#16a34a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Normal threshold: RMSSD &gt; 20 ms (vagal tone marker)
            </div>
          </Card>
        </div>
      )}

      {/* ─── Frequency Domain Analysis Tab ─── */}
      {tab === 'freq_domain' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Parameter summary table */}
          <Card title="Frequency Domain Parameter Summary" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Mean</th>
                    <th style={thStyle}>Min</th>
                    <th style={thStyle}>Max</th>
                    <th style={thStyle}>Unit</th>
                    <th style={thStyle}>Ref Range</th>
                    <th style={thStyle}>Abnormal N</th>
                    <th style={thStyle}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {freqDomainSummary.map((row, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>
                        {PARAM_LABELS[row.param_key] || row.parameter}
                      </td>
                      <td style={tdStyle}>{fmt(row.mean)}</td>
                      <td style={tdStyle}>{fmt(row.min)}</td>
                      <td style={tdStyle}>{fmt(row.max)}</td>
                      <td style={tdStyle}>{row.unit || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                        {row.ref_low != null && row.ref_high != null
                          ? `${row.ref_low}–${row.ref_high}`
                          : row.ref_low != null ? `≥ ${row.ref_low}`
                          : row.ref_high != null ? `≤ ${row.ref_high}`
                          : '--'}
                      </td>
                      <td style={tdStyle}>{fmt(row.abnormal_n)}</td>
                      <td style={tdStyle}>
                        <span style={{ color: (row.abnormal_pct || 0) > 50 ? '#ef4444' : (row.abnormal_pct || 0) > 20 ? '#eab308' : '#16a34a', fontWeight: 600 }}>
                          {fmt(row.abnormal_pct)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* LF/HF Ratio Distribution */}
          <Card title="LF/HF Ratio Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={lfHfHistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} label={{ value: 'Ratio', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Normal LF/HF ratio: 0.5–2.0 (sympathovagal balance)
            </div>
          </Card>

          {/* Autonomic Score Distribution */}
          <Card title="Autonomic Score Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={autoScoreHistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} label={{ value: 'Score', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Autonomic Score: composite HRV severity index (0–100)
            </div>
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gap: 12 }}>
          {patientDetailCards.map((p, i) => {
            const isExpanded = expandedPatient === p.patient_id
            const timeDomain = p.time_domain || {}
            const freqDomain = p.frequency_domain || {}
            return (
              <Card key={i}>
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
                  onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                >
                  <span style={{ fontSize: 18, transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', display: 'inline-block' }}>
                    &#9654;
                  </span>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{p.name || p.patient_id}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 10 }}>
                      Age {p.age} | {p.disease}
                    </span>
                  </div>
                  <SeverityBadge severity={p.severity} />
                  <PatternBadge pattern={p.diagnostic_pattern} label={p.pattern_label} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    Score: <strong>{fmt(p.autonomic_score)}</strong>
                  </span>
                  {p.is_abnormal && (
                    <span style={{
                      fontSize: 11, fontWeight: 700, color: '#ef4444',
                      background: '#fee2e2', padding: '2px 8px', borderRadius: 6
                    }}>Abnormal</span>
                  )}
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Demographics row */}
                    <div style={{ display: 'flex', gap: 20, marginBottom: 16, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569' }}>
                      <span><strong>Patient ID:</strong> {p.patient_id}</span>
                      <span><strong>Age:</strong> {p.age}</span>
                      <span><strong>Disease:</strong> {p.disease}</span>
                      <span><strong>Seizure Count:</strong> {fmt(p.seizure_count)}</span>
                      <span><strong>Med Count:</strong> {fmt(p.med_count)}</span>
                      <span><strong>Recording:</strong> {p.recording_type || '--'}</span>
                      <span><strong>Pattern:</strong> {p.diagnostic_pattern ? p.diagnostic_pattern.replace(/_/g, ' ') : '--'}</span>
                    </div>

                    {/* Time Domain Table */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Time Domain Parameters</h4>
                    {Object.keys(timeDomain).length > 0 ? (
                      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Parameter</th>
                            <th style={thStyle}>Value</th>
                            <th style={thStyle}>Unit</th>
                            <th style={thStyle}>Flag</th>
                            <th style={thStyle}>Ref Range</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(timeDomain).map(([key, param], j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 600 }}>{PARAM_LABELS[key] || key}</td>
                              <td style={tdStyle}>
                                <span style={{
                                  color: param.flag === 'normal' ? '#16a34a'
                                    : param.flag === 'high' ? '#ef4444'
                                    : param.flag === 'low' ? '#3b82f6'
                                    : '#64748b',
                                  fontWeight: 600
                                }}>
                                  {fmt(param.value)}
                                </span>
                              </td>
                              <td style={tdStyle}>{param.unit || '--'}</td>
                              <td style={tdStyle}><FlagBadge flag={param.flag} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                                {param.ref_low != null && param.ref_high != null
                                  ? `${param.ref_low}–${param.ref_high}`
                                  : param.ref_low != null ? `≥ ${param.ref_low}`
                                  : param.ref_high != null ? `≤ ${param.ref_high}`
                                  : '--'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>No time domain data available.</p>
                    )}

                    {/* Frequency Domain Table */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Frequency Domain Parameters</h4>
                    {Object.keys(freqDomain).length > 0 ? (
                      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Parameter</th>
                            <th style={thStyle}>Value</th>
                            <th style={thStyle}>Unit</th>
                            <th style={thStyle}>Flag</th>
                            <th style={thStyle}>Ref Range</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(freqDomain).map(([key, param], j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 600 }}>{PARAM_LABELS[key] || key}</td>
                              <td style={tdStyle}>
                                <span style={{
                                  color: param.flag === 'normal' ? '#16a34a'
                                    : param.flag === 'high' ? '#ef4444'
                                    : param.flag === 'low' ? '#3b82f6'
                                    : '#64748b',
                                  fontWeight: 600
                                }}>
                                  {fmt(param.value)}
                                </span>
                              </td>
                              <td style={tdStyle}>{param.unit || '--'}</td>
                              <td style={tdStyle}><FlagBadge flag={param.flag} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                                {param.ref_low != null && param.ref_high != null
                                  ? `${param.ref_low}–${param.ref_high}`
                                  : param.ref_low != null ? `≥ ${param.ref_low}`
                                  : param.ref_high != null ? `≤ ${param.ref_high}`
                                  : '--'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8' }}>No frequency domain data available.</p>
                    )}
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
          {/* Protocol Section */}
          {defs.protocol && (
            <Card title={`${defs.test_name || 'HRV'} — Protocol`}>
              <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: '0 0 10px' }}>
                {defs.protocol.description}
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, fontSize: 12 }}>
                {defs.protocol.recording_types && (
                  <div>
                    <strong style={{ color: '#334155' }}>Recording Types:</strong>
                    <span style={{ color: '#475569', marginLeft: 6 }}>
                      {Array.isArray(defs.protocol.recording_types)
                        ? defs.protocol.recording_types.join(', ')
                        : defs.protocol.recording_types}
                    </span>
                  </div>
                )}
                {defs.protocol.standard && (
                  <div>
                    <strong style={{ color: '#334155' }}>Standard:</strong>
                    <span style={{ color: '#475569', marginLeft: 6 }}>{defs.protocol.standard}</span>
                  </div>
                )}
                {defs.protocol.ai_models && (
                  <div>
                    <strong style={{ color: '#334155' }}>AI Models:</strong>
                    <span style={{ color: '#475569', marginLeft: 6 }}>
                      {Array.isArray(defs.protocol.ai_models)
                        ? defs.protocol.ai_models.join(', ')
                        : defs.protocol.ai_models}
                    </span>
                  </div>
                )}
                {defs.protocol.biomarkers && (
                  <div>
                    <strong style={{ color: '#334155' }}>Biomarkers:</strong>
                    <span style={{ color: '#475569', marginLeft: 6 }}>
                      {Array.isArray(defs.protocol.biomarkers)
                        ? defs.protocol.biomarkers.join(', ')
                        : defs.protocol.biomarkers}
                    </span>
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* Time Domain Parameters */}
          {defs.parameters?.time_domain && defs.parameters.time_domain.length > 0 && (
            <Card title="Time Domain Parameters">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Unit</th>
                    <th style={thStyle}>Description</th>
                    <th style={thStyle}>Ref Range</th>
                    <th style={thStyle}>Clinical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.parameters.time_domain.map((param, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{param.label || param.key}</td>
                      <td style={tdStyle}>{param.unit || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569' }}>{param.description || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                        {param.ref_low != null && param.ref_high != null
                          ? `${param.ref_low}–${param.ref_high}`
                          : param.ref_low != null ? `≥ ${param.ref_low}`
                          : param.ref_high != null ? `≤ ${param.ref_high}`
                          : '--'}
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{param.clinical || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Frequency Domain Parameters */}
          {defs.parameters?.frequency_domain && defs.parameters.frequency_domain.length > 0 && (
            <Card title="Frequency Domain Parameters">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Unit</th>
                    <th style={thStyle}>Description</th>
                    <th style={thStyle}>Ref Range</th>
                    <th style={thStyle}>Clinical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.parameters.frequency_domain.map((param, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{param.label || param.key}</td>
                      <td style={tdStyle}>{param.unit || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569' }}>{param.description || '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                        {param.ref_low != null && param.ref_high != null
                          ? `${param.ref_low}–${param.ref_high}`
                          : param.ref_low != null ? `≥ ${param.ref_low}`
                          : param.ref_high != null ? `≤ ${param.ref_high}`
                          : '--'}
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{param.clinical || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Diagnostic Patterns */}
          {defs.diagnostic_patterns && Object.keys(defs.diagnostic_patterns).length > 0 && (
            <Card title="Diagnostic Patterns">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '25%' }}>Pattern</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defs.diagnostic_patterns).map(([key, val], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>
                        <PatternBadge pattern={key} label={val.label} />
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                        {val.description || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Severity Levels */}
          {defs.severity_levels && Object.keys(defs.severity_levels).length > 0 && (
            <Card title="Severity Levels">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '15%' }}>Level</th>
                    <th style={{ ...thStyle, width: '25%' }}>Autonomic Score Range</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defs.severity_levels).map(([level, val], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>
                        <SeverityBadge severity={level} />
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>
                        {val.autonomic_score_range || '--'}
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                        {val.description || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Clinical Significance */}
          {defs.clinical_significance && (
            <Card title="Clinical Significance">
              {Object.entries(defs.clinical_significance).map(([key, val], i) => (
                <div key={i} style={{ marginBottom: 12 }}>
                  <strong style={{ fontSize: 13, color: '#334155', display: 'block', marginBottom: 4 }}>
                    {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                  </strong>
                  {typeof val === 'string' ? (
                    <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{val}</p>
                  ) : typeof val === 'object' && val !== null ? (
                    <div style={{ paddingLeft: 12 }}>
                      {Object.entries(val).map(([subKey, subVal], j) => (
                        <p key={j} style={{ fontSize: 12, color: '#475569', lineHeight: 1.5, margin: '2px 0' }}>
                          <strong>{subKey}:</strong> {subVal}
                        </p>
                      ))}
                    </div>
                  ) : null}
                </div>
              ))}
            </Card>
          )}

          {/* Fallback if no defs content */}
          {!defs.protocol && !defs.parameters && !defs.diagnostic_patterns && !defs.severity_levels && !defs.clinical_significance && (
            <Card title="Definitions">
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
            </Card>
          )}
        </div>
      )}

      {!defs && tab === 'definitions' && (
        <Card title="Definitions">
          <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
        </Card>
      )}

      <div style={{ marginTop: 24, padding: 16, background: '#f1f5f9', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
        HRV / RR Variation Dashboard — Real clinical.db data ({fmt(kpis.total_studies)} studies, {patientDetailCards.length || patientSummary.length} patients) |
        Task Force ESC/NASPE 1996 standards
      </div>
    </div>
  )
}
