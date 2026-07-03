import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'

const QUALITY_COLORS = {
  Excellent: '#16a34a',
  Good: '#3b82f6',
  Acceptable: '#eab308',
  Poor: '#ef4444',
}
const GRADE_COLORS = {
  Good: '#16a34a',
  Fair: '#eab308',
  Poor: '#ef4444',
}
const PIE_COLORS = ['#16a34a', '#3b82f6', '#8b5cf6', '#f59e0b']
const ARTIFACT_COLORS = {
  eye_blink: '#3b82f6',
  muscle: '#ef4444',
  movement: '#f59e0b',
  electrode_pop: '#8b5cf6',
  sweat: '#06b6d4',
  ECG: '#ec4899',
}
const RECORDING_TYPE_COLORS = {
  routine: '#3b82f6',
  ambulatory: '#8b5cf6',
  video_eeg: '#16a34a',
  LTM: '#f59e0b',
}
const SEV_COLORS = {
  mild: '#16a34a',
  moderate: '#eab308',
  severe: '#ef4444',
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
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

function QualityBadge({ grade }) {
  const color = QUALITY_COLORS[grade] || GRADE_COLORS[grade] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color
    }}>{grade || 'Unknown'}</span>
  )
}

function SeverityBadge({ severity }) {
  const color = SEV_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color,
      textTransform: 'capitalize'
    }}>{severity || 'Unknown'}</span>
  )
}

function BoolBadge({ value, trueLabel = 'Yes', falseLabel = 'No' }) {
  const color = value ? '#16a34a' : '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color
    }}>{value ? trueLabel : falseLabel}</span>
  )
}

function ImpBadge({ pass }) {
  const color = pass ? '#16a34a' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color
    }}>{pass ? 'Pass' : 'Fail'}</span>
  )
}

export default function EEGTechnicianDashboard() {
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
          axios.get(`${API_URL}/eeg-technician/overview`),
          axios.get(`${API_URL}/eeg-technician/breakdown`),
          axios.get(`${API_URL}/eeg-technician/definitions`),
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG Technician data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'signal_quality', label: 'Signal Quality' },
    { id: 'artifacts', label: 'Artifact Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const recTypeDist = overview?.recording_type_distribution || []
  const artTypeDist = overview?.artifact_type_distribution || []
  const qualityHist = overview?.quality_histogram || []
  const impedanceHist = overview?.impedance_histogram || []

  const patients = breakdown?.patients || []

  const thStyle = {
    textAlign: 'left', padding: '8px 10px', fontSize: 12,
    color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600
  }
  const tdStyle = {
    padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9'
  }

  // ── Compute per-channel quality aggregation across all patients
  const channelAgg = {}
  patients.forEach(p => {
    (p.channel_detail || []).forEach(ch => {
      if (!channelAgg[ch.channel]) {
        channelAgg[ch.channel] = { good: 0, fair: 0, poor: 0, imp_total: 0, n: 0 }
      }
      const a = channelAgg[ch.channel]
      a.n++
      a.imp_total += ch.impedance_kohm || 0
      if (ch.quality_grade === 'Good') a.good++
      else if (ch.quality_grade === 'Fair') a.fair++
      else a.poor++
    })
  })
  const channelQualityData = Object.entries(channelAgg).map(([ch, a]) => ({
    channel: ch,
    good: a.good,
    fair: a.fair,
    poor: a.poor,
    avg_impedance: a.n > 0 ? Math.round((a.imp_total / a.n) * 10) / 10 : 0,
    pass_rate: a.n > 0 ? Math.round(100 * (a.good + a.fair) / a.n) : 0,
  }))

  // ── Artifact severity distribution
  const artSevCounts = { mild: 0, moderate: 0, severe: 0 }
  patients.forEach(p => {
    (p.artifact_detail || []).forEach(art => {
      const sev = art.severity
      if (artSevCounts[sev] !== undefined) artSevCounts[sev]++
    })
  })
  const artifactSevDist = Object.entries(artSevCounts).map(([sev, count]) => ({ severity: sev, count }))

  // ── Artifact by channel aggregation (top 10)
  const artByCh = {}
  patients.forEach(p => {
    (p.artifact_detail || []).forEach(art => {
      const ch = art.channel
      if (!artByCh[ch]) artByCh[ch] = 0
      artByCh[ch]++
    })
  })
  const topArtChannels = Object.entries(artByCh)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([channel, count]) => ({ channel, count }))

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        EEG Technician Quality Control
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Acquisition quality hub: signal quality, artifact analysis, impedance checks, and daily recording logs
        — ACNS minimum technical standards
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13,
            fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b',
            background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>

          {/* KPI Row 1 */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Recordings" value={fmt(kpis.total_recordings)} />
              <KPI label="Signal Quality Pass" value={`${fmt(kpis.signal_quality_pass_rate)}%`}
                color={kpis.signal_quality_pass_rate >= 90 ? '#16a34a' : kpis.signal_quality_pass_rate >= 70 ? '#eab308' : '#ef4444'}
                sub="Good + Fair channels" />
              <KPI label="Impedance Failure Rate" value={`${fmt(kpis.impedance_failure_rate)}%`}
                color={kpis.impedance_failure_rate > 20 ? '#ef4444' : kpis.impedance_failure_rate > 10 ? '#eab308' : '#16a34a'}
                sub="Studies with >10 kΩ electrode" />
              <KPI label="Avg Artifacts / Study" value={fmt(kpis.artifact_rate)}
                color={kpis.artifact_rate > 8 ? '#ef4444' : kpis.artifact_rate > 5 ? '#eab308' : '#16a34a'}
                sub="Total artifact events" />
              <KPI label="Total Artifacts" value={fmt(kpis.total_artifacts_logged)} sub="Logged annotations" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Mean Recording Duration" value={`${fmt(kpis.mean_recording_duration_min)} min`} sub="All recording types" />
              <KPI label="Photic Stim Rate" value={`${fmt(kpis.photic_stim_rate)}%`}
                color={kpis.photic_stim_rate >= 90 ? '#16a34a' : '#eab308'}
                sub="Target ≥ 90%" />
              <KPI label="HV Completion Rate" value={`${fmt(kpis.hv_completion_rate)}%`}
                color={kpis.hv_completion_rate >= 80 ? '#16a34a' : '#eab308'}
                sub="Target ≥ 80%" />
              <KPI label="Sleep Capture Rate" value={`${fmt(kpis.sleep_capture_rate)}%`}
                color={kpis.sleep_capture_rate >= 40 ? '#16a34a' : '#eab308'}
                sub="Target ≥ 40%" />
            </div>
          </Card>

          {/* Study Quality Grades */}
          <Card title="Overall Study Quality Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sevDist.filter(d => d.count > 0)} dataKey="count" nameKey="grade"
                  cx="50%" cy="50%" outerRadius={80}
                  label={({ grade, count }) => `${grade}: ${count}`}>
                  {sevDist.map((d, i) => (
                    <Cell key={i} fill={QUALITY_COLORS[d.grade] || PIE_COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Recording Type Distribution */}
          <Card title="Recording Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={recTypeDist.filter(d => d.count > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {recTypeDist.map((d, i) => (
                    <Cell key={i} fill={RECORDING_TYPE_COLORS[d.type] || PIE_COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Routine / Ambulatory / Video-EEG / LTM
            </div>
          </Card>

          {/* Artifact Type Distribution */}
          <Card title="Artifact Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={artTypeDist.filter(d => d.count > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {artTypeDist.map((d, i) => (
                    <Cell key={i} fill={ARTIFACT_COLORS[d.type] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Signal Quality Histogram */}
          <Card title="Signal Quality Score Distribution (% Good Channels per Study)" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={qualityHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }}
                  label={{ value: '% Good Channels', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              SNR ≥ 20 dB = Good channel | Target: ≥ 80% of channels Good
            </div>
          </Card>

          {/* Impedance Histogram */}
          <Card title="Electrode Impedance Distribution (all channels, kΩ)">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={impedanceHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {impedanceHist.map((d, i) => {
                    const lo = parseInt(d.range)
                    const color = lo < 5 ? '#16a34a' : lo < 10 ? '#eab308' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Green &lt; 5 kΩ (Good) | Yellow 5–10 kΩ (Fair) | Red &gt; 10 kΩ (Poor — ACNS)
            </div>
          </Card>
        </div>
      )}

      {/* ─── Signal Quality Tab ─── */}
      {tab === 'signal_quality' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>

          {/* Per-channel pass rate bar chart */}
          <Card title="Per-Channel Signal Quality Pass Rate (Good + Fair)" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={channelQualityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="pass_rate" name="Pass Rate" radius={[4, 4, 0, 0]}>
                  {channelQualityData.map((d, i) => (
                    <Cell key={i}
                      fill={d.pass_rate >= 90 ? '#16a34a' : d.pass_rate >= 70 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'center', marginTop: 4 }}>
              Green ≥ 90% | Yellow 70–89% | Red &lt; 70%
            </div>
          </Card>

          {/* Per-channel stacked bar: good/fair/poor */}
          <Card title="Per-Channel Quality Breakdown (Good / Fair / Poor)" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={channelQualityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="good" name="Good" fill="#16a34a" stackId="a" />
                <Bar dataKey="fair" name="Fair" fill="#eab308" stackId="a" />
                <Bar dataKey="poor" name="Poor" fill="#ef4444" stackId="a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Average impedance per channel */}
          <Card title="Average Electrode Impedance per Channel (kΩ)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={channelQualityData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                <YAxis unit=" kΩ" />
                <Tooltip formatter={(v) => `${v} kΩ`} />
                <Bar dataKey="avg_impedance" name="Avg Impedance" radius={[4, 4, 0, 0]}>
                  {channelQualityData.map((d, i) => (
                    <Cell key={i}
                      fill={d.avg_impedance < 5 ? '#16a34a' : d.avg_impedance < 10 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'center', marginTop: 4 }}>
              ACNS threshold: &lt; 5 kΩ (Good) | 5–10 kΩ (Fair) | &gt; 10 kΩ (Poor, must correct)
            </div>
          </Card>

          {/* Channel quality table */}
          <Card title="Channel Quality Summary Table" span={2}>
            <div style={{ maxHeight: 420, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Channel</th>
                    <th style={thStyle}>Good</th>
                    <th style={thStyle}>Fair</th>
                    <th style={thStyle}>Poor</th>
                    <th style={thStyle}>Pass Rate</th>
                    <th style={thStyle}>Avg Impedance</th>
                  </tr>
                </thead>
                <tbody>
                  {channelQualityData.map((ch, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{ch.channel}</td>
                      <td style={{ ...tdStyle, color: '#16a34a', fontWeight: 600 }}>{ch.good}</td>
                      <td style={{ ...tdStyle, color: '#eab308', fontWeight: 600 }}>{ch.fair}</td>
                      <td style={{ ...tdStyle, color: '#ef4444', fontWeight: 600 }}>{ch.poor}</td>
                      <td style={tdStyle}>
                        <span style={{
                          fontWeight: 700,
                          color: ch.pass_rate >= 90 ? '#16a34a' : ch.pass_rate >= 70 ? '#eab308' : '#ef4444'
                        }}>{ch.pass_rate}%</span>
                      </td>
                      <td style={tdStyle}>
                        <span style={{
                          fontWeight: 600,
                          color: ch.avg_impedance < 5 ? '#16a34a' : ch.avg_impedance < 10 ? '#eab308' : '#ef4444'
                        }}>{fmt(ch.avg_impedance)} kΩ</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Artifact Analysis Tab ─── */}
      {tab === 'artifacts' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>

          {/* Artifact type distribution */}
          <Card title="Artifact Type Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={artTypeDist.filter(d => d.count > 0)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {artTypeDist.map((d, i) => (
                    <Cell key={i} fill={ARTIFACT_COLORS[d.type] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Artifact severity distribution */}
          <Card title="Artifact Severity Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={artifactSevDist.filter(d => d.count > 0)} dataKey="count" nameKey="severity"
                  cx="50%" cy="50%" outerRadius={90}
                  label={({ severity, count }) => `${severity}: ${count}`}>
                  {artifactSevDist.map((d, i) => (
                    <Cell key={i} fill={SEV_COLORS[d.severity] || PIE_COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Top artifact channels */}
          <Card title="Top 10 Channels by Artifact Count" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topArtChannels} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="channel" width={50} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'center', marginTop: 4 }}>
              Frontal/temporal channels typically have highest artifact burden (muscle, blink, ECG)
            </div>
          </Card>

          {/* All artifacts table */}
          <Card title="Artifact Log — All Patients" span={2}>
            <div style={{ maxHeight: 480, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Artifact Type</th>
                    <th style={thStyle}>Channel</th>
                    <th style={thStyle}>Start (min)</th>
                    <th style={thStyle}>Duration (s)</th>
                    <th style={thStyle}>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.flatMap((p, pi) =>
                    (p.artifact_detail || []).map((art, ai) => (
                      <tr key={`${pi}-${ai}`} style={{ background: (pi + ai) % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>
                          <span style={{
                            display: 'inline-block', padding: '1px 6px', borderRadius: 4,
                            fontSize: 11, background: (ARTIFACT_COLORS[art.artifact_type] || '#94a3b8') + '22',
                            color: ARTIFACT_COLORS[art.artifact_type] || '#64748b', fontWeight: 600
                          }}>
                            {(art.artifact_type || '').replace(/_/g, ' ')}
                          </span>
                        </td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{art.channel || '--'}</td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{fmt(art.start_time_min)}</td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{fmt(art.duration_sec)}</td>
                        <td style={tdStyle}><SeverityBadge severity={art.severity} /></td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gap: 12 }}>
          {patients.map((p, i) => {
            const isExpanded = expandedPatient === p.patient_id
            return (
              <Card key={i}>
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
                  onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                >
                  <span style={{
                    fontSize: 18,
                    transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s',
                    display: 'inline-block'
                  }}>&#9654;</span>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{p.name || p.patient_id}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 10 }}>
                      {p.age ? `Age ${p.age} | ` : ''}{p.recording_type} | {p.study_date}
                    </span>
                  </div>
                  <QualityBadge grade={p.overall_quality_grade} />
                  <ImpBadge pass={p.impedance_pass} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    Artifacts: <strong>{p.artifact_count}</strong>
                  </span>
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    Good ch: <strong style={{ color: '#16a34a' }}>{p.good_channels}</strong>
                    {' / '}{p.channel_count}
                  </span>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Demographics / acquisition summary */}
                    <div style={{
                      display: 'flex', gap: 20, flexWrap: 'wrap',
                      marginBottom: 16, padding: '10px 14px',
                      background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569'
                    }}>
                      <span><strong>Patient ID:</strong> {p.patient_id}</span>
                      {p.age && <span><strong>Age:</strong> {p.age}</span>}
                      {p.gender && <span><strong>Gender:</strong> {p.gender}</span>}
                      <span><strong>Recording Type:</strong> {p.recording_type}</span>
                      <span><strong>Duration:</strong> {fmt(p.duration_min)} min</span>
                      <span><strong>Sampling Rate:</strong> {p.sampling_rate} Hz</span>
                      <span><strong>Montage:</strong> {p.montage}</span>
                      <span><strong>Electrode System:</strong> {p.electrode_system}</span>
                    </div>

                    {/* Conditions row */}
                    <div style={{
                      display: 'flex', gap: 20, flexWrap: 'wrap',
                      marginBottom: 16, padding: '8px 14px',
                      background: '#eff6ff', borderRadius: 8, fontSize: 12, color: '#1e40af'
                    }}>
                      <span>Eyes Open: <BoolBadge value={p.eyes_open} /></span>
                      <span>HV Done: <BoolBadge value={p.hyperventilation_done} trueLabel="Yes" falseLabel="No" /></span>
                      <span>Photic Stim: <BoolBadge value={p.photic_done} trueLabel="Done" falseLabel="Not done" /></span>
                      <span>Sleep Recorded: <BoolBadge value={p.sleep_recorded} /></span>
                      <span><strong>Patient State:</strong> <span style={{ textTransform: 'capitalize' }}>{p.patient_state}</span></span>
                      <span><strong>Cooperation:</strong> <span style={{ textTransform: 'capitalize' }}>{p.cooperation}</span></span>
                    </div>

                    {/* Technician notes */}
                    {p.technician_notes && (
                      <div style={{
                        marginBottom: 16, padding: '8px 14px',
                        background: '#fefce8', borderRadius: 8, fontSize: 12,
                        color: '#713f12', borderLeft: '3px solid #eab308'
                      }}>
                        <strong>Technician Notes:</strong> {p.technician_notes}
                      </div>
                    )}

                    {/* Channel quality table */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                      Channel Quality — {p.channel_count} channels (10-20 system)
                    </h4>
                    <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
                      <span style={{ fontSize: 12, color: '#16a34a' }}>
                        Good: <strong>{p.good_channels}</strong>
                      </span>
                      <span style={{ fontSize: 12, color: '#eab308' }}>
                        Fair: <strong>{p.fair_channels}</strong>
                      </span>
                      <span style={{ fontSize: 12, color: '#ef4444' }}>
                        Poor: <strong>{p.poor_channels}</strong>
                      </span>
                      <span style={{ fontSize: 12, color: '#64748b' }}>
                        Avg Impedance: <strong>{fmt(p.avg_impedance_kohm)} kΩ</strong>
                      </span>
                      <span style={{ fontSize: 12, color: '#64748b' }}>
                        Avg SNR: <strong>{fmt(p.avg_snr_db)} dB</strong>
                      </span>
                    </div>
                    {(p.channel_detail || []).length > 0 && (
                      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Channel</th>
                            <th style={thStyle}>Impedance (kΩ)</th>
                            <th style={thStyle}>Impedance Grade</th>
                            <th style={thStyle}>SNR (dB)</th>
                            <th style={thStyle}>Quality Grade</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(p.channel_detail || []).map((ch, j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 600 }}>{ch.channel}</td>
                              <td style={tdStyle}>
                                <span style={{
                                  fontWeight: 600,
                                  color: ch.impedance_kohm < 5 ? '#16a34a' : ch.impedance_kohm < 10 ? '#eab308' : '#ef4444'
                                }}>{fmt(ch.impedance_kohm)}</span>
                              </td>
                              <td style={tdStyle}><QualityBadge grade={ch.impedance_grade} /></td>
                              <td style={tdStyle}>
                                <span style={{
                                  fontWeight: 600,
                                  color: ch.snr_db >= 20 ? '#16a34a' : ch.snr_db >= 10 ? '#eab308' : '#ef4444'
                                }}>{fmt(ch.snr_db)}</span>
                              </td>
                              <td style={tdStyle}><QualityBadge grade={ch.quality_grade} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}

                    {/* Artifact annotations */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>
                      Artifact Annotations ({p.artifact_count} events
                      {p.dominant_artifact ? ` — dominant: ${p.dominant_artifact.replace(/_/g, ' ')}` : ''})
                    </h4>
                    {(p.artifact_detail || []).length > 0 ? (
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Artifact Type</th>
                            <th style={thStyle}>Channel</th>
                            <th style={thStyle}>Start (min)</th>
                            <th style={thStyle}>Duration (s)</th>
                            <th style={thStyle}>Severity</th>
                          </tr>
                        </thead>
                        <tbody>
                          {p.artifact_detail.map((art, j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle }}>
                                <span style={{
                                  display: 'inline-block', padding: '1px 6px', borderRadius: 4,
                                  fontSize: 11, fontWeight: 600,
                                  background: (ARTIFACT_COLORS[art.artifact_type] || '#94a3b8') + '22',
                                  color: ARTIFACT_COLORS[art.artifact_type] || '#64748b'
                                }}>
                                  {(art.artifact_type || '').replace(/_/g, ' ')}
                                </span>
                              </td>
                              <td style={tdStyle}>{art.channel || '--'}</td>
                              <td style={tdStyle}>{fmt(art.start_time_min)}</td>
                              <td style={tdStyle}>{fmt(art.duration_sec)}</td>
                              <td style={tdStyle}><SeverityBadge severity={art.severity} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8' }}>No artifact annotations logged.</p>
                    )}
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
          {defs?.title && (
            <div style={{ padding: '12px 16px', background: '#eff6ff', borderRadius: 8, borderLeft: '4px solid #2563eb' }}>
              <h3 style={{ margin: 0, fontSize: 15, color: '#1e40af' }}>{defs.title}</h3>
            </div>
          )}
          {(defs?.sections || []).map((section, si) => (
            <Card key={si} title={section.heading}>
              {Array.isArray(section.items) && section.items.length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ ...thStyle, width: '28%' }}>Term</th>
                      <th style={thStyle}>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.items.map((item, ii) => (
                      <tr key={ii} style={{ background: ii % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{item.term}</td>
                        <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.6 }}>{item.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : section.description ? (
                <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{section.description}</p>
              ) : null}
            </Card>
          ))}
          {(!defs?.sections || defs.sections.length === 0) && (
            <Card title="Definitions">
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
            </Card>
          )}
        </div>
      )}

      <div style={{
        marginTop: 24, padding: 16, background: '#f1f5f9',
        borderRadius: 8, fontSize: 12, color: '#64748b'
      }}>
        EEG Technician QC Dashboard — Real clinical.db data ({kpis.total_recordings} recordings,
        {' '}{kpis.total_artifacts_logged} artifact annotations) |
        ACNS 2016 Minimum Technical Standards | 10-20 International Electrode System
      </div>
    </div>
  )
}
