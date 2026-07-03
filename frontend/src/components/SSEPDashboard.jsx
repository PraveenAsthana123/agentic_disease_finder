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
  normal: '#16a34a', peripheral_lesion: '#3b82f6', cervical_cord_lesion: '#8b5cf6',
  cortical_subcortical: '#f59e0b', diffuse_dysfunction: '#ef4444'
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

export default function SSEPDashboard() {
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
          axios.get(`${API_URL}/ssep/overview`),
          axios.get(`${API_URL}/ssep/breakdown`),
          axios.get(`${API_URL}/ssep/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading SSEP data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'limbs', label: 'Limb Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const limbRates = overview?.limb_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const upperSummary = breakdown?.upper_summary || {}
  const lowerSummary = breakdown?.lower_summary || {}
  const n20Hist = breakdown?.n20_latency_histogram || []
  const p37Hist = breakdown?.p37_latency_histogram || []
  const limbComp = breakdown?.limb_comparison || []
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
        Somatosensory Evoked Potentials (SSEP)
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Dorsal column-medial lemniscal pathway assessment: N9/N13/N20 (upper) and N22/P37 (lower) with severity grading
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
              <KPI label="Mean N20" value={`${kpis.mean_n20_latency_ms} ms`} sub="Upper cortical" />
              <KPI label="Mean P37" value={`${kpis.mean_p37_latency_ms} ms`} sub="Lower cortical" />
              <KPI label="Limbs/Study" value={kpis.limbs_per_study} sub="Upper + Lower" />
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

          <Card title="Limb Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={limbRates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="limb" tick={{ fontSize: 12 }} />
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
                    <th style={thStyle}>Abnormal Limbs</th>
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
                      <td style={tdStyle}>{p.abnormal_limbs} / {p.total_limbs}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Limb Analysis Tab ─── */}
      {tab === 'limbs' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Upper Limb Summary (Median Nerve)">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
              <KPI label="Mean N9" value={`${fmt(upperSummary.mean_n9_ms)} ms`} sub="Brachial plexus" />
              <KPI label="Mean N13" value={`${fmt(upperSummary.mean_n13_ms)} ms`} sub="Cervical cord" />
              <KPI label="Mean N20" value={`${fmt(upperSummary.mean_n20_ms)} ms`} sub="Somatosensory cortex" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <KPI label="N20 Amp" value={`${fmt(upperSummary.mean_n20_amp_uv)} uV`} />
              <KPI label="N9-N13 IPL" value={`${fmt(upperSummary.mean_n9_n13_ipl_ms)} ms`} sub="Peripheral-cord" />
              <KPI label="N13-N20 CCT" value={`${fmt(upperSummary.mean_n13_n20_ipl_ms)} ms`} sub="Central conduction" />
            </div>
            <div style={{ marginTop: 12, fontSize: 13 }}>
              <span style={{ fontWeight: 600, color: upperSummary.abnormal_pct > 20 ? '#ef4444' : '#16a34a' }}>
                {upperSummary.abnormal_pct}% abnormal
              </span>
              <span style={{ color: '#94a3b8', marginLeft: 8 }}>({upperSummary.count} studies)</span>
            </div>
          </Card>

          <Card title="Lower Limb Summary (Posterior Tibial Nerve)">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
              <KPI label="Mean N22" value={`${fmt(lowerSummary.mean_n22_ms)} ms`} sub="Lumbar cord" />
              <KPI label="Mean P37" value={`${fmt(lowerSummary.mean_p37_ms)} ms`} sub="Somatosensory cortex" />
              <KPI label="P37 Amp" value={`${fmt(lowerSummary.mean_p37_amp_uv)} uV`} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              <KPI label="N22-P37 CCT" value={`${fmt(lowerSummary.mean_n22_p37_ipl_ms)} ms`} sub="Central conduction" />
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontWeight: 600, color: lowerSummary.abnormal_pct > 20 ? '#ef4444' : '#16a34a', fontSize: 18 }}>
                  {lowerSummary.abnormal_pct}% abnormal
                </span>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>({lowerSummary.count} studies)</div>
              </div>
            </div>
          </Card>

          <Card title="N20 Latency Distribution (Upper Limb)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={n20Hist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="P37 Latency Distribution (Lower Limb)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={p37Hist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} label={{ value: 'ms', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Upper vs Lower Limb Comparison" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Limb</th>
                  <th style={thStyle}>Total Studies</th>
                  <th style={thStyle}>Abnormal</th>
                  <th style={thStyle}>Abnormal %</th>
                  <th style={thStyle}>Mean Cortical Latency</th>
                  <th style={thStyle}>Mean CCT</th>
                </tr>
              </thead>
              <tbody>
                {limbComp.map((l, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{l.limb}</td>
                    <td style={tdStyle}>{l.total}</td>
                    <td style={tdStyle}>{l.abnormal}</td>
                    <td style={tdStyle}>
                      <span style={{ color: l.abnormal_pct > 20 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                        {l.abnormal_pct}%
                      </span>
                    </td>
                    <td style={tdStyle}>{fmt(l.mean_cortical_latency_ms)} ms</td>
                    <td style={tdStyle}>{fmt(l.mean_cct_ms)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Upper severity distribution */}
          {upperSummary.severity_dist && (
            <Card title="Upper Limb Severity">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={Object.entries(upperSummary.severity_dist).map(([s, c]) => ({ severity: s, count: c }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {Object.entries(upperSummary.severity_dist).map(([s], i) => (
                      <Cell key={i} fill={SEV_COLORS[s] || '#6366f1'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Lower severity distribution */}
          {lowerSummary.severity_dist && (
            <Card title="Lower Limb Severity">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={Object.entries(lowerSummary.severity_dist).map(([s, c]) => ({ severity: s, count: c }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {Object.entries(lowerSummary.severity_dist).map(([s], i) => (
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
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Upper Limb */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Upper Limb (Median Nerve)</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Peak</th>
                          <th style={thStyle}>Latency (ms)</th>
                          <th style={thStyle}>Reference</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N9 (Erb's point)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n9_latency_ms} refVal={p.upper.n9_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n9_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n9_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N13 (Cervical)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n13_latency_ms} refVal={p.upper.n13_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n13_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n13_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N20 (Cortical)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n20_latency_ms} refVal={p.upper.n20_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n20_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n20_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N20 Amplitude</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n20_amplitude_uv} refVal={p.upper.n20_amp_ref} higher_is_bad={false} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n20_amp_ref)} uV</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n20_amp_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                      </tbody>
                    </table>

                    <h4 style={{ fontSize: 12, color: '#64748b', margin: '0 0 8px' }}>Interpeak Latencies</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>IPL</th>
                          <th style={thStyle}>Value (ms)</th>
                          <th style={thStyle}>Reference</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N9-N13 (Peripheral-Cord)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n9_n13_ipl_ms} refVal={p.upper.n9_n13_ipl_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n9_n13_ipl_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n9_n13_ipl_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N13-N20 (CCT)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.upper.n13_n20_ipl_ms} refVal={p.upper.n13_n20_ipl_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.upper.n13_n20_ipl_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.upper.n13_n20_ipl_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                      </tbody>
                    </table>

                    {/* Lower Limb */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Lower Limb (Posterior Tibial Nerve)</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Peak</th>
                          <th style={thStyle}>Latency (ms)</th>
                          <th style={thStyle}>Reference</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N22 (Lumbar cord)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.lower.n22_latency_ms} refVal={p.lower.n22_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.lower.n22_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.lower.n22_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P37 (Cortical)</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.lower.p37_latency_ms} refVal={p.lower.p37_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.lower.p37_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.lower.p37_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>P37 Amplitude</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.lower.p37_amplitude_uv} refVal={p.lower.p37_amp_ref} higher_is_bad={false} />
                          </td>
                          <td style={tdStyle}>{fmt(p.lower.p37_amp_ref)} uV</td>
                          <td style={tdStyle}><SeverityBadge severity={p.lower.p37_amp_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>N22-P37 CCT</td>
                          <td style={tdStyle}>
                            <RefIndicator value={p.lower.n22_p37_ipl_ms} refVal={p.lower.n22_p37_ipl_ref} higher_is_bad={true} />
                          </td>
                          <td style={tdStyle}>{fmt(p.lower.n22_p37_ipl_ref)} ms</td>
                          <td style={tdStyle}><SeverityBadge severity={p.lower.n22_p37_ipl_abnormal ? 'Abnormal' : 'Normal'} /></td>
                        </tr>
                      </tbody>
                    </table>

                    <div style={{ fontSize: 12, color: '#64748b', padding: '4px 10px', background: '#f1f5f9', borderRadius: 6 }}>
                      Overall: <SeverityBadge severity={p.upper.severity} /> (Upper) &nbsp;|&nbsp; <SeverityBadge severity={p.lower.severity} /> (Lower)
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
            {protocol.upper_limb && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 6px' }}>Upper Limb</h4>
                <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 4px' }}>Stimulus: {protocol.upper_limb.stimulus_site}</p>
                <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                  {protocol.upper_limb.recording_sites?.map((s, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
            {protocol.lower_limb && (
              <div style={{ marginTop: 12 }}>
                <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 6px' }}>Lower Limb</h4>
                <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 4px' }}>Stimulus: {protocol.lower_limb.stimulus_site}</p>
                <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                  {protocol.lower_limb.recording_sites?.map((s, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 2 }}>{s}</li>
                  ))}
                </ul>
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
          <Card title="SSEP Parameters">
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
          {refRanges.upper_limb && (
            <Card title="Reference Ranges">
              <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 8px' }}>Upper Limb (Median Nerve)</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Upper Limit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(refRanges.upper_limb).map(([k, v], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{k.replace(/_/g, ' ')}</td>
                      <td style={tdStyle}>{typeof v === 'number' ? `${v} ms` : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <h4 style={{ fontSize: 13, color: '#1e293b', margin: '0 0 8px' }}>Lower Limb (Posterior Tibial Nerve)</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Parameter</th>
                    <th style={thStyle}>Upper Limit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(refRanges.lower_limb).map(([k, v], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{k.replace(/_/g, ' ')}</td>
                      <td style={tdStyle}>{typeof v === 'number' ? `${v} ms` : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Diagnostic Patterns */}
          <Card title="Diagnostic Patterns">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Pattern</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Clinical Significance</th>
                </tr>
              </thead>
              <tbody>
                {diagPatterns.map((dp, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{dp.pattern?.replace(/_/g, ' ') || dp.name}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{dp.description}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{dp.clinical_significance || dp.significance || '--'}</td>
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
        SSEP Dashboard — Real clinical.db data ({kpis.total_studies} studies, {patientDetails.length} patients) |
        Dorsal column-medial lemniscal pathway analysis with automated severity grading
      </div>
    </div>
  )
}
