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
  normal: '#16a34a', sustained_hypertension: '#ef4444', white_coat_hypertension: '#f59e0b',
  masked_hypertension: '#8b5cf6', autonomic_dysregulation: '#ec4899', arrhythmia_burden: '#dc2626'
}
const DIPPING_COLORS = {
  extreme_dipper: '#8b5cf6', normal_dipper: '#16a34a', non_dipper: '#f59e0b', reverse_dipper: '#ef4444'
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

function DippingBadge({ category }) {
  const color = DIPPING_COLORS[category] || '#94a3b8'
  const label = category ? category.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function FlagIndicator({ value, flag, unit, refLow, refHigh }) {
  const isNormal = flag === 'normal'
  return (
    <span style={{ color: isNormal ? '#16a34a' : '#ef4444', fontWeight: isNormal ? 400 : 600 }}>
      {fmt(value)} {unit && <span style={{ fontSize: 10, color: '#94a3b8' }}>{unit}</span>}
      {(refLow != null || refHigh != null) && (
        <span style={{ fontSize: 10, color: '#94a3b8' }}> ({fmt(refLow)}-{fmt(refHigh)})</span>
      )}
    </span>
  )
}

export default function ABPMHolterDashboard() {
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
          axios.get(`${API_URL}/abpm-holter/overview`),
          axios.get(`${API_URL}/abpm-holter/breakdown`),
          axios.get(`${API_URL}/abpm-holter/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ABPM/Holter data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'abpm', label: 'ABPM Analysis' },
    { id: 'holter', label: 'Holter Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'cardiac', label: 'Cardiac Score' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = Object.entries(overview?.severity_distribution || {}).map(([severity, count]) => ({ severity, count }))
  const patternDist = Object.entries(overview?.pattern_distribution || {}).map(([pattern, count]) => ({
    pattern, count, label: pattern.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }))
  const dippingDist = Object.entries(overview?.dipping_distribution || {}).map(([category, count]) => ({
    category, count, label: category.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }))
  const patientSummary = overview?.patient_summary || []

  const abpmSummary = breakdown?.abpm_summary || []
  const holterSummary = breakdown?.holter_summary || []
  const systolicHist = breakdown?.systolic_histogram || []
  const dippingHist = breakdown?.dipping_histogram || []
  const qtcHist = breakdown?.qtc_histogram || []
  const pvcHist = breakdown?.pvc_histogram || []
  const cardiacScoreHist = breakdown?.cardiac_score_histogram || []
  const patientDetails = breakdown?.patient_detail_cards || []

  const protocol = defs?.protocol || {}
  const abpmParams = defs?.parameters?.abpm || []
  const holterParams = defs?.parameters?.holter || []
  const refRanges = defs?.reference_ranges || {}
  const dippingCategories = defs?.dipping_categories || []
  const diagPatterns = defs?.diagnostic_patterns || []
  const sevLevels = defs?.severity_levels || []
  const clinSig = defs?.clinical_significance || []

  const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
  const thStyle = { textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }
  const tdStyle = { padding: '7px 10px', borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        ABPM / Holter Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Ambulatory Blood Pressure Monitoring + Holter ECG: cardiovascular autonomic assessment with BP dipping, QTc, and arrhythmia analysis
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
              <KPI label="Total Studies" value={fmt(kpis.total_studies)} />
              <KPI label="Abnormal" value={fmt(kpis.abnormal_count)} color="#ef4444" />
              <KPI label="Abnormal Rate" value={`${fmt(kpis.abnormal_rate_pct)}%`}
                   color={kpis.abnormal_rate_pct > 40 ? '#ef4444' : '#16a34a'} />
              <KPI label="Mean Systolic 24h" value={`${fmt(kpis.mean_systolic_24h)} mmHg`} sub="24-hour average" />
              <KPI label="Mean Diastolic 24h" value={`${fmt(kpis.mean_diastolic_24h)} mmHg`} sub="24-hour average" />
              <KPI label="Mean QTc" value={`${fmt(kpis.mean_qtc_ms)} ms`} sub="Corrected QT" />
            </div>
          </Card>

          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={sevDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {sevDist.map((d, i) => <Cell key={i} fill={SEV_COLORS[d.severity] || PIE_COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Diagnostic Pattern Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={patternDist.filter(d => d.count > 0)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="label" width={160} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Dipping Category Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={dippingDist} dataKey="count" nameKey="label" cx="50%" cy="50%"
                     outerRadius={80} label={({ label, count }) => `${label}: ${count}`}>
                  {dippingDist.map((d, i) => <Cell key={i} fill={DIPPING_COLORS[d.category] || PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── ABPM Analysis Tab ─── */}
      {tab === 'abpm' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="ABPM Parameter Summary" span={2}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Mean</th>
                  <th style={thStyle}>Min</th>
                  <th style={thStyle}>Max</th>
                  <th style={thStyle}>Unit</th>
                  <th style={thStyle}>Ref Low</th>
                  <th style={thStyle}>Ref High</th>
                  <th style={thStyle}>Abnormal N</th>
                  <th style={thStyle}>Abnormal %</th>
                </tr>
              </thead>
              <tbody>
                {abpmSummary.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.parameter}</td>
                    <td style={tdStyle}>{fmt(r.mean)}</td>
                    <td style={tdStyle}>{fmt(r.min)}</td>
                    <td style={tdStyle}>{fmt(r.max)}</td>
                    <td style={{ ...tdStyle, color: '#94a3b8' }}>{r.unit}</td>
                    <td style={tdStyle}>{fmt(r.ref_low)}</td>
                    <td style={tdStyle}>{fmt(r.ref_high)}</td>
                    <td style={{ ...tdStyle, color: r.abnormal_n > 0 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{r.abnormal_n}</td>
                    <td style={{ ...tdStyle, color: r.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{fmt(r.abnormal_pct)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Systolic BP Histogram">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={systolicHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} tickFormatter={v => `${v}-${v + (systolicHist[0]?.bin_end - systolicHist[0]?.bin_start || 10)}`} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={v => `${v} mmHg`} />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Dipping % Histogram">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={dippingHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={v => `${v}%`} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Holter Analysis Tab ─── */}
      {tab === 'holter' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Holter Parameter Summary" span={2}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Mean</th>
                  <th style={thStyle}>Min</th>
                  <th style={thStyle}>Max</th>
                  <th style={thStyle}>Unit</th>
                  <th style={thStyle}>Ref Low</th>
                  <th style={thStyle}>Ref High</th>
                  <th style={thStyle}>Abnormal N</th>
                  <th style={thStyle}>Abnormal %</th>
                </tr>
              </thead>
              <tbody>
                {holterSummary.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.parameter}</td>
                    <td style={tdStyle}>{fmt(r.mean)}</td>
                    <td style={tdStyle}>{fmt(r.min)}</td>
                    <td style={tdStyle}>{fmt(r.max)}</td>
                    <td style={{ ...tdStyle, color: '#94a3b8' }}>{r.unit}</td>
                    <td style={tdStyle}>{fmt(r.ref_low)}</td>
                    <td style={tdStyle}>{fmt(r.ref_high)}</td>
                    <td style={{ ...tdStyle, color: r.abnormal_n > 0 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>{r.abnormal_n}</td>
                    <td style={{ ...tdStyle, color: r.abnormal_pct > 30 ? '#ef4444' : '#16a34a' }}>{fmt(r.abnormal_pct)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="QTc Histogram">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={qtcHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} tickFormatter={v => `${v} ms`} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={v => `${v} ms`} />
                <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="PVC Count Histogram">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={pvcHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={v => `PVC ${v}`} />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
          {patientDetails.length === 0 && <Card><p style={{ color: '#94a3b8' }}>No patient detail cards available.</p></Card>}
          {patientDetails.map((p) => {
            const isExpanded = expandedPatient === p.patient_id
            return (
              <Card key={p.patient_id}>
                <div
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
                  onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                >
                  <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, fontSize: 14 }}>{p.name || `Patient ${p.patient_id}`}</span>
                    <span style={{ fontSize: 12, color: '#64748b' }}>Age: {p.age}</span>
                    <span style={{ fontSize: 12, color: '#64748b' }}>{p.disease}</span>
                    <SeverityBadge severity={p.severity} />
                    <PatternBadge pattern={p.diagnostic_pattern} />
                    <DippingBadge category={p.dipping_category} />
                    {p.is_abnormal && <span style={{ fontSize: 10, color: '#ef4444', fontWeight: 700 }}>ABNORMAL</span>}
                  </div>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                    <span style={{ fontSize: 12, color: '#64748b' }}>Score: <strong>{fmt(p.cardiac_score)}</strong></span>
                    <span style={{ fontSize: 18, color: '#94a3b8' }}>{isExpanded ? '\u25B2' : '\u25BC'}</span>
                  </div>
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                    {/* ABPM detail */}
                    <div>
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>ABPM Parameters</h4>
                      {p.abpm && Object.keys(p.abpm).length > 0 ? (
                        <table style={tableStyle}>
                          <thead>
                            <tr>
                              <th style={thStyle}>Parameter</th>
                              <th style={thStyle}>Value</th>
                              <th style={thStyle}>Flag</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(p.abpm).map(([key, d], i) => (
                              <tr key={key} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 500 }}>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                                <td style={tdStyle}>
                                  <FlagIndicator value={d.value} flag={d.flag} unit={d.unit} refLow={d.ref_low} refHigh={d.ref_high} />
                                </td>
                                <td style={tdStyle}>
                                  <span style={{ color: d.flag === 'normal' ? '#16a34a' : '#ef4444', fontWeight: 600, fontSize: 11 }}>
                                    {d.flag ? d.flag.toUpperCase() : '--'}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No ABPM data</p>}
                    </div>

                    {/* Holter detail */}
                    <div>
                      <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Holter Parameters</h4>
                      {p.holter && Object.keys(p.holter).length > 0 ? (
                        <table style={tableStyle}>
                          <thead>
                            <tr>
                              <th style={thStyle}>Parameter</th>
                              <th style={thStyle}>Value</th>
                              <th style={thStyle}>Flag</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(p.holter).map(([key, d], i) => (
                              <tr key={key} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                                <td style={{ ...tdStyle, fontWeight: 500 }}>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                                <td style={tdStyle}>
                                  <FlagIndicator value={d.value} flag={d.flag} unit={d.unit} refLow={d.ref_low} refHigh={d.ref_high} />
                                </td>
                                <td style={tdStyle}>
                                  <span style={{ color: d.flag === 'normal' ? '#16a34a' : '#ef4444', fontWeight: 600, fontSize: 11 }}>
                                    {d.flag ? d.flag.toUpperCase() : '--'}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : <p style={{ color: '#94a3b8', fontSize: 12 }}>No Holter data</p>}
                    </div>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* ─── Cardiac Score Tab ─── */}
      {tab === 'cardiac' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Cardiac Score Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={cardiacScoreHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 10 }} tickFormatter={v => `${v}-${v + (cardiacScoreHist[0]?.bin_end - cardiacScoreHist[0]?.bin_start || 10)}`} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={v => `Score ${v}`} />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Patients by Cardiac Score">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Age</th>
                  <th style={thStyle}>Disease</th>
                  <th style={thStyle}>Cardiac Score</th>
                  <th style={thStyle}>Severity</th>
                  <th style={thStyle}>Pattern</th>
                  <th style={thStyle}>Systolic 24h</th>
                  <th style={thStyle}>Diastolic 24h</th>
                  <th style={thStyle}>QTc (ms)</th>
                  <th style={thStyle}>Dipping</th>
                  <th style={thStyle}>Abnormal</th>
                </tr>
              </thead>
              <tbody>
                {[...patientSummary].sort((a, b) => (b.cardiac_score || 0) - (a.cardiac_score || 0)).map((p, i) => (
                  <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || `Patient ${p.patient_id}`}</td>
                    <td style={tdStyle}>{p.age}</td>
                    <td style={tdStyle}>{p.disease}</td>
                    <td style={{ ...tdStyle, fontWeight: 700, color: (p.cardiac_score || 0) >= 60 ? '#ef4444' : '#1e293b' }}>{fmt(p.cardiac_score)}</td>
                    <td style={tdStyle}><SeverityBadge severity={p.severity} /></td>
                    <td style={tdStyle}><PatternBadge pattern={p.diagnostic_pattern} /></td>
                    <td style={tdStyle}>{fmt(p.systolic_24h)}</td>
                    <td style={tdStyle}>{fmt(p.diastolic_24h)}</td>
                    <td style={tdStyle}>{fmt(p.qtc_ms)}</td>
                    <td style={tdStyle}><DippingBadge category={p.dipping_category} /></td>
                    <td style={tdStyle}>
                      {p.is_abnormal
                        ? <span style={{ color: '#ef4444', fontWeight: 700, fontSize: 11 }}>YES</span>
                        : <span style={{ color: '#16a34a', fontSize: 11 }}>No</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Protocol">
            <table style={tableStyle}>
              <tbody>
                {protocol.description && <tr><td style={{ ...tdStyle, fontWeight: 600, width: 180 }}>Description</td><td style={tdStyle}>{protocol.description}</td></tr>}
                {protocol.recording_methods && (
                  <tr><td style={{ ...tdStyle, fontWeight: 600 }}>Recording Methods</td>
                    <td style={tdStyle}>{Array.isArray(protocol.recording_methods) ? protocol.recording_methods.join(', ') : protocol.recording_methods}</td></tr>
                )}
                {protocol.indications && (
                  <tr><td style={{ ...tdStyle, fontWeight: 600 }}>Indications</td>
                    <td style={tdStyle}>{Array.isArray(protocol.indications) ? protocol.indications.join(', ') : protocol.indications}</td></tr>
                )}
                {protocol.standard && <tr><td style={{ ...tdStyle, fontWeight: 600 }}>Standard</td><td style={tdStyle}>{protocol.standard}</td></tr>}
                {protocol.ai_models && (
                  <tr><td style={{ ...tdStyle, fontWeight: 600 }}>AI Models</td>
                    <td style={tdStyle}>{Array.isArray(protocol.ai_models) ? protocol.ai_models.join(', ') : protocol.ai_models}</td></tr>
                )}
                {protocol.biomarkers && (
                  <tr><td style={{ ...tdStyle, fontWeight: 600 }}>Biomarkers</td>
                    <td style={tdStyle}>{Array.isArray(protocol.biomarkers) ? protocol.biomarkers.join(', ') : protocol.biomarkers}</td></tr>
                )}
              </tbody>
            </table>
          </Card>

          <Card title="ABPM Parameters">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {abpmParams.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || p.parameter || p}</td>
                    <td style={tdStyle}>{p.description || p.unit || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Holter Parameters">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Parameter</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {holterParams.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || p.parameter || p}</td>
                    <td style={tdStyle}>{p.description || p.unit || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Dipping Categories">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Category</th>
                  <th style={thStyle}>Label</th>
                  <th style={thStyle}>Range</th>
                  <th style={thStyle}>Risk</th>
                </tr>
              </thead>
              <tbody>
                {dippingCategories.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><DippingBadge category={d.category} /></td>
                    <td style={tdStyle}>{d.label}</td>
                    <td style={tdStyle}>{d.range}</td>
                    <td style={tdStyle}>{d.risk}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Diagnostic Patterns">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Pattern</th>
                  <th style={thStyle}>Label</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {diagPatterns.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><PatternBadge pattern={d.pattern} /></td>
                    <td style={tdStyle}>{d.label}</td>
                    <td style={tdStyle}>{d.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Severity Levels">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Level</th>
                  <th style={thStyle}>Score Range</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {sevLevels.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><SeverityBadge severity={s.level} /></td>
                    <td style={tdStyle}>{s.score_range}</td>
                    <td style={tdStyle}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {clinSig && (
            <Card title="Clinical Significance">
              {Array.isArray(clinSig) ? (
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {clinSig.map((item, i) => <li key={i} style={{ fontSize: 13, marginBottom: 4, color: '#334155' }}>{typeof item === 'string' ? item : item.description || JSON.stringify(item)}</li>)}
                </ul>
              ) : (
                <p style={{ fontSize: 13, color: '#334155' }}>{typeof clinSig === 'string' ? clinSig : JSON.stringify(clinSig)}</p>
              )}
            </Card>
          )}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 32, textAlign: 'center', color: '#94a3b8', fontSize: 11 }}>
        ABPM / Holter Dashboard — Ambulatory Blood Pressure Monitoring + Holter ECG
      </div>
    </div>
  )
}
