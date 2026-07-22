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
  normal: '#16a34a',
  mild_parasympathetic: '#3b82f6',
  moderate_autonomic_neuropathy: '#8b5cf6',
  severe_autonomic_neuropathy: '#dc2626',
  pots: '#f59e0b',
  cardiovagal_failure: '#ef4444',
  adrenergic_failure: '#d946ef',
  sudep_risk: '#b91c1c'
}
const TEST_STATUS_COLORS = {
  Normal: '#16a34a',
  Borderline: '#eab308',
  Abnormal: '#ef4444',
  Absent: '#94a3b8'
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

function TestStatusBadge({ status }) {
  const color = TEST_STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status || 'Unknown'}</span>
  )
}

export default function AutonomicDashboard() {
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
          axios.get(`${API_URL}/api/autonomic/overview`),
          axios.get(`${API_URL}/api/autonomic/breakdown`),
          axios.get(`${API_URL}/api/autonomic/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Autonomic data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'parasympathetic', label: 'Parasympathetic Analysis' },
    { id: 'sympathetic', label: 'Sympathetic Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const patternDist = overview?.pattern_distribution || []
  const valsalvaHist = overview?.valsalva_histogram || []
  const orthoHist = overview?.orthostatic_histogram || []
  const casiHist = overview?.casi_histogram || []

  const patients = breakdown?.patients || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  // Compute parasympathetic pass rates across patients
  const paraTestNames = ['Valsalva Ratio', 'E:I Ratio', '30:15 Ratio']
  const paraPassRates = paraTestNames.map(name => {
    const relevant = patients.filter(p => (p.parasympathetic_tests || []).some(t => t.test === name))
    const normal = relevant.filter(p => (p.parasympathetic_tests || []).find(t => t.test === name)?.status === 'Normal').length
    return { test: name, pass_rate: relevant.length ? Math.round((normal / relevant.length) * 100) : 0, total: relevant.length }
  })

  // Compute sympathetic pass rates across patients
  const sympTestNames = ['Orthostatic BP Drop', 'SSR Hand', 'SSR Foot', 'Handgrip DBP Rise', 'Cold Pressor DBP Rise']
  const sympPassRates = sympTestNames.map(name => {
    const relevant = patients.filter(p => (p.sympathetic_tests || []).some(t => t.test === name))
    const normal = relevant.filter(p => (p.sympathetic_tests || []).find(t => t.test === name)?.status === 'Normal').length
    return { test: name, pass_rate: relevant.length ? Math.round((normal / relevant.length) * 100) : 0, total: relevant.length }
  })

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Autonomic Function Tests
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Cardiovascular autonomic battery: Ewing parasympathetic + sympathetic testing with CASI scoring and SUDEP risk assessment
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
          {/* 8 KPI cards */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Studies" value={fmt(kpis.total_studies)} />
              <KPI label="Total Patients" value={fmt(kpis.total_patients)} />
              <KPI label="Normal %" value={`${fmt(kpis.normal_pct)}%`} color="#16a34a" />
              <KPI label="Abnormal %" value={`${fmt(kpis.abnormal_pct)}%`}
                   color={kpis.abnormal_pct > 50 ? '#ef4444' : '#eab308'} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Avg CASI Score" value={fmt(kpis.avg_casi_score)} sub="0–100 composite" />
              <KPI label="SUDEP Risk Count" value={fmt(kpis.sudep_risk_count)}
                   color={kpis.sudep_risk_count > 0 ? '#b91c1c' : '#16a34a'} sub="High-risk patients" />
              <KPI label="Mean Valsalva Ratio" value={fmt(kpis.mean_valsalva_ratio)} sub="Normal ≥ 1.20" />
              <KPI label="Mean Orthostatic Drop" value={`${fmt(kpis.mean_orthostatic_drop)} mmHg`} sub="Normal < 20 mmHg" />
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
                <YAxis type="category" dataKey="label" width={180} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {patternDist.filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={PATTERN_COLORS[d.pattern] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Valsalva Ratio Histogram */}
          <Card title="Valsalva Ratio Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={valsalvaHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} label={{ value: 'Ratio', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Normal threshold: ≥ 1.20 (Ewing criterion)
            </div>
          </Card>

          {/* Orthostatic Drop Histogram */}
          <Card title="Orthostatic BP Drop Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={orthoHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} label={{ value: 'mmHg', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              Orthostatic hypotension threshold: ≥ 20 mmHg systolic drop
            </div>
          </Card>

          {/* CASI Score Histogram */}
          <Card title="CASI Score Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={casiHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} label={{ value: 'Score', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              CASI = Composite Autonomic Severity Index (0–10 scale)
            </div>
          </Card>
        </div>
      )}

      {/* ─── Parasympathetic Analysis Tab ─── */}
      {tab === 'parasympathetic' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Parasympathetic Test Pass Rates" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={paraPassRates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="test" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="pass_rate" fill="#16a34a" radius={[4, 4, 0, 0]}>
                  {paraPassRates.map((d, i) => (
                    <Cell key={i} fill={d.pass_rate >= 70 ? '#16a34a' : d.pass_rate >= 50 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Patients — Parasympathetic Tests" span={2}>
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Age</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Valsalva Ratio</th>
                    <th style={thStyle}>E:I Ratio</th>
                    <th style={thStyle}>30:15 Ratio</th>
                    <th style={thStyle}>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => {
                    const getTest = (name) => (p.parasympathetic_tests || []).find(t => t.test === name)
                    const vr = getTest('Valsalva Ratio')
                    const ei = getTest('E:I Ratio')
                    const r3015 = getTest('30:15 Ratio')
                    return (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || p.patient_id}</td>
                        <td style={tdStyle}>{p.age}</td>
                        <td style={tdStyle}>{p.disease}</td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{vr ? fmt(vr.value) : '--'}</span>
                          {vr && <TestStatusBadge status={vr.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{ei ? fmt(ei.value) : '--'}</span>
                          {ei && <TestStatusBadge status={ei.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{r3015 ? fmt(r3015.value) : '--'}</span>
                          {r3015 && <TestStatusBadge status={r3015.status} />}
                        </td>
                        <td style={tdStyle}><SeverityBadge severity={p.severity} /></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Sympathetic Analysis Tab ─── */}
      {tab === 'sympathetic' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Sympathetic Test Pass Rates" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sympPassRates}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="test" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="pass_rate" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {sympPassRates.map((d, i) => (
                    <Cell key={i} fill={d.pass_rate >= 70 ? '#16a34a' : d.pass_rate >= 50 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Patients — Sympathetic Tests" span={2}>
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Orthostatic Drop</th>
                    <th style={thStyle}>SSR Hand</th>
                    <th style={thStyle}>SSR Foot</th>
                    <th style={thStyle}>Handgrip DBP Rise</th>
                    <th style={thStyle}>Cold Pressor DBP</th>
                    <th style={thStyle}>SUDEP Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => {
                    const getTest = (name) => (p.sympathetic_tests || []).find(t => t.test === name)
                    const ortho = getTest('Orthostatic BP Drop')
                    const ssrHand = getTest('SSR Hand')
                    const ssrFoot = getTest('SSR Foot')
                    const handgrip = getTest('Handgrip DBP Rise')
                    const cold = getTest('Cold Pressor DBP Rise')
                    return (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || p.patient_id}</td>
                        <td style={tdStyle}>{p.disease}</td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{ortho ? `${fmt(ortho.value)} ${ortho.unit || 'mmHg'}` : '--'}</span>
                          {ortho && <TestStatusBadge status={ortho.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{ssrHand ? `${fmt(ssrHand.value)} ${ssrHand.unit || ''}` : '--'}</span>
                          {ssrHand && <TestStatusBadge status={ssrHand.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{ssrFoot ? `${fmt(ssrFoot.value)} ${ssrFoot.unit || ''}` : '--'}</span>
                          {ssrFoot && <TestStatusBadge status={ssrFoot.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{handgrip ? `${fmt(handgrip.value)} ${handgrip.unit || 'mmHg'}` : '--'}</span>
                          {handgrip && <TestStatusBadge status={handgrip.status} />}
                        </td>
                        <td style={tdStyle}>
                          <span style={{ marginRight: 6 }}>{cold ? `${fmt(cold.value)} ${cold.unit || 'mmHg'}` : '--'}</span>
                          {cold && <TestStatusBadge status={cold.status} />}
                        </td>
                        <td style={tdStyle}>
                          {p.sudep_risk
                            ? <span style={{ fontSize: 11, fontWeight: 700, color: '#b91c1c', background: '#fee2e2', padding: '2px 8px', borderRadius: 6 }}>&#9888; SUDEP Risk</span>
                            : <span style={{ fontSize: 11, color: '#16a34a' }}>Low</span>}
                        </td>
                      </tr>
                    )
                  })}
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
                  <PatternBadge pattern={p.diagnostic_pattern} />
                  <span style={{ fontSize: 12, color: '#64748b' }}>
                    CASI: <strong>{fmt(p.casi_score)}</strong>
                  </span>
                  {p.sudep_risk && (
                    <span style={{
                      fontSize: 11, fontWeight: 700, color: '#b91c1c',
                      background: '#fee2e2', padding: '2px 8px', borderRadius: 6
                    }}>&#9888; SUDEP Risk</span>
                  )}
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Demographics row */}
                    <div style={{ display: 'flex', gap: 20, marginBottom: 16, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569' }}>
                      <span><strong>Patient ID:</strong> {p.patient_id}</span>
                      <span><strong>Age:</strong> {p.age}</span>
                      <span><strong>Disease:</strong> {p.disease}</span>
                      <span><strong>CASI Score:</strong> {fmt(p.casi_score)}</span>
                      <span><strong>Pattern:</strong> {p.diagnostic_pattern ? p.diagnostic_pattern.replace(/_/g, ' ') : '--'}</span>
                      {p.sudep_risk && (
                        <span style={{ color: '#b91c1c', fontWeight: 700 }}>&#9888; SUDEP Risk Flagged</span>
                      )}
                    </div>

                    {/* Parasympathetic Tests */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Parasympathetic Tests (Cardiovagal)</h4>
                    {(p.parasympathetic_tests || []).length > 0 ? (
                      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Test</th>
                            <th style={thStyle}>Value</th>
                            <th style={thStyle}>Unit</th>
                            <th style={thStyle}>Status</th>
                            <th style={thStyle}>Reference</th>
                          </tr>
                        </thead>
                        <tbody>
                          {p.parasympathetic_tests.map((t, j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 600 }}>{t.test}</td>
                              <td style={tdStyle}>
                                <span style={{ color: t.status === 'Normal' ? '#16a34a' : t.status === 'Borderline' ? '#eab308' : '#ef4444', fontWeight: 600 }}>
                                  {fmt(t.value)}
                                </span>
                              </td>
                              <td style={tdStyle}>{t.unit || '--'}</td>
                              <td style={tdStyle}><TestStatusBadge status={t.status} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>{t.reference || '--'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>No parasympathetic test data available.</p>
                    )}

                    {/* Sympathetic Tests */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Sympathetic Tests (Adrenergic)</h4>
                    {(p.sympathetic_tests || []).length > 0 ? (
                      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
                        <thead>
                          <tr>
                            <th style={thStyle}>Test</th>
                            <th style={thStyle}>Value</th>
                            <th style={thStyle}>Unit</th>
                            <th style={thStyle}>Status</th>
                            <th style={thStyle}>Reference</th>
                          </tr>
                        </thead>
                        <tbody>
                          {p.sympathetic_tests.map((t, j) => (
                            <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                              <td style={{ ...tdStyle, fontWeight: 600 }}>{t.test}</td>
                              <td style={tdStyle}>
                                <span style={{ color: t.status === 'Normal' ? '#16a34a' : t.status === 'Borderline' ? '#eab308' : t.status === 'Absent' ? '#94a3b8' : '#ef4444', fontWeight: 600 }}>
                                  {fmt(t.value)}
                                </span>
                              </td>
                              <td style={tdStyle}>{t.unit || '--'}</td>
                              <td style={tdStyle}><TestStatusBadge status={t.status} /></td>
                              <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>{t.reference || '--'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8' }}>No sympathetic test data available.</p>
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
                      <th style={{ ...thStyle, width: '30%' }}>Term</th>
                      <th style={thStyle}>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.items.map((item, ii) => (
                      <tr key={ii} style={{ background: ii % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{item.term}</td>
                        <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.detail}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : section.description ? (
                <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{section.description}</p>
              ) : null}
            </Card>
          ))}
          {/* Fallback if no sections */}
          {(!defs?.sections || defs.sections.length === 0) && (
            <Card title="Definitions">
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
            </Card>
          )}
        </div>
      )}

      <div style={{ marginTop: 24, padding: 16, background: '#f1f5f9', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
        Autonomic Function Tests Dashboard — Real clinical.db data ({kpis.total_studies} studies, {patients.length} patients) |
        Ewing autonomic battery with SUDEP risk assessment
      </div>
    </div>
  )
}
