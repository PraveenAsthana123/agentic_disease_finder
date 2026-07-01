import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const SEV_COLORS = { Normal: '#16a34a', Mild: '#3b82f6', Moderate: '#eab308', Severe: '#ef4444' }
const TYPE_COLORS = { normal: '#16a34a', demyelinating: '#8b5cf6', axonal: '#f59e0b', mixed: '#ef4444' }
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

function TypeBadge({ type }) {
  const color = TYPE_COLORS[type] || '#94a3b8'
  const label = type ? type.charAt(0).toUpperCase() + type.slice(1) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function RefIndicator({ value, ref, higher_is_bad }) {
  if (value == null || ref == null) return <span>{fmt(value)}</span>
  const abnormal = higher_is_bad ? value > ref : value < ref
  return (
    <span style={{ color: abnormal ? '#ef4444' : '#16a34a', fontWeight: abnormal ? 600 : 400 }}>
      {fmt(value)} <span style={{ fontSize: 10, color: '#94a3b8' }}>({higher_is_bad ? '<' : '>'}{fmt(ref)})</span>
    </span>
  )
}

export default function NCVDashboard() {
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
          axios.get(`${API_URL}/ncv/overview`),
          axios.get(`${API_URL}/ncv/breakdown`),
          axios.get(`${API_URL}/ncv/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading NCV data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'nerves', label: 'Nerve Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const sevDist = overview?.severity_distribution || []
  const typeDist = overview?.neuropathy_type_distribution || []
  const nerveRates = overview?.nerve_abnormality_rates || []
  const patientSummary = overview?.patient_summary || []
  const motorSummary = breakdown?.motor_summary || []
  const sensorySummary = breakdown?.sensory_summary || []
  const mcvHist = breakdown?.mcv_histogram || []
  const motorVsSensory = breakdown?.motor_vs_sensory || []
  const limbComp = breakdown?.limb_comparison || []
  const patientDetails = breakdown?.patient_details || []

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>Nerve Conduction Velocity (NCV)</h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Automated NCV analysis: latency, amplitude, conduction velocity with neuropathy grading
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
              <KPI label="Mean MCV" value={`${kpis.mean_mcv} m/s`} sub="Motor" />
              <KPI label="Mean SCV" value={`${kpis.mean_scv} m/s`} sub="Sensory" />
              <KPI label="Nerves/Study" value={kpis.nerves_tested_per_study} sub="8 nerves" />
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

          {/* Neuropathy Type Distribution */}
          <Card title="Neuropathy Type">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={typeDist} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="label" width={100} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]}>
                  {typeDist.map((d, i) => <Cell key={i} fill={TYPE_COLORS[d.type] || '#6366f1'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Nerve Abnormality Rate */}
          <Card title="Per-Nerve Abnormality Rate">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={nerveRates} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="nerve" width={140} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="rate_pct" fill="#f59e0b" radius={[0, 4, 4, 0]} />
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
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Abnormal Nerves</th>
                  </tr>
                </thead>
                <tbody>
                  {patientSummary.map((p, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{p.name || p.patient_id}</td>
                      <td style={tdStyle}>{p.age}</td>
                      <td style={tdStyle}>{p.disease}</td>
                      <td style={tdStyle}><SeverityBadge severity={p.overall_severity} /></td>
                      <td style={tdStyle}><TypeBadge type={p.neuropathy_type} /></td>
                      <td style={tdStyle}>{p.abnormal_nerves} / {p.total_nerves}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── Nerve Analysis Tab ─── */}
      {tab === 'nerves' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Motor Nerve Summary */}
          <Card title="Motor Nerve Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Nerve</th>
                  <th style={thStyle}>Limb</th>
                  <th style={thStyle}>Mean DML (ms)</th>
                  <th style={thStyle}>Mean CMAP (mV)</th>
                  <th style={thStyle}>Mean MCV (m/s)</th>
                  <th style={thStyle}>Mean F-wave (ms)</th>
                  <th style={thStyle}>Abnormal %</th>
                </tr>
              </thead>
              <tbody>
                {motorSummary.map((n, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{n.nerve}</td>
                    <td style={tdStyle}>{n.limb === 'upper' ? 'Upper' : 'Lower'}</td>
                    <td style={tdStyle}><RefIndicator value={n.mean_dml} ref={n.dml_ref} higher_is_bad={true} /></td>
                    <td style={tdStyle}><RefIndicator value={n.mean_cmap} ref={n.cmap_ref} higher_is_bad={false} /></td>
                    <td style={tdStyle}><RefIndicator value={n.mean_mcv} ref={n.mcv_ref} higher_is_bad={false} /></td>
                    <td style={tdStyle}><RefIndicator value={n.mean_fwave} ref={n.fwave_ref} higher_is_bad={true} /></td>
                    <td style={tdStyle}>
                      <span style={{ color: n.abnormal_pct > 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                        {n.abnormal_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Sensory Nerve Summary */}
          <Card title="Sensory Nerve Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Nerve</th>
                  <th style={thStyle}>Limb</th>
                  <th style={thStyle}>Mean SNAP (uV)</th>
                  <th style={thStyle}>Mean SCV (m/s)</th>
                  <th style={thStyle}>Abnormal %</th>
                </tr>
              </thead>
              <tbody>
                {sensorySummary.map((n, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{n.nerve}</td>
                    <td style={tdStyle}>{n.limb === 'upper' ? 'Upper' : 'Lower'}</td>
                    <td style={tdStyle}><RefIndicator value={n.mean_snap} ref={n.snap_ref} higher_is_bad={false} /></td>
                    <td style={tdStyle}><RefIndicator value={n.mean_scv} ref={n.scv_ref} higher_is_bad={false} /></td>
                    <td style={tdStyle}>
                      <span style={{ color: n.abnormal_pct > 30 ? '#ef4444' : '#16a34a', fontWeight: 600 }}>
                        {n.abnormal_pct}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* MCV Distribution Histogram */}
          <Card title="Motor Conduction Velocity Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={mcvHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Motor vs Sensory Comparison */}
          <Card title="Motor vs Sensory Abnormality">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={motorVsSensory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="normal" stackId="a" fill="#16a34a" name="Normal" />
                <Bar dataKey="abnormal" stackId="a" fill="#ef4444" name="Abnormal" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Limb Comparison */}
          <Card title="Upper vs Lower Limb Abnormality" span={2}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={limbComp}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="limb" tick={{ fontSize: 12 }} />
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
                  <TypeBadge type={p.neuropathy_type} />
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Motor Nerves</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Nerve</th>
                          <th style={thStyle}>DML (ms)</th>
                          <th style={thStyle}>CMAP (mV)</th>
                          <th style={thStyle}>MCV (m/s)</th>
                          <th style={thStyle}>F-wave (ms)</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(p.motor || []).map((r, j) => (
                          <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                            <td style={{ ...tdStyle, fontWeight: 500 }}>{r.nerve}</td>
                            <td style={tdStyle}><RefIndicator value={r.dml_ms} ref={r.dml_ref} higher_is_bad={true} /></td>
                            <td style={tdStyle}><RefIndicator value={r.cmap_mv} ref={r.cmap_ref} higher_is_bad={false} /></td>
                            <td style={tdStyle}><RefIndicator value={r.mcv_ms} ref={r.mcv_ref} higher_is_bad={false} /></td>
                            <td style={tdStyle}><RefIndicator value={r.fwave_ms} ref={r.fwave_ref} higher_is_bad={true} /></td>
                            <td style={tdStyle}><SeverityBadge severity={r.severity} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Sensory Nerves</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Nerve</th>
                          <th style={thStyle}>SNAP (uV)</th>
                          <th style={thStyle}>SCV (m/s)</th>
                          <th style={thStyle}>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(p.sensory || []).map((r, j) => (
                          <tr key={j} style={{ background: j % 2 === 0 ? '#fff' : '#f8fafc' }}>
                            <td style={{ ...tdStyle, fontWeight: 500 }}>{r.nerve}</td>
                            <td style={tdStyle}><RefIndicator value={r.snap_uv} ref={r.snap_ref} higher_is_bad={false} /></td>
                            <td style={tdStyle}><RefIndicator value={r.scv_ms} ref={r.scv_ref} higher_is_bad={false} /></td>
                            <td style={tdStyle}><SeverityBadge severity={r.severity} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
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
          <Card title="NCV Study Protocol">
            <p style={{ fontSize: 13, color: '#475569', margin: '0 0 12px' }}>{defs.protocol?.description}</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Motor Nerves</h4>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
                  {(defs.protocol?.nerves_tested?.motor || []).map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
              <div>
                <h4 style={{ fontSize: 13, margin: '0 0 6px', color: '#334155' }}>Sensory Nerves</h4>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
                  {(defs.protocol?.nerves_tested?.sensory || []).map((n, i) => <li key={i}>{n}</li>)}
                </ul>
              </div>
            </div>
            <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 10 }}>Standard: {defs.protocol?.standard}</p>
            <h4 style={{ fontSize: 13, margin: '12px 0 6px', color: '#334155' }}>Indications</h4>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
              {(defs.protocol?.indications || []).map((ind, i) => <li key={i}>{ind}</li>)}
            </ul>
          </Card>

          <Card title="NCV Parameters">
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
            <h4 style={{ fontSize: 13, margin: '0 0 8px', color: '#334155' }}>Motor Nerves</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
              <thead>
                <tr>
                  <th style={thStyle}>Nerve</th>
                  <th style={thStyle}>DML upper (ms)</th>
                  <th style={thStyle}>CMAP lower (mV)</th>
                  <th style={thStyle}>MCV lower (m/s)</th>
                  <th style={thStyle}>F-wave upper (ms)</th>
                </tr>
              </thead>
              <tbody>
                {(defs.reference_ranges?.motor || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.nerve}</td>
                    <td style={tdStyle}>{r.dml_upper_ms}</td>
                    <td style={tdStyle}>{r.cmap_lower_mv}</td>
                    <td style={tdStyle}>{r.mcv_lower_ms}</td>
                    <td style={tdStyle}>{r.fwave_upper_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4 style={{ fontSize: 13, margin: '0 0 8px', color: '#334155' }}>Sensory Nerves</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Nerve</th>
                  <th style={thStyle}>SNAP lower (uV)</th>
                  <th style={thStyle}>SCV lower (m/s)</th>
                </tr>
              </thead>
              <tbody>
                {(defs.reference_ranges?.sensory || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.nerve}</td>
                    <td style={tdStyle}>{r.snap_lower_uv}</td>
                    <td style={tdStyle}>{r.scv_lower_ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Neuropathy Types">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.neuropathy_types || []).map((t, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><TypeBadge type={t.type} /></td>
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
