import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const RISK_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' }
const RISK_LABELS = { high: 'High', medium: 'Medium', low: 'Low' }
const ADS_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444']
const HRV_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#22c55e']

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

function RiskBadge({ risk }) {
  const color = RISK_COLORS[risk] || '#94a3b8'
  const label = RISK_LABELS[risk] || risk || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function TrendBadge({ direction }) {
  const map = { improving: { color: '#22c55e', arrow: '\u2191' }, declining: { color: '#ef4444', arrow: '\u2193' }, stable: { color: '#3b82f6', arrow: '\u2192' } }
  const cfg = map[direction] || map.stable
  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: cfg.color + '22', color: cfg.color }}>
      {cfg.arrow} {direction}
    </span>
  )
}

export default function AutonomicAnalysisDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [selectedPatient, setSelectedPatient] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/autonomic-analysis/overview`),
          axios.get(`${API_URL}/api/autonomic-analysis/breakdown`),
          axios.get(`${API_URL}/api/autonomic-analysis/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message || 'Failed to load autonomic analysis data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading autonomic analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const tabs = ['overview', 'patients', 'correlation', 'devices', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: '24px 32px', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Autonomic Analysis Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        HRV trends, autonomic dysfunction scoring, seizure-autonomic correlation &amp; risk stratification
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t ? '#3b82f6' : '#64748b', fontWeight: tab === t ? 600 : 400,
            cursor: 'pointer', fontSize: 13, textTransform: 'capitalize'
          }}>{t}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {/* KPI row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
              <KPI label="Patients" value={fmt(kpis.total_patients)} />
              <KPI label="Readings" value={fmt(kpis.total_readings)} />
              <KPI label="Devices" value={fmt(kpis.total_devices)} />
              <KPI label="Avg ADS" value={fmt(kpis.avg_ads)} sub="0-100 scale" color={kpis.avg_ads > 50 ? '#ef4444' : '#3b82f6'} />
              <KPI label="Median HRV" value={fmt(kpis.median_hrv)} sub="ms" />
              <KPI label="High Risk" value={fmt(kpis.high_risk_patients)} color="#ef4444" />
              <KPI label="Seizure Detections" value={fmt(kpis.seizure_detections)} color="#f59e0b" />
            </div>
          </Card>

          {/* Risk Distribution */}
          <Card title="Risk Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.risk_distribution || []} dataKey="count" nameKey="risk" cx="50%" cy="50%" outerRadius={80} label={({ risk, count }) => `${risk}: ${count}`}>
                  {(overview.risk_distribution || []).map((e, i) => <Cell key={i} fill={RISK_COLORS[e.risk] || '#94a3b8'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* ADS Distribution */}
          <Card title="ADS Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.ads_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {(overview.ads_distribution || []).map((e, i) => <Cell key={i} fill={ADS_COLORS[i] || '#3b82f6'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* HRV Distribution */}
          <Card title="HRV Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.hrv_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {(overview.hrv_distribution || []).map((e, i) => <Cell key={i} fill={HRV_COLORS[i] || '#3b82f6'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* HRV Trend */}
          <Card title="Monthly HRV Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={overview.hrv_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_hrv" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg HRV (ms)" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'patients' && breakdown && (
        <div>
          {selectedPatient ? (
            <div>
              <button onClick={() => setSelectedPatient(null)} style={{ marginBottom: 16, padding: '6px 14px', border: '1px solid #e2e8f0', borderRadius: 8, background: '#fff', cursor: 'pointer', fontSize: 13 }}>
                &larr; Back to list
              </button>
              {(() => {
                const p = selectedPatient
                const m = p.metrics || {}
                const radarData = [
                  { metric: 'HRV', value: Math.min(100, (m.avg_hrv || 0) * 1.5), fullMark: 100 },
                  { metric: 'HR Stability', value: Math.max(0, 100 - Math.abs((m.avg_hr || 70) - 70) * 3), fullMark: 100 },
                  { metric: 'SpO2', value: (m.avg_spo2 || 95), fullMark: 100 },
                  { metric: 'Sleep', value: Math.min(100, (m.avg_sleep || 0) * 12), fullMark: 100 },
                  { metric: 'Low Stress', value: Math.max(0, 100 - (m.avg_stress || 0)), fullMark: 100 },
                  { metric: 'Activity', value: Math.min(100, (m.avg_steps || 0) / 100), fullMark: 100 },
                ]
                return (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                    <Card title={`${p.name} (${p.patient_id})`} span={2}>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
                        <KPI label="ADS" value={fmt(p.ads)} color={p.risk === 'high' ? '#ef4444' : p.risk === 'medium' ? '#f59e0b' : '#22c55e'} />
                        <KPI label="Risk" value={<RiskBadge risk={p.risk} />} />
                        <KPI label="HRV Trend" value={<TrendBadge direction={p.hrv_trend_direction} />} />
                        <KPI label="Avg HRV" value={fmt(m.avg_hrv)} sub="ms" />
                        <KPI label="Avg HR" value={fmt(m.avg_hr)} sub="bpm" />
                        <KPI label="Avg SpO2" value={fmt(m.avg_spo2)} sub="%" />
                        <KPI label="Avg Sleep" value={fmt(m.avg_sleep)} sub="hours" />
                        <KPI label="Avg Stress" value={fmt(m.avg_stress)} sub="/100" />
                        <KPI label="Seizure Events" value={fmt(m.seizure_events)} color={m.seizure_events > 0 ? '#ef4444' : '#22c55e'} />
                      </div>
                    </Card>
                    <Card title="Autonomic Radar">
                      <ResponsiveContainer width="100%" height={260}>
                        <RadarChart data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                          <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </Card>
                    <Card title="HRV Timeline">
                      <ResponsiveContainer width="100%" height={260}>
                        <LineChart data={p.daily_readings || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={{ fontSize: 9 }} />
                          <YAxis domain={['auto', 'auto']} />
                          <Tooltip />
                          <Line type="monotone" dataKey="hrv" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} name="HRV" />
                          <Line type="monotone" dataKey="stress" stroke="#f59e0b" strokeWidth={1} dot={false} name="Stress" />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                    <Card title="Vitals Timeline" span={2}>
                      <ResponsiveContainer width="100%" height={220}>
                        <LineChart data={p.daily_readings || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={{ fontSize: 9 }} />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="hr_avg" stroke="#8b5cf6" strokeWidth={1.5} dot={false} name="Heart Rate" />
                          <Line type="monotone" dataKey="spo2" stroke="#22c55e" strokeWidth={1.5} dot={false} name="SpO2" />
                          <Line type="monotone" dataKey="seizure_risk" stroke="#ef4444" strokeWidth={1.5} dot={false} name="Seizure Risk" />
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  </div>
                )
              })()}
            </div>
          ) : (
            <Card title={`Patient Autonomic Profiles (${breakdown.total_patients})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Patient', 'Age', 'Disease', 'ADS', 'Risk', 'HRV', 'HRV Trend', 'Seizure Risk', 'Device', 'Readings'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.patient_profiles || []).map(p => (
                      <tr key={p.patient_id} onClick={() => setSelectedPatient(p)} style={{ borderBottom: '1px solid #f1f5f9', cursor: 'pointer' }}
                        onMouseOver={e => e.currentTarget.style.background = '#f8fafc'} onMouseOut={e => e.currentTarget.style.background = 'transparent'}>
                        <td style={{ padding: '8px 10px', fontWeight: 500 }}>{p.name}</td>
                        <td style={{ padding: '8px 10px' }}>{p.age}</td>
                        <td style={{ padding: '8px 10px' }}>{p.disease}</td>
                        <td style={{ padding: '8px 10px', fontWeight: 600, color: p.ads > 50 ? '#ef4444' : p.ads > 30 ? '#f59e0b' : '#22c55e' }}>{fmt(p.ads)}</td>
                        <td style={{ padding: '8px 10px' }}><RiskBadge risk={p.risk} /></td>
                        <td style={{ padding: '8px 10px' }}>{fmt(p.metrics?.avg_hrv)}</td>
                        <td style={{ padding: '8px 10px' }}><TrendBadge direction={p.hrv_trend_direction} /></td>
                        <td style={{ padding: '8px 10px' }}>{fmt(p.avg_seizure_risk)}</td>
                        <td style={{ padding: '8px 10px', fontSize: 11 }}>{p.device_brand}</td>
                        <td style={{ padding: '8px 10px' }}>{p.readings_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === 'correlation' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Seizure-Autonomic Correlation" span={2}>
            <p style={{ fontSize: 13, color: '#64748b', marginBottom: 12 }}>
              Comparison of autonomic metrics during seizure detection events vs non-seizure periods
            </p>
            {(() => {
              const c = overview.seizure_autonomic_correlation || {}
              const compData = [
                { metric: 'Heart Rate (bpm)', seizure: c.avg_hr_during_seizure, normal: c.avg_hr_no_seizure },
                { metric: 'HRV (ms)', seizure: c.avg_hrv_during_seizure, normal: c.avg_hrv_no_seizure },
                { metric: 'Stress Score', seizure: c.avg_stress_during_seizure, normal: c.avg_stress_no_seizure },
              ].filter(d => d.seizure != null)
              return (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 20 }}>
                    <KPI label="Seizure Events" value={fmt(c.seizure_events)} color="#ef4444" />
                    <KPI label="Non-Seizure Events" value={fmt(c.non_seizure_events)} color="#22c55e" />
                    <KPI label="Detection Rate" value={c.seizure_events && c.non_seizure_events ? `${((c.seizure_events / (c.seizure_events + c.non_seizure_events)) * 100).toFixed(1)}%` : '--'} />
                  </div>
                  {compData.length > 0 && (
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={compData} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="metric" width={120} tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="seizure" fill="#ef4444" name="During Seizure" barSize={16} radius={[0, 4, 4, 0]} />
                        <Bar dataKey="normal" fill="#3b82f6" name="Normal" barSize={16} radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                  {compData.length === 0 && <p style={{ color: '#94a3b8', fontSize: 13 }}>No seizure detection events recorded in wearable data.</p>}
                </div>
              )
            })()}
          </Card>

          {/* High risk patients */}
          {breakdown && breakdown.high_risk_patients && breakdown.high_risk_patients.length > 0 && (
            <Card title={`High Risk Patients (${breakdown.high_risk_patients.length})`}>
              {breakdown.high_risk_patients.map(p => (
                <div key={p.patient_id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>{p.name}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8' }}>{p.disease} | HRV: {fmt(p.metrics?.avg_hrv)} ms</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontWeight: 600, color: '#ef4444', fontSize: 15 }}>ADS {fmt(p.ads)}</div>
                    <RiskBadge risk={p.risk} />
                  </div>
                </div>
              ))}
            </Card>
          )}

          {/* Declining HRV patients */}
          {breakdown && breakdown.declining_hrv_patients && breakdown.declining_hrv_patients.length > 0 && (
            <Card title={`Declining HRV Trend (${breakdown.declining_hrv_patients.length})`}>
              {breakdown.declining_hrv_patients.map(p => (
                <div key={p.patient_id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <div>
                    <div style={{ fontWeight: 500, fontSize: 13 }}>{p.name}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8' }}>Slope: {p.hrv_trend_slope}</div>
                  </div>
                  <TrendBadge direction="declining" />
                </div>
              ))}
            </Card>
          )}
        </div>
      )}

      {tab === 'devices' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Device Type Breakdown">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.device_breakdown || []} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={80} label={({ type, count }) => `${type}: ${count}`}>
                  {(overview.device_breakdown || []).map((e, i) => <Cell key={i} fill={['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444'][i % 5]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {breakdown && (
            <Card title="Device Registry">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Patient', 'Device', 'Brand', 'Status'].map(h => (
                        <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.patient_profiles || []).map(p => (
                      <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px' }}>{p.name}</td>
                        <td style={{ padding: '6px 8px' }}>{p.device_type}</td>
                        <td style={{ padding: '6px 8px' }}>{p.device_brand}</td>
                        <td style={{ padding: '6px 8px' }}>
                          <span style={{ padding: '2px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600, background: p.device_status === 'active' ? '#dcfce7' : '#fef3c7', color: p.device_status === 'active' ? '#16a34a' : '#d97706' }}>
                            {p.device_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="ADS Scoring" span={2}>
            <p style={{ fontSize: 13, color: '#334155', marginBottom: 8 }}>{defs.metrics?.ads?.name} ({defs.metrics?.ads?.range})</p>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.metrics?.ads?.components || []).map((c, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{c}</li>
              ))}
            </ul>
          </Card>

          <Card title="Risk Levels">
            {Object.entries(defs.metrics?.risk_levels || {}).map(([level, desc]) => (
              <div key={level} style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                <RiskBadge risk={level} />
                <span style={{ fontSize: 12, color: '#475569' }}>{desc}</span>
              </div>
            ))}
          </Card>

          <Card title="HRV Ranges">
            {Object.entries(defs.metrics?.hrv?.ranges || {}).map(([level, desc]) => (
              <div key={level} style={{ marginBottom: 6, fontSize: 12, color: '#475569' }}>
                <strong>{level}:</strong> {desc}
              </div>
            ))}
          </Card>

          <Card title="Data Sources" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Table', 'Rows', 'Fields'].map(h => (
                    <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.data_sources || []).map((ds, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 500 }}>{ds.table}</td>
                    <td style={{ padding: '6px 8px' }}>{ds.rows}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{ds.fields}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary" span={2}>
            {(defs.glossary || []).map((g, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <strong style={{ fontSize: 13, color: '#1e293b' }}>{g.term}</strong>
                <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{g.definition}</span>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
