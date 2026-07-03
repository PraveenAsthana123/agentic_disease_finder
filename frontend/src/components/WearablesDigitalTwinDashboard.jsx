import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const TRAJECTORY_COLORS = { improving: '#2ecc71', stable: '#f39c12', declining: '#e74c3c' }
const RISK_COLORS = { low: '#2ecc71', moderate: '#f39c12', high: '#e67e22', critical: '#e74c3c' }

function TrajectoryBadge({ trajectory }) {
  const key = trajectory?.toLowerCase()
  return <Badge text={trajectory || '--'} color={TRAJECTORY_COLORS[key] || '#64748b'} />
}

function RiskBadge({ level }) {
  const key = level?.toLowerCase()
  return <Badge text={level || '--'} color={RISK_COLORS[key] || '#64748b'} />
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'biomarkers', label: 'Biomarkers & Trends' },
  { id: 'digital_twin', label: 'Digital Twin' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function WearablesDigitalTwinDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/wearables-digital-twin/overview`),
      axios.get(`${API_URL}/api/wearables-digital-twin/breakdown`),
      axios.get(`${API_URL}/api/wearables-digital-twin/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Wearables & Digital Twin Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Wearables & Digital Twin Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Device fleet monitoring — HR/HRV biomarkers, sleep architecture, seizure detection, health & risk scores, digital twin trajectories
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Total Devices"><KPI value={ov.total_devices} label="Registered devices" color="#3b82f6" /></Card>
          <Card title="Active Devices"><KPI value={ov.active_devices} label="Currently active" color="#2ecc71" /></Card>
          <Card title="Total Readings"><KPI value={ov.total_readings} label="Biomarker readings" color="#8b5cf6" /></Card>
          <Card title="Patients Monitored"><KPI value={ov.total_patients} label="Unique patients" color="#06b6d4" /></Card>

          <Card title="Avg Health Score"><KPI value={ov.avg_health_score != null ? ov.avg_health_score.toFixed(1) : '--'} label="Health (0-100)" color="#10b981" /></Card>
          <Card title="Avg Risk Score"><KPI value={ov.avg_seizure_risk_score != null ? ov.avg_seizure_risk_score.toFixed(1) : '--'} label="Seizure risk (0-100)" color="#e74c3c" /></Card>
          <Card title="Seizure Detection"><KPI value={ov.seizure_detection_rate != null ? `${ov.seizure_detection_rate.toFixed(1)}%` : '--'} label="Detection rate" color="#f59e0b" /></Card>
          <Card title="Avg Heart Rate"><KPI value={ov.avg_heart_rate != null ? `${ov.avg_heart_rate.toFixed(1)} bpm` : '--'} label="Mean HR" color="#ec4899" /></Card>

          <Card title="Avg HRV"><KPI value={ov.avg_hrv != null ? `${ov.avg_hrv.toFixed(1)} ms` : '--'} label="SDNN" color="#3b82f6" /></Card>
          <Card title="Avg Steps"><KPI value={ov.avg_steps != null ? Math.round(ov.avg_steps).toLocaleString() : '--'} label="Daily steps" color="#f97316" /></Card>
          <Card title="Avg Sleep"><KPI value={ov.avg_sleep_duration != null ? `${ov.avg_sleep_duration.toFixed(1)} hrs` : '--'} label="Sleep duration" color="#8b5cf6" /></Card>
          <Card title="Sleep Quality"><KPI value={ov.avg_sleep_quality != null ? ov.avg_sleep_quality.toFixed(1) : '--'} label="Quality (0-100)" color="#06b6d4" /></Card>

          {/* Device Type Distribution */}
          <Card title="Device Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.device_type_distribution || []} dataKey="count" nameKey="device_type" cx="50%" cy="50%" outerRadius={80} label={e => `${e.device_type}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.device_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Device Status Distribution */}
          <Card title="Device Status" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.device_status_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="status" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Devices" radius={[4, 4, 0, 0]}>
                  {(ov.device_status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={e.status === 'active' ? '#2ecc71' : e.status === 'charging' ? '#f39c12' : e.status === 'offline' ? '#e74c3c' : '#95a5a6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Brand Distribution */}
          <Card title="Brand Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.brand_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="brand" fontSize={10} angle={-20} textAnchor="end" height={60} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Devices" radius={[4, 4, 0, 0]}>
                  {(ov.brand_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Seizure Risk Distribution */}
          <Card title="Seizure Risk Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.seizure_risk_distribution || []} dataKey="count" nameKey="risk_level" cx="50%" cy="50%" outerRadius={80} label={e => `${e.risk_level}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.seizure_risk_distribution || []).map((e, i) => (
                    <Cell key={i} fill={RISK_COLORS[e.risk_level?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* BIOMARKERS & TRENDS TAB */}
      {tab === 'biomarkers' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Health Score Trend */}
          <Card title="Health Score Trend (30 Days)">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={ov.health_score_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_health_score" name="Health Score" stroke="#10b981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Heart Rate Trend */}
          <Card title="Average Heart Rate Trend (30 Days)">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={ov.heart_rate_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_hr" name="Avg HR (bpm)" stroke="#ec4899" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* HRV Distribution */}
          <Card title="HRV Distribution (SDNN)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.hrv_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Readings" radius={[4, 4, 0, 0]}>
                  {(ov.hrv_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Sleep Quality Distribution */}
          <Card title="Sleep Quality Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.sleep_quality_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Readings" radius={[4, 4, 0, 0]}>
                  {(ov.sleep_quality_distribution || []).map((e, i) => (
                    <Cell key={i} fill={i === 0 ? '#e74c3c' : i === 1 ? '#f39c12' : i === 2 ? '#2ecc71' : '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* DIGITAL TWIN TAB */}
      {tab === 'digital_twin' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Digital Twin — Patient Health Trajectories">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Patient ID', 'Health Score', 'Risk Score', 'Trajectory', 'Avg HR', 'Avg HRV', 'Avg Sleep', 'Avg Steps', '1yr Projection', '5yr Projection'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.patients || []).map((p, i) => {
                    const twin = p.digital_twin || {}
                    const baseline = twin.physiological_baseline || {}
                    const sleepP = twin.sleep_profile || {}
                    const actP = twin.activity_profile || {}
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#10b981' }}>
                          {p.avg_health_score != null ? p.avg_health_score.toFixed(1) : '--'}
                        </td>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#e74c3c' }}>
                          {p.avg_risk_score != null ? p.avg_risk_score.toFixed(1) : '--'}
                        </td>
                        <td style={{ padding: '6px' }}><TrajectoryBadge trajectory={twin.health_trajectory} /></td>
                        <td style={{ padding: '6px' }}>{baseline.avg_hr != null ? `${baseline.avg_hr.toFixed(1)}` : '--'}</td>
                        <td style={{ padding: '6px' }}>{baseline.avg_hrv != null ? `${baseline.avg_hrv.toFixed(1)}` : '--'}</td>
                        <td style={{ padding: '6px' }}>{sleepP.avg_duration != null ? `${sleepP.avg_duration.toFixed(1)}h` : '--'}</td>
                        <td style={{ padding: '6px' }}>{actP.avg_steps != null ? Math.round(actP.avg_steps).toLocaleString() : '--'}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>{twin.longitudinal_1yr_projection || '--'}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>{twin.longitudinal_5yr_projection || '--'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Trajectory Distribution */}
          <Card title="Health Trajectory Distribution">
            {(() => {
              const trajCounts = {}
              ;(bd.patients || []).forEach(p => {
                const t = p.digital_twin?.health_trajectory || 'unknown'
                trajCounts[t] = (trajCounts[t] || 0) + 1
              })
              const trajData = Object.entries(trajCounts).map(([trajectory, count]) => ({ trajectory, count }))
              return (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={trajData} dataKey="count" nameKey="trajectory" cx="50%" cy="50%" outerRadius={80} label={e => `${e.trajectory}: ${e.count}`} labelLine fontSize={11}>
                      {trajData.map((e, i) => (
                        <Cell key={i} fill={TRAJECTORY_COLORS[e.trajectory?.toLowerCase()] || COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <Card title="Patient Device & Biomarker Detail">
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient ID', 'Device', 'Type', 'Brand', 'Status', 'Battery', 'Readings', 'Avg HR', 'Avg HRV', 'Avg Sleep', 'Seizures Detected'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => {
                  const dev = p.device || {}
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(dev.device_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(dev.device_type)}</td>
                      <td style={{ padding: '6px' }}>{fmt(dev.brand)}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={dev.status || '--'} color={dev.status === 'active' ? '#2ecc71' : dev.status === 'charging' ? '#f39c12' : '#e74c3c'} />
                      </td>
                      <td style={{ padding: '6px' }}>{dev.battery_level != null ? `${dev.battery_level}%` : '--'}</td>
                      <td style={{ padding: '6px' }}>{fmt(p.total_readings)}</td>
                      <td style={{ padding: '6px' }}>{p.avg_hr != null ? p.avg_hr.toFixed(1) : '--'}</td>
                      <td style={{ padding: '6px' }}>{p.avg_hrv != null ? p.avg_hrv.toFixed(1) : '--'}</td>
                      <td style={{ padding: '6px' }}>{p.avg_sleep != null ? `${p.avg_sleep.toFixed(1)}h` : '--'}</td>
                      <td style={{ padding: '6px', fontWeight: 600, color: '#e74c3c' }}>{fmt(p.seizures_detected)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {(defs.concepts || []).map((c, i) => (
            <Card key={i}>
              <h4 style={{ margin: '0 0 8px', fontSize: 14, color: '#1e293b' }}>{c.name}</h4>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{c.description}</p>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
