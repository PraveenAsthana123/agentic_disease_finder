import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const QUALITY_COLORS = { excellent: '#10b981', good: '#3b82f6', fair: '#f59e0b', poor: '#ef4444' }
const STATUS_COLORS = { completed: '#10b981', active: '#3b82f6', interrupted: '#f59e0b', failed: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Sessions & Patients' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CameraSeizureMonitoringDashboard() {
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
      axios.get(`${API_URL}/api/camera-seizure-monitoring/overview`),
      axios.get(`${API_URL}/api/camera-seizure-monitoring/breakdown`),
      axios.get(`${API_URL}/api/camera-seizure-monitoring/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Camera Seizure Monitoring data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Camera Seizure Monitoring Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Video camera monitoring analytics — seizure detection events, recording quality, response times, camera coverage tracking
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const locationData = Object.entries(data.camera_location_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const typeData = Object.entries(data.session_type_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const statusData = Object.entries(data.status_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const qualityData = Object.entries(data.recording_quality_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const rs = data.response_time_stats || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Total Sessions">
        <KPI value={data.total_sessions} label="monitoring sessions" sub={`${data.unique_patients} patients`} color="#3b82f6" />
      </Card>
      <Card title="Seizure Detection Rate">
        <KPI value={`${data.seizure_detection_rate_pct}%`} label="sessions with seizure"
          sub={`${data.total_seizure_events} total seizure events`} color="#ef4444" />
      </Card>
      <Card title="Avg Response Time">
        <KPI value={`${rs.avg_seconds || '--'}s`} label="alert to response"
          sub={`${rs.pct_under_2min || 0}% under 2 min`} color="#f59e0b" />
      </Card>
      <Card title="Alert Rate">
        <KPI value={`${data.alert_rate_pct}%`} label="sessions with alerts"
          sub={`Night vision: ${data.night_vision_pct}%`}
          color="#8b5cf6" />
      </Card>

      <Card title="Camera Location Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={locationData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, value }) => `${name}: ${value}`}>
              {locationData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Session Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={typeData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Sessions" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recording Quality" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={qualityData} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={70} />
            <Tooltip />
            <Bar dataKey="value" name="Sessions">
              {qualityData.map((entry, i) => (
                <Cell key={i} fill={QUALITY_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Session Status" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {statusData.map((entry, i) => (
                <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Response Time Range" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, padding: '20px 0' }}>
          <KPI value={`${rs.min_seconds || '--'}s`} label="Fastest" color="#10b981" />
          <KPI value={`${rs.avg_seconds || '--'}s`} label="Average" color="#f59e0b" />
          <KPI value={`${rs.max_seconds || '--'}s`} label="Slowest" color="#ef4444" />
        </div>
        <div style={{ fontSize: 12, color: '#64748b', textAlign: 'center', marginTop: 4 }}>
          Avg session duration: {data.avg_duration_hours}h | False alarm rate: {data.false_alarm_rate_pct}%
        </div>
      </Card>

      <Card title="Monthly Trend" span={2}>
        {(data.monthly_trend || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={data.monthly_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="sessions" stroke="#3b82f6" strokeWidth={2} name="Sessions" dot={{ r: 3 }} />
              <Line type="monotone" dataKey="seizure_events" stroke="#ef4444" strokeWidth={2} name="Seizure Events" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const locationStats = data.location_stats || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(data.active_sessions || []).length > 0 && (
        <Card title={`Active Monitoring Sessions (${data.active_sessions.length})`} span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#eff6ff' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Duration (h)</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Events</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Quality</th>
                </tr>
              </thead>
              <tbody>
                {data.active_sessions.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f0f9ff' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{s.camera_location}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{s.session_type}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{s.duration_hours}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{s.events_detected}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                        background: QUALITY_COLORS[s.recording_quality] ? QUALITY_COLORS[s.recording_quality] + '20' : '#f1f5f9',
                        color: QUALITY_COLORS[s.recording_quality] || '#64748b'
                      }}>{s.recording_quality}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Location Performance Stats" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={locationStats}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="camera_location" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="session_count" fill="#3b82f6" name="Sessions" radius={[4, 4, 0, 0]} />
            <Bar dataKey="total_seizure_events" fill="#ef4444" name="Seizure Events" radius={[4, 4, 0, 0]} />
            <Bar dataKey="total_false_alarms" fill="#94a3b8" name="False Alarms" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Location x Session Type Cross-Tab" span={1}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Session Type</th>
                <th style={{ padding: '8px 12px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {(data.location_type_crosstab || []).map((r, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.camera_location}</td>
                  <td style={{ padding: '6px 12px', borderBottom: '1px solid #f1f5f9' }}>{r.session_type}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{r.cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`Per-Patient Monitoring Summary (${(data.patient_summary || []).length} patients)`} span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Sessions</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Seizures</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Events</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>False Alarms</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Avg Dur (h)</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Top Location</th>
              </tr>
            </thead>
            <tbody>
              {(data.patient_summary || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total_sessions}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.total_seizure_events > 0 ? '#ef4444' : undefined, fontWeight: p.total_seizure_events > 0 ? 600 : 400 }}>{p.total_seizure_events}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total_events_detected}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: p.total_false_alarms > 0 ? '#94a3b8' : undefined }}>{p.total_false_alarms}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.avg_duration}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{p.most_used_location}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {(data.high_activity_patients || []).length > 0 && (
        <Card title={`High-Activity Patients (${data.high_activity_patients.length})`} span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Seizure Events</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Sessions</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Total Events</th>
                </tr>
              </thead>
              <tbody>
                {data.high_activity_patients.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#fff5f5' : '#fff' }}>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: '#ef4444', fontWeight: 600 }}>{p.total_seizure_events}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total_sessions}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{p.total_events_detected}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Recent Sessions (last 30)" span={1}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Location</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Type</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Dur (h)</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0' }}>Seizures</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Quality</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_sessions || []).map((s, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{(s.session_date || '').slice(0, 10)}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 500 }}>{s.patient_id}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{s.camera_location}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>{s.session_type}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9' }}>{s.duration_hours}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '1px solid #f1f5f9', color: s.seizure_events > 0 ? '#ef4444' : undefined, fontWeight: s.seizure_events > 0 ? 600 : 400 }}>{s.seizure_events}</td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: QUALITY_COLORS[s.recording_quality] ? QUALITY_COLORS[s.recording_quality] + '20' : '#f1f5f9',
                      color: QUALITY_COLORS[s.recording_quality] || '#64748b'
                    }}>{s.recording_quality}</span>
                  </td>
                  <td style={{ padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
                      background: STATUS_COLORS[s.status] ? STATUS_COLORS[s.status] + '20' : '#f1f5f9',
                      color: STATUS_COLORS[s.status] || '#64748b'
                    }}>{s.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title={`Clinical Glossary (${(data.glossary || []).length} terms)`} span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {(data.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
              <strong style={{ color: '#1e293b' }}>{g.term}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{g.definition}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Camera Locations">
        {(data.camera_locations || []).map((l, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.camera_locations.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#3b82f6' }}>{l.location}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{l.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Session Types">
        {(data.session_types || []).map((t, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.session_types.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <strong style={{ color: '#8b5cf6' }}>{t.type}</strong>
            <div style={{ color: '#64748b', marginTop: 2 }}>{t.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Recording Quality Tiers">
        {(data.quality_tiers || []).map((q, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.quality_tiers.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: QUALITY_COLORS[q.tier] || '#94a3b8', marginRight: 6 }} />
            <strong>{q.tier}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{q.description}</div>
          </div>
        ))}
      </Card>

      <Card title="Session Statuses">
        {(data.statuses || []).map((s, i) => (
          <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.statuses.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
            <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: STATUS_COLORS[s.status] || '#94a3b8', marginRight: 6 }} />
            <strong>{s.status}</strong>
            <div style={{ color: '#64748b', marginTop: 2, marginLeft: 16 }}>{s.description}</div>
          </div>
        ))}
      </Card>

      {(data.clinical_notes || []).length > 0 && (
        <Card title="Clinical Notes" span={2}>
          {data.clinical_notes.map((n, i) => (
            <div key={i} style={{ padding: '8px 12px', background: i % 2 ? '#f0fdf4' : '#f8fafc', borderRadius: 8, marginBottom: 6, fontSize: 12 }}>
              <div style={{ color: '#1e293b' }}>{n}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.performance_targets || []).length > 0 && (
        <Card title="Performance Targets" span={2}>
          {data.performance_targets.map((t, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: i < (data.performance_targets.length - 1) ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
              <strong style={{ color: '#1e293b' }}>{t.metric}</strong>
              <div style={{ color: '#10b981', marginTop: 2 }}>Target: {t.target}</div>
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}
