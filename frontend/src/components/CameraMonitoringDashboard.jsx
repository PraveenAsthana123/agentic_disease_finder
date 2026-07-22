import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'completed' ? '#10b981' : s === 'active' ? '#3b82f6' : s === 'interrupted' ? '#f59e0b' : s === 'failed' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s || '--'}</span>
  )
}

function QualityBadge({ quality }) {
  const q = String(quality || '').toLowerCase()
  const color = q === 'excellent' ? '#10b981' : q === 'good' ? '#0d9488' : q === 'fair' ? '#f59e0b' : q === 'poor' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{q || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sessions', label: 'All Sessions' },
  { id: 'patients', label: 'By Patient' },
  { id: 'locations', label: 'By Location' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CameraMonitoringDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/camera-monitoring/overview`),
      axios.get(`${API_URL}/api/camera-monitoring/breakdown`),
      axios.get(`${API_URL}/api/camera-monitoring/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading camera monitoring sessions...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Camera Monitoring Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Video monitoring session analytics — {fmt(overview?.total_sessions)} sessions, {fmt(overview?.total_patients)} patients, {fmt(overview?.seizure_events)} seizure events
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'sessions' && <SessionsTab sessions={breakdown?.sessions || []} />}
      {tab === 'patients' && <PatientsTab data={breakdown} />}
      {tab === 'locations' && <LocationsTab data={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const locationData = (data.location_distribution || []).map(d => ({ name: d.location, value: d.count }))
  const qualityData = (data.quality_distribution || []).map(d => ({ name: d.quality, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 12 }}>
          <KPI label="Total Sessions" value={data.total_sessions} color="#3b82f6" />
          <KPI label="Patients" value={data.total_patients} color="#8b5cf6" />
          <KPI label="Seizure Events" value={data.seizure_events} color="#ef4444" />
          <KPI label="Movement Events" value={data.movement_events} color="#f59e0b" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          <KPI label="False Alarms" value={data.false_alarms} color="#64748b" />
          <KPI label="Avg Duration" value={data.avg_duration} sub="minutes" color="#06b6d4" />
          <KPI label="Avg Response Time" value={data.avg_response_time} sub="seconds" color="#ec4899" />
          <KPI label="Alert Rate" value={`${data.alert_rate}%`} color="#f97316" />
        </div>
      </Card>

      <Card title="Location Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={locationData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {locationData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recording Quality Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={qualityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {qualityData.map((d, i) => {
                const c = d.name === 'excellent' ? '#10b981' : d.name === 'good' ? '#0d9488' : d.name === 'fair' ? '#f59e0b' : d.name === 'poor' ? '#ef4444' : COLORS[i % COLORS.length]
                return <Cell key={i} fill={c} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Session Type Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.session_type_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Sessions" radius={[4, 4, 0, 0]}>
              {(data.session_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="sessions" stroke="#3b82f6" name="Sessions" strokeWidth={2} />
            <Line type="monotone" dataKey="seizure_events" stroke="#ef4444" name="Seizure Events" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function SessionsTab({ sessions }) {
  const [sortKey, setSortKey] = useState('session_date')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = sessions.filter(s => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (s.patient_id || '').toLowerCase().includes(f) ||
           (s.location || '').toLowerCase().includes(f) ||
           (s.session_type || '').toLowerCase().includes(f) ||
           (s.status || '').toLowerCase().includes(f)
  })

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir
  })

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d * -1)
    else { setSortKey(key); setSortDir(-1) }
  }

  const hdr = (label, key) => (
    <th onClick={() => toggleSort(key)} style={{
      padding: '8px 10px', cursor: 'pointer', whiteSpace: 'nowrap', fontSize: 12,
      background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left',
      color: sortKey === key ? '#3b82f6' : '#475569'
    }}>{label} {sortKey === key ? (sortDir > 0 ? '\u25B2' : '\u25BC') : ''}</th>
  )

  return (
    <Card title={`All Camera Sessions (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient, location, type, or status..." value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Date', 'session_date')}
              {hdr('Patient', 'patient_id')}
              {hdr('Location', 'location')}
              {hdr('Type', 'session_type')}
              {hdr('Duration', 'duration_min')}
              {hdr('Events', 'total_events')}
              {hdr('Seizures', 'seizure_events')}
              {hdr('Movements', 'movement_events')}
              {hdr('False Alarms', 'false_alarms')}
              {hdr('Quality', 'recording_quality')}
              {hdr('Status', 'status')}
              <th style={{ padding: '8px 10px', fontSize: 12, background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'center', color: '#475569' }}>Night Vision</th>
              <th style={{ padding: '8px 10px', fontSize: 12, background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'center', color: '#475569' }}>Alert</th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: s.seizure_events > 0 ? '#fef2f2' : undefined }}>
                <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{s.session_date}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{s.patient_id}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(s.location || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(s.session_type || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(s.duration_min)} min</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{fmt(s.total_events)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: s.seizure_events > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{fmt(s.seizure_events)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center', color: s.movement_events > 0 ? '#f59e0b' : '#64748b' }}>{fmt(s.movement_events)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(s.false_alarms)}</td>
                <td style={{ padding: '6px 10px' }}><QualityBadge quality={s.recording_quality} /></td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={s.status} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                  {s.night_vision
                    ? <span style={{ color: '#8b5cf6', fontWeight: 600 }}>ON</span>
                    : <span style={{ color: '#94a3b8' }}>OFF</span>}
                </td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                  {s.alert_triggered
                    ? <span style={{ color: '#ef4444', fontWeight: 700 }}>ALERT</span>
                    : <span style={{ color: '#10b981' }}>OK</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} sessions</div>}
    </Card>
  )
}

function PatientsTab({ data }) {
  if (!data) return null
  const patients = data.patient_summary || []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Patient Summary (${patients.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Sessions</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Total Duration</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Seizure Events</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Movement Events</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>False Alarms</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Response Time</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.sessions)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.total_duration)} min</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: p.seizure_events > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{fmt(p.seizure_events)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: p.movement_events > 0 ? '#f59e0b' : '#64748b' }}>{fmt(p.movement_events)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.false_alarms)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.avg_response_time)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function LocationsTab({ data }) {
  if (!data) return null
  const locations = data.location_summary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title={`Location Analysis (${locations.length} locations)`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Location</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Sessions</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Seizure Events</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Movement Events</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>False Alarms</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Duration</th>
              </tr>
            </thead>
            <tbody>
              {locations.map((loc, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b', textTransform: 'capitalize' }}>{(loc.location || '').replace(/_/g, ' ')}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(loc.sessions)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: loc.seizure_events > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{fmt(loc.seizure_events)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: loc.movement_events > 0 ? '#f59e0b' : '#64748b' }}>{fmt(loc.movement_events)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(loc.false_alarms)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(loc.avg_duration)} min</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Quality Breakdown by Location" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={locations.map(l => ({ ...l, location: (l.location || '').replace(/_/g, ' ') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="location" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="quality_excellent" stackId="q" fill="#10b981" name="Excellent" />
            <Bar dataKey="quality_good" stackId="q" fill="#0d9488" name="Good" />
            <Bar dataKey="quality_fair" stackId="q" fill="#f59e0b" name="Fair" />
            <Bar dataKey="quality_poor" stackId="q" fill="#ef4444" name="Poor" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return <Card>No definitions available.</Card>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title={data.title || 'Definitions'} span={2}>
        {(data.concepts || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.name}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</div>
          </div>
        ))}
      </Card>

      {(data.session_types || []).length > 0 && (
        <Card title="Session Types">
          {data.session_types.map((st, i) => (
            <div key={i} style={{ marginBottom: 10, fontSize: 13 }}>
              <strong style={{ color: '#8b5cf6', textTransform: 'capitalize' }}>{st.type}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{st.description}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.data_sources.map((src, i) => <li key={i} style={{ marginBottom: 4 }}>{src}</li>)}
          </ul>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
