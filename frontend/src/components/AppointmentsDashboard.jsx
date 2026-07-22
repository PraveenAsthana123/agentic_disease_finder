import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = {
  completed: '#10b981', booked: '#3b82f6', confirmed: '#06b6d4',
  'no-show': '#ef4444', cancelled: '#64748b', rescheduled: '#f59e0b'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{status}</span>
  )
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

export default function AppointmentsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/appointments/overview`),
          axios.get(`${API_URL}/api/appointments/breakdown`),
          axios.get(`${API_URL}/api/appointments/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load appointment data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128197;</div>
      Loading appointment data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview || overview.available === false) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No appointment data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'providers', label: 'Providers & Departments' },
    { id: 'trends', label: 'Activity & Trends' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpi = overview.kpi || {}
  const compColor = kpi.completion_rate_pct >= 85 ? '#10b981' : kpi.completion_rate_pct >= 70 ? '#f59e0b' : '#ef4444'
  const noshowColor = kpi.no_show_rate_pct <= 5 ? '#10b981' : kpi.no_show_rate_pct <= 10 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Appointments Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {kpi.total_appointments} appointments | {kpi.unique_patients} patients | {(overview.provider_workload || []).length} providers
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} kpi={kpi} compColor={compColor} noshowColor={noshowColor} />}
      {tab === 'providers' && <ProvidersTab overview={overview} breakdown={breakdown} />}
      {tab === 'trends' && <TrendsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, kpi, compColor, noshowColor }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Appointments" value={kpi.total_appointments} /></Card>
      <Card><KPI label="Completion Rate" value={`${kpi.completion_rate_pct}%`} color={compColor} sub="Target: >85%" /></Card>
      <Card><KPI label="No-show Rate" value={`${kpi.no_show_rate_pct}%`} color={noshowColor} sub="Target: <10%" /></Card>
      <Card><KPI label="Avg Duration" value={`${kpi.avg_duration_min} min`} /></Card>

      <Card title="Completed" ><KPI label="" value={kpi.completed} color="#10b981" /></Card>
      <Card title="Upcoming"><KPI label="Booked + Confirmed" value={kpi.booked_upcoming} color="#3b82f6" /></Card>
      <Card title="No-shows"><KPI label="" value={kpi.no_shows} color="#ef4444" /></Card>
      <Card title="Cancelled / Rescheduled"><KPI label="" value={(kpi.cancelled || 0) + (kpi.rescheduled || 0)} color="#64748b" /></Card>

      {/* Status distribution pie */}
      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.status_distribution || []} dataKey="count" nameKey="status"
              cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`}>
              {(overview.status_distribution || []).map((e, i) => (
                <Cell key={i} fill={STATUS_COLORS[e.status] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Appointment types bar */}
      <Card title="By Appointment Type" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.appointment_types || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="type" width={150} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ProvidersTab({ overview, breakdown }) {
  const provWork = overview.provider_workload || []
  const provDept = breakdown?.provider_department_matrix || []
  const patients = breakdown?.patient_appointments || []
  const noshowByType = breakdown?.noshow_by_type || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Provider workload */}
      <Card title="Provider Workload" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={provWork}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="provider" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="completed" fill="#10b981" name="Completed" stackId="a" />
            <Bar dataKey="no_show" fill="#ef4444" name="No-show" stackId="a" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Department distribution */}
      <Card title="Department Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.department_distribution || []} dataKey="count" nameKey="department"
              cx="50%" cy="50%" outerRadius={70} label={({ department, count }) => `${department} (${count})`}>
              {(overview.department_distribution || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* No-show by type */}
      <Card title="No-show by Appointment Type">
        {noshowByType.length === 0 ? <div style={{ color: '#94a3b8', fontSize: 13 }}>No no-shows recorded</div> : (
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Total</th>
              <th style={{ textAlign: 'right', padding: 6 }}>No-show</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Rate</th>
            </tr></thead>
            <tbody>{noshowByType.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{r.type}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.total}</td>
                <td style={{ padding: 6, textAlign: 'right', color: '#ef4444' }}>{r.no_show}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.rate_pct}%</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </Card>

      {/* Per-patient summary */}
      <Card title="Patient Appointment Summary" span={2}>
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Total</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Completed</th>
              <th style={{ textAlign: 'right', padding: 6 }}>No-show</th>
            </tr></thead>
            <tbody>{patients.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{r.name} <span style={{ color: '#94a3b8' }}>({r.patient_id})</span></td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.total}</td>
                <td style={{ padding: 6, textAlign: 'right', color: '#10b981' }}>{r.completed}</td>
                <td style={{ padding: 6, textAlign: 'right', color: r.no_show > 0 ? '#ef4444' : '#94a3b8' }}>{r.no_show}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function TrendsTab({ breakdown }) {
  const daily = breakdown?.daily_trend || []
  const hourly = breakdown?.hourly_pattern || []
  const recent = breakdown?.recent_appointments || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Daily trend */}
      <Card title="Daily Appointment Trend">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={daily}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} name="Total" dot={false} />
            <Line type="monotone" dataKey="completed" stroke="#10b981" strokeWidth={2} name="Completed" dot={false} />
            <Line type="monotone" dataKey="no_show" stroke="#ef4444" strokeWidth={1} name="No-show" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Hourly pattern */}
      <Card title="Hourly Booking Pattern">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={hourly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" tick={{ fontSize: 11 }} tickFormatter={h => `${h}:00`} />
            <YAxis />
            <Tooltip labelFormatter={h => `${h}:00`} />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Recent appointments */}
      <Card title="Recent Appointments">
        <div style={{ maxHeight: 350, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Provider</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Status</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Scheduled</th>
              <th style={{ textAlign: 'right', padding: 6 }}>Min</th>
            </tr></thead>
            <tbody>{recent.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{r.name} <span style={{ color: '#94a3b8' }}>({r.patient_id})</span></td>
                <td style={{ padding: 6 }}>{r.provider}</td>
                <td style={{ padding: 6 }}>{r.type}</td>
                <td style={{ padding: 6 }}><StatusBadge status={r.status} /></td>
                <td style={{ padding: 6, fontSize: 11 }}>{r.scheduled_for}</td>
                <td style={{ padding: 6, textAlign: 'right' }}>{r.duration_min}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  const definitions = defs?.definitions || {}
  return (
    <Card title="Metric Definitions">
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <tbody>{Object.entries(definitions).map(([k, v]) => (
          <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
            <td style={{ padding: 8, fontWeight: 600, color: '#334155', whiteSpace: 'nowrap', verticalAlign: 'top' }}>
              {k.replace(/_/g, ' ')}
            </td>
            <td style={{ padding: 8, color: '#64748b' }}>{v}</td>
          </tr>
        ))}</tbody>
      </table>
    </Card>
  )
}
