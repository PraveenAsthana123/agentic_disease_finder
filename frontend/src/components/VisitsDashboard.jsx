import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

export default function VisitsDashboard() {
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
          axios.get(`${API_URL}/api/visits/overview`),
          axios.get(`${API_URL}/api/visits/breakdown`),
          axios.get(`${API_URL}/api/visits/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load visit data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128202;</div>
      Loading visit data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No visit data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Visits' },
    { id: 'trends', label: 'Trends & Duration' },
    { id: 'recent', label: 'Recent Visits' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpi = overview.kpis || {}
  const compColor = kpi.completion_rate_pct >= 70 ? '#10b981' : kpi.completion_rate_pct >= 50 ? '#f59e0b' : '#ef4444'
  const noshowColor = kpi.no_show_rate_pct <= 5 ? '#10b981' : kpi.no_show_rate_pct <= 10 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>True Visits Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {kpi.total_visits} completed visits from {kpi.total_appointments} appointments | {kpi.unique_patients} patients | {kpi.unique_providers} providers
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
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'trends' && <TrendsTab breakdown={breakdown} />}
      {tab === 'recent' && <RecentTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, kpi, compColor, noshowColor }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Visits" value={kpi.total_visits} color="#10b981" /></Card>
      <Card><KPI label="Completion Rate" value={`${kpi.completion_rate_pct}%`} color={compColor} sub="completed / all appts" /></Card>
      <Card><KPI label="No-Show Rate" value={`${kpi.no_show_rate_pct}%`} color={noshowColor} sub="Target: <10%" /></Card>
      <Card><KPI label="Avg Duration" value={`${kpi.avg_duration_min} min`} /></Card>

      <Card><KPI label="Unique Patients" value={kpi.unique_patients} color="#3b82f6" /></Card>
      <Card><KPI label="Providers" value={kpi.unique_providers} color="#8b5cf6" /></Card>
      <Card><KPI label="Departments" value={kpi.unique_departments} color="#06b6d4" /></Card>
      <Card><KPI label="Total Appointments" value={kpi.total_appointments} color="#64748b" /></Card>

      {/* Status distribution pie */}
      <Card title="Appointment Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.status_distribution || []} dataKey="count" nameKey="status"
              cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`}>
              {(overview.status_distribution || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Visit types bar */}
      <Card title="By Visit Type" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.visit_types || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="type" width={150} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Provider visit load */}
      <Card title="Provider Visit Load" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.provider_visits || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="provider" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="visits" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Department distribution */}
      <Card title="Department Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.department_visits || []} dataKey="visits" nameKey="department"
              cx="50%" cy="50%" outerRadius={80} label={({ department, visits }) => `${department} (${visits})`}>
              {(overview.department_visits || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.per_patient || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Per-Patient Visit Summary (${patients.length} patients)`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Visits</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Avg Duration</th>
              <th style={{ textAlign: 'left', padding: 8 }}>First Visit</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Last Visit</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Departments</th>
            </tr></thead>
            <tbody>{patients.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8 }}>
                  {r.name} <span style={{ color: '#94a3b8', fontSize: 11 }}>({r.patient_id})</span>
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: '#10b981' }}>{r.visit_count}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.avg_duration_min} min</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.first_visit}</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.last_visit}</td>
                <td style={{ padding: 8, fontSize: 11 }}>{r.departments}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      {/* Provider-department cross-tab */}
      <Card title="Provider-Department Cross-Tab">
        <div style={{ maxHeight: 300, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Provider</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Department</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Visits</th>
            </tr></thead>
            <tbody>{(breakdown?.provider_department || []).map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8 }}>{r.provider}</td>
                <td style={{ padding: 8 }}>{r.department}</td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{r.visits}</td>
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
  const monthly = breakdown?.monthly_trend || []
  const duration = breakdown?.duration_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Daily visit trend */}
      <Card title="Daily Visit Trend">
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={daily}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="visits" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly visit trend */}
      <Card title="Monthly Visit Trend">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={monthly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="visits" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Duration distribution */}
      <Card title="Visit Duration Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={duration}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function RecentTab({ breakdown }) {
  const recent = breakdown?.recent_visits || []
  return (
    <Card title={`Recent Visits (last ${recent.length})`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
            <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Provider</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Department</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Scheduled</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Completed</th>
            <th style={{ textAlign: 'right', padding: 8 }}>Duration</th>
          </tr></thead>
          <tbody>{recent.map((r, i) => (
            <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: 8 }}>
                {r.name} <span style={{ color: '#94a3b8', fontSize: 11 }}>({r.patient_id})</span>
              </td>
              <td style={{ padding: 8 }}>{r.provider}</td>
              <td style={{ padding: 8 }}>{r.department}</td>
              <td style={{ padding: 8 }}>{r.type}</td>
              <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.scheduled_for}</td>
              <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.completed_at}</td>
              <td style={{ padding: 8, textAlign: 'right' }}>{r.duration_min} min</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </Card>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {(defs.sections || []).map((sec, si) => (
        <Card key={si} title={sec.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
