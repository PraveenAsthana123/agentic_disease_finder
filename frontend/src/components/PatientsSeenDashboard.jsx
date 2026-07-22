import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const STATUS_COLORS = {
  completed: '#10b981', scheduled: '#3b82f6', confirmed: '#8b5cf6',
  no_show: '#ef4444', cancelled: '#64748b', rescheduled: '#f59e0b'
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function PatientsSeenDashboard() {
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
          axios.get(`${API_URL}/api/patients-seen/overview`),
          axios.get(`${API_URL}/api/patients-seen/breakdown`),
          axios.get(`${API_URL}/api/patients-seen/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load patients seen data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128101;</div>
      Loading patients seen data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No patients seen data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'providers', label: 'By Provider' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'recent', label: 'Recent Visits' }
  ]

  const byProvider = (breakdown && breakdown.by_provider) || []
  const byDept = (breakdown && breakdown.by_department) || []
  const byType = (breakdown && breakdown.by_appt_type) || []
  const byStatus = (breakdown && breakdown.by_status) || []
  const dailyTrend = (breakdown && breakdown.daily_trend) || []
  const perPatient = (breakdown && breakdown.per_patient) || []
  const recentCompleted = (breakdown && breakdown.recent_completed) || []

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Patients Seen Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(overview.total_patients_seen)} patients | {fmt(overview.total_completed_appointments)} completed | {fmt(overview.total_providers)} providers | {fmt(overview.total_departments)} departments
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

      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients Seen" value={overview.total_patients_seen} color="#3b82f6" /></Card>
            <Card><KPI label="Completed Appts" value={overview.total_completed_appointments} color="#10b981" /></Card>
            <Card><KPI label="Avg Duration" value={overview.avg_duration_min} sub="minutes" color="#8b5cf6" /></Card>
            <Card><KPI label="Completion Rate" value={`${fmt(overview.completion_rate_pct)}%`} color="#f59e0b" /></Card>
            <Card><KPI label="No-Show Rate" value={`${fmt(overview.no_show_rate_pct)}%`} color="#ef4444" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Appointment Status">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={byStatus} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                    {byStatus.map((d, i) => <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="By Department">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={byDept} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="department" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="patients_seen" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Patients Seen" />
                  <Bar dataKey="completed" fill="#10b981" radius={[4, 4, 0, 0]} name="Completed" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Appointment Types">
              <ResponsiveContainer width="100%" height={Math.max(200, byType.length * 32)}>
                <BarChart data={byType} layout="vertical" margin={{ left: 120, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="appt_type" width={115} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Total" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Daily Trend">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyTrend} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={v => v.slice(5)} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="patients_seen" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} name="Patients" />
                  <Line type="monotone" dataKey="appointments" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 2 }} name="Appointments" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {tab === 'providers' && (
        <Card title={`Provider Summary (${byProvider.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Provider</th>
                  <th style={{ padding: '8px 12px' }}>Patients Seen</th>
                  <th style={{ padding: '8px 12px' }}>Total Appts</th>
                  <th style={{ padding: '8px 12px' }}>Completed</th>
                  <th style={{ padding: '8px 12px' }}>No-Show</th>
                  <th style={{ padding: '8px 12px' }}>Cancelled</th>
                  <th style={{ padding: '8px 12px' }}>Completion %</th>
                  <th style={{ padding: '8px 12px' }}>Avg Duration</th>
                </tr>
              </thead>
              <tbody>
                {byProvider.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.provider}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(p.patients_seen)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(p.total_appts)}</td>
                    <td style={{ padding: '8px 12px', color: '#10b981' }}>{fmt(p.completed)}</td>
                    <td style={{ padding: '8px 12px', color: p.no_show > 0 ? '#ef4444' : undefined }}>{fmt(p.no_show)}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{fmt(p.cancelled)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(p.completion_rate_pct)}%</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{fmt(p.avg_duration_min)} min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 20 }}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={byProvider} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="provider" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="patients_seen" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Patients Seen" />
                <Bar dataKey="no_show" fill="#ef4444" radius={[4, 4, 0, 0]} name="No-Show" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {tab === 'patients' && (
        <Card title={`Patient Detail (${perPatient.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Patient</th>
                  <th style={{ padding: '8px 10px' }}>Disease</th>
                  <th style={{ padding: '8px 10px' }}>Total Visits</th>
                  <th style={{ padding: '8px 10px' }}>Providers</th>
                  <th style={{ padding: '8px 10px' }}>Departments</th>
                  <th style={{ padding: '8px 10px' }}>Last Visit</th>
                </tr>
              </thead>
              <tbody>
                {perPatient.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 10px', textTransform: 'capitalize' }}>{p.disease || '--'}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.total_visits)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.providers_seen)}</td>
                    <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12 }}>{p.departments || '--'}</td>
                    <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12 }}>{(p.last_visit || '').slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'recent' && (
        <Card title={`Recent Completed Visits (${recentCompleted.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Patient</th>
                  <th style={{ padding: '8px 12px' }}>Provider</th>
                  <th style={{ padding: '8px 12px' }}>Department</th>
                  <th style={{ padding: '8px 12px' }}>Type</th>
                  <th style={{ padding: '8px 12px' }}>Completed</th>
                  <th style={{ padding: '8px 12px' }}>Duration</th>
                </tr>
              </thead>
              <tbody>
                {recentCompleted.map((v, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{v.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{v.provider}</td>
                    <td style={{ padding: '8px 12px' }}>{v.department}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{v.appt_type}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{(v.completed_at || '').replace('T', ' ').slice(0, 16)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(v.duration_min)} min</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {defs && (
        <div style={{ marginTop: 20, padding: 16, background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
          <strong>Definitions:</strong> {(defs.concepts || []).map(c => c.term || c.name).join(', ')}
          {defs.clinical_relevance && <span> | <strong>Relevance:</strong> {defs.clinical_relevance.slice(0, 150)}</span>}
        </div>
      )}
    </div>
  )
}
