import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
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

export default function PatientSeenDashboard() {
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
          axios.get(`${API_URL}/patients-seen/overview`),
          axios.get(`${API_URL}/patients-seen/breakdown`),
          axios.get(`${API_URL}/patients-seen/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load patients-seen data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128202;</div>
      Loading patients-seen data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No patients-seen data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'provider', label: 'By Provider' },
    { id: 'department', label: 'By Department' },
    { id: 'patients', label: 'Patient Details' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const o = overview
  const compColor = o.completion_rate_pct >= 70 ? '#10b981' : o.completion_rate_pct >= 50 ? '#f59e0b' : '#ef4444'
  const noshowColor = o.no_show_rate_pct <= 5 ? '#10b981' : o.no_show_rate_pct <= 10 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Patients Seen Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(o.total_patients_seen)} patients seen across {fmt(o.total_completed_appointments)} completed appointments | {fmt(o.total_providers)} providers | {fmt(o.total_departments)} departments
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

      {tab === 'overview' && <OverviewTab overview={o} breakdown={breakdown} compColor={compColor} noshowColor={noshowColor} />}
      {tab === 'provider' && <ProviderTab breakdown={breakdown} />}
      {tab === 'department' && <DepartmentTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, breakdown, compColor, noshowColor }) {
  const o = overview
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Patients Seen" value={o.total_patients_seen} color="#10b981" /></Card>
      <Card><KPI label="Completed Appointments" value={o.total_completed_appointments} color="#3b82f6" /></Card>
      <Card><KPI label="Total Providers" value={o.total_providers} color="#8b5cf6" /></Card>
      <Card><KPI label="Total Departments" value={o.total_departments} color="#06b6d4" /></Card>

      <Card><KPI label="Avg Patients / Provider" value={o.avg_patients_per_provider} color="#f59e0b" /></Card>
      <Card><KPI label="Avg Duration" value={`${fmt(o.avg_duration_min)} min`} /></Card>
      <Card><KPI label="No-Show Rate" value={`${fmt(o.no_show_rate_pct)}%`} color={noshowColor} sub="Target: <10%" /></Card>
      <Card><KPI label="Completion Rate" value={`${fmt(o.completion_rate_pct)}%`} color={compColor} sub="completed / all appts" /></Card>

      {/* Appointment status distribution pie */}
      <Card title="Appointment Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={breakdown?.by_status || []} dataKey="count" nameKey="status"
              cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`}>
              {(breakdown?.by_status || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* By appointment type bar */}
      <Card title="By Appointment Type" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={breakdown?.by_appt_type || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="type" width={150} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Daily trend line chart */}
      <Card title="Daily Patients Seen Trend" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={breakdown?.daily_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="patients_seen" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ProviderTab({ breakdown }) {
  const providers = breakdown?.by_provider || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Bar chart: patients seen by provider */}
      <Card title="Patients Seen by Provider">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={providers}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="provider" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="patients_seen" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Provider performance table */}
      <Card title={`Provider Performance (${providers.length} providers)`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Provider</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Patients Seen</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Completed Appts</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Avg Duration</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Completion Rate</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Department</th>
            </tr></thead>
            <tbody>{providers.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{r.provider}</td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: '#10b981' }}>{fmt(r.patients_seen)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.completed_appointments)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.avg_duration_min)} min</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.completion_rate_pct)}%</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.department}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DepartmentTab({ breakdown }) {
  const depts = breakdown?.by_department || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Bar chart: patients by department */}
      <Card title="Patients Seen by Department">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={depts}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="department" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="patients_seen" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Department table */}
      <Card title={`Department Breakdown (${depts.length} departments)`}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Department</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Patients Seen</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Completed Appts</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Avg Duration</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Providers</th>
            </tr></thead>
            <tbody>{depts.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{r.department}</td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: '#8b5cf6' }}>{fmt(r.patients_seen)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.completed_appointments)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.avg_duration_min)} min</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.providers)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.per_patient || []
  const recent = breakdown?.recent_completed || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Per-patient visit summary */}
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
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: '#10b981' }}>{fmt(r.visit_count)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.avg_duration_min)} min</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.first_visit}</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.last_visit}</td>
                <td style={{ padding: 8, fontSize: 11 }}>{r.departments}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      {/* Recent completed appointments */}
      <Card title={`Recent Completed Appointments (last ${recent.length})`}>
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
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.duration_min)} min</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>

  const renderSection = (title, data) => {
    if (!data) return null
    const entries = typeof data === 'object' && !Array.isArray(data) ? Object.entries(data) : []
    if (entries.length === 0 && !Array.isArray(data)) return null
    return (
      <Card title={title}>
        <div style={{ display: 'grid', gap: 12 }}>
          {Array.isArray(data)
            ? data.map((item, i) => (
                <div key={i} style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5, paddingLeft: 12, borderLeft: '3px solid #e2e8f0' }}>
                  {typeof item === 'string' ? item : JSON.stringify(item)}
                </div>
              ))
            : entries.map(([key, val], i) => (
                <div key={i}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{key}</div>
                  <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                    {typeof val === 'string' ? val : JSON.stringify(val)}
                  </div>
                </div>
              ))
          }
        </div>
      </Card>
    )
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {renderSection('Concepts', defs.concepts)}
      {renderSection('Quality Metrics', defs.quality_metrics)}
      {renderSection('Clinical Relevance', defs.clinical_relevance)}
      {renderSection('Remediation Strategies', defs.remediation_strategies)}
    </div>
  )
}
