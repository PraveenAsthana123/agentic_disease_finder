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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PatientAppointmentsDashboard() {
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
      axios.get(`${API_URL}/api/patient-appointments/overview`),
      axios.get(`${API_URL}/api/patient-appointments/breakdown`),
      axios.get(`${API_URL}/api/patient-appointments/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Patient Appointments data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Patient Appointments Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Appointment analytics — status distribution, type breakdown, provider workload, location tracking, per-patient detail
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
  const kpis = data.kpis || {}
  const statusDist = data.status_distribution || []
  const typeDist = data.type_distribution || []
  const providerDist = data.provider_distribution || []
  const locationDist = data.location_distribution || []
  const durationDist = data.duration_distribution || []
  const monthlyTrend = data.monthly_trend || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
      <Card title="Total Appointments">
        <KPI value={kpis.total_appointments} label="appointments" color="#3b82f6" />
      </Card>
      <Card title="Completion Rate">
        <KPI value={`${kpis.completion_rate}%`} label="completed" color="#10b981" />
      </Card>
      <Card title="No-Show Rate">
        <KPI value={`${kpis.no_show_rate}%`} label="no-shows" color="#ef4444" />
      </Card>
      <Card title="Cancellation Rate">
        <KPI value={`${kpis.cancellation_rate}%`} label="cancelled" color="#f59e0b" />
      </Card>
      <Card title="Reminder Sent">
        <KPI value={`${kpis.reminder_sent_pct}%`} label="reminders sent" color="#8b5cf6" />
      </Card>
      <Card title="Total Patients">
        <KPI value={kpis.total_patients} label="unique patients" color="#06b6d4" />
      </Card>

      {/* Status Distribution Pie */}
      <Card title="Status Distribution" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={statusDist} cx="50%" cy="50%" outerRadius={100} dataKey="count" nameKey="status"
              label={({ status, pct }) => `${status}: ${pct}%`}>
              {statusDist.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Appointment Type Distribution Bar */}
      <Card title="Appointment Type Distribution" span={3}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={typeDist} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={140} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Provider Workload Bar */}
      <Card title="Provider Workload" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={providerDist} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="provider" tick={{ fontSize: 11 }} width={140} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Location Distribution Pie */}
      <Card title="Location Distribution" span={3}>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={locationDist} cx="50%" cy="50%" outerRadius={80} dataKey="count" nameKey="location"
              label={({ location, pct }) => `${location}: ${pct}%`}>
              {locationDist.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Duration Distribution Bar */}
      <Card title="Duration Distribution" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={durationDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="duration" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly Trend Line */}
      <Card title="Monthly Appointment Trend" span={3}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={monthlyTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="appointments" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Appointments" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  const perPatient = data.per_patient || []
  const upcoming = data.upcoming_appointments || []
  const noShows = data.no_show_records || []
  const providerStats = data.provider_stats || []

  const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569', whiteSpace: 'nowrap' }
  const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 12, color: '#334155' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Per-Patient Summary */}
      <Card title={`Per-Patient Summary (${perPatient.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Patient</th>
              <th style={thStyle}>Total</th>
              <th style={thStyle}>Completed</th>
              <th style={thStyle}>Scheduled</th>
              <th style={thStyle}>Cancelled</th>
              <th style={thStyle}>No-Shows</th>
              <th style={thStyle}>Completion Rate</th>
              <th style={thStyle}>Top Type</th>
              <th style={thStyle}>Top Provider</th>
            </tr></thead>
            <tbody>
              {perPatient.map((p, i) => {
                const rate = p.completion_rate || 0
                const barColor = rate >= 75 ? '#10b981' : rate >= 50 ? '#f59e0b' : '#ef4444'
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={tdStyle}>{p.total}</td>
                    <td style={tdStyle}>{p.completed}</td>
                    <td style={tdStyle}>{p.scheduled}</td>
                    <td style={tdStyle}>{p.cancelled}</td>
                    <td style={tdStyle}>{p.no_shows}</td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                          width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${rate}%`, height: '100%', borderRadius: 3,
                            background: barColor
                          }} />
                        </div>
                        <span style={{ color: barColor, fontWeight: 600 }}>{rate}%</span>
                      </div>
                    </td>
                    <td style={tdStyle}>{p.top_type}</td>
                    <td style={tdStyle}>{p.top_provider}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Upcoming Appointments */}
      {upcoming.length > 0 && (
        <Card title={`Upcoming Appointments (${upcoming.length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Type</th>
                <th style={thStyle}>Provider</th>
                <th style={thStyle}>Location</th>
                <th style={thStyle}>Duration (min)</th>
                <th style={thStyle}>Reminder Sent</th>
              </tr></thead>
              <tbody>
                {upcoming.map((a, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{a.patient_id}</td>
                    <td style={tdStyle}>{a.date}</td>
                    <td style={tdStyle}>{a.type}</td>
                    <td style={tdStyle}>{a.provider}</td>
                    <td style={tdStyle}>{a.location}</td>
                    <td style={tdStyle}>{a.duration}</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>
                      {a.reminder_sent ? <span style={{ color: '#10b981', fontSize: 16 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontSize: 14 }}>&#10007;</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* No-Show Records */}
      {noShows.length > 0 && (
        <Card title={`No-Show Records (${noShows.length})`}>
          <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12 }}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Provider</th>
                  <th style={thStyle}>Location</th>
                  <th style={thStyle}>Reminder Sent</th>
                </tr></thead>
                <tbody>
                  {noShows.map((n, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.5)' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{n.patient_id}</td>
                      <td style={tdStyle}>{n.date}</td>
                      <td style={tdStyle}>{n.type}</td>
                      <td style={tdStyle}>{n.provider}</td>
                      <td style={tdStyle}>{n.location}</td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>
                        {n.reminder_sent ? <span style={{ color: '#10b981', fontSize: 16 }}>&#10003;</span> : <span style={{ color: '#ef4444', fontSize: 14 }}>&#10007;</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      )}

      {/* Provider Stats */}
      <Card title={`Provider Stats (${providerStats.length} providers)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Provider</th>
              <th style={thStyle}>Total</th>
              <th style={thStyle}>Completed</th>
              <th style={thStyle}>No-Show Rate</th>
              <th style={thStyle}>Avg Duration (min)</th>
            </tr></thead>
            <tbody>
              {providerStats.map((p, i) => {
                const nsRate = p.no_show_rate || 0
                const barColor = nsRate <= 10 ? '#10b981' : nsRate <= 20 ? '#f59e0b' : '#ef4444'
                return (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.provider}</td>
                    <td style={tdStyle}>{p.total}</td>
                    <td style={tdStyle}>{p.completed}</td>
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{
                          width: 60, height: 6, background: '#e2e8f0', borderRadius: 3, overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${nsRate}%`, height: '100%', borderRadius: 3,
                            background: barColor
                          }} />
                        </div>
                        <span style={{ color: barColor, fontWeight: 600 }}>{nsRate}%</span>
                      </div>
                    </td>
                    <td style={tdStyle}>{p.avg_duration}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const glossary = data.glossary || {}
  const typeDescriptions = data.type_descriptions || {}
  const statusDefinitions = data.status_definitions || {}
  const locationNotes = data.location_notes || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Glossary */}
      {Object.keys(glossary).length > 0 && (
        <Card title={`Clinical Glossary (${Object.keys(glossary).length} terms)`}>
          <dl style={{ margin: 0 }}>
            {Object.entries(glossary).map(([term, desc], i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <dt style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{term}</dt>
                <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{desc}</dd>
              </div>
            ))}
          </dl>
        </Card>
      )}

      {/* Appointment Type Descriptions */}
      {Object.keys(typeDescriptions).length > 0 && (
        <Card title={`Appointment Types (${Object.keys(typeDescriptions).length})`}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {Object.entries(typeDescriptions).map(([name, desc], i) => (
              <div key={i} style={{
                background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{name}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Status Definitions */}
      {Object.keys(statusDefinitions).length > 0 && (
        <Card title={`Status Definitions (${Object.keys(statusDefinitions).length})`}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {Object.entries(statusDefinitions).map(([name, desc], i) => (
              <div key={i} style={{
                background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{name}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Location Notes */}
      {Object.keys(locationNotes).length > 0 && (
        <Card title={`Location Notes (${Object.keys(locationNotes).length})`}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {Object.entries(locationNotes).map(([name, desc], i) => (
              <div key={i} style={{
                background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0'
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{name}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
