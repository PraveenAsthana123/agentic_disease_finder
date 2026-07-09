import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'seizure_diary', label: 'Seizure Diary' },
  { id: 'appointments', label: 'Appointments' },
  { id: 'education', label: 'Education' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PatientPortalDashboard() {
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
      axios.get(`${API_URL}/api/patient-portal/overview`),
      axios.get(`${API_URL}/api/patient-portal/breakdown`),
      axios.get(`${API_URL}/api/patient-portal/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Patient Portal Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Patient Portal Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Seizure diary, appointments, education modules, and clinical definitions
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {!ov.available && tab !== 'definitions' && (
        <Card><p style={{ color: '#94a3b8' }}>{ov.reason || 'No data available'}</p></Card>
      )}

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && ov.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Total Seizures" value={fmt(ov?.kpis?.total_seizures)} sub="Logged events" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Upcoming Appointments" value={fmt(ov?.kpis?.upcoming_appointments)} sub="Scheduled visits" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Unread Messages" value={fmt(ov?.kpis?.unread_messages)} sub="From care team" color="#f59e0b" />
          </Card>
          <Card>
            <KPI label="Education Completion" value={pct(ov?.kpis?.education_completion_pct)} sub="Module progress" color="#10b981" />
          </Card>

          {/* Seizure Frequency Trend Line Chart */}
          <Card title="Seizure Frequency Trend" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={ov?.seizure_frequency_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="count" name="Seizures" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Appointment Status Pie Chart */}
          <Card title="Appointment Status" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={(ov?.appointment_status_distribution || []).filter(d => d.value > 0)}
                  dataKey="value"
                  nameKey="status"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.appointment_status_distribution || []).filter(d => d.value > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── SEIZURE DIARY TAB ─── */}
      {tab === 'seizure_diary' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Seizure Timeline Bar Chart */}
          <Card title="Seizure Events Per Month" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={bd?.seizure_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="events" name="Seizure Events" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Recent Seizure Events Table */}
          <Card title="Recent Seizure Events" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Date</th>
                    <th style={{ padding: '8px 12px' }}>Duration (s)</th>
                    <th style={{ padding: '8px 12px' }}>Aura</th>
                    <th style={{ padding: '8px 12px' }}>Awareness</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.recent_seizure_events || []).map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{fmt(ev.date)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(ev.duration_sec)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: ev.aura ? '#dcfce7' : '#fee2e2',
                          color: ev.aura ? '#166534' : '#991b1b'
                        }}>{ev.aura ? 'Yes' : 'No'}</span>
                      </td>
                      <td style={{ padding: '8px 12px' }}>{fmt(ev.awareness)}</td>
                    </tr>
                  ))}
                  {(bd?.recent_seizure_events || []).length === 0 && (
                    <tr><td colSpan={4} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No seizure events recorded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── APPOINTMENTS TAB ─── */}
      {tab === 'appointments' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Upcoming Appointments Table */}
          <Card title="Upcoming Appointments" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Date</th>
                    <th style={{ padding: '8px 12px' }}>Time</th>
                    <th style={{ padding: '8px 12px' }}>Type</th>
                    <th style={{ padding: '8px 12px' }}>Provider</th>
                    <th style={{ padding: '8px 12px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.upcoming_appointments || []).map((appt, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{fmt(appt.date)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(appt.time)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(appt.type)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(appt.provider)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: appt.status === 'Confirmed' ? '#dcfce7' : '#fef9c3',
                          color: appt.status === 'Confirmed' ? '#166534' : '#854d0e'
                        }}>{fmt(appt.status)}</span>
                      </td>
                    </tr>
                  ))}
                  {(bd?.upcoming_appointments || []).length === 0 && (
                    <tr><td colSpan={5} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No upcoming appointments</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Appointment Type Distribution Bar Chart */}
          <Card title="Appointment Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={bd?.appointment_type_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Appointments" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {(bd?.appointment_type_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── EDUCATION TAB ─── */}
      {tab === 'education' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Education Module Progress Table */}
          <Card title="Education Module Progress" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Module</th>
                    <th style={{ padding: '8px 12px' }}>Completion</th>
                    <th style={{ padding: '8px 12px' }}>Quiz Score</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.education_modules || []).map((mod, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}>{fmt(mod.module_name)}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3 }}>
                            <div style={{
                              width: `${mod.completion_pct || 0}%`, height: '100%',
                              background: (mod.completion_pct || 0) >= 80 ? '#10b981' : '#f59e0b',
                              borderRadius: 3
                            }} />
                          </div>
                          <span style={{ fontSize: 11, color: '#64748b', minWidth: 32 }}>{pct(mod.completion_pct)}</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: (mod.quiz_score || 0) >= 80 ? '#dcfce7' : '#fee2e2',
                          color: (mod.quiz_score || 0) >= 80 ? '#166534' : '#991b1b'
                        }}>{mod.quiz_score != null ? `${mod.quiz_score}%` : '--'}</span>
                      </td>
                    </tr>
                  ))}
                  {(bd?.education_modules || []).length === 0 && (
                    <tr><td colSpan={3} style={{ padding: '12px', color: '#94a3b8', textAlign: 'center' }}>No education modules assigned</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Education Completion Radar Chart */}
          <Card title="Module Completion Radar" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={(bd?.education_modules || []).map(m => ({
                module: m.module_name || 'Unknown',
                completion: m.completion_pct || 0,
                quiz: m.quiz_score || 0
              }))}>
                <PolarGrid />
                <PolarAngleAxis dataKey="module" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar name="Completion %" dataKey="completion" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Radar name="Quiz Score %" dataKey="quiz" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {(defs?.terms || []).map((d, i) => (
            <Card key={i} title={d.term}>
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{d.definition}</p>
              {d.category && (
                <span style={{
                  display: 'inline-block', marginTop: 8, padding: '2px 8px', borderRadius: 4,
                  fontSize: 11, background: '#ede9fe', color: '#6d28d9'
                }}>{d.category}</span>
              )}
            </Card>
          ))}
          {(defs?.terms || []).length === 0 && (
            <Card span={2}><p style={{ color: '#94a3b8' }}>No definitions available</p></Card>
          )}
        </div>
      )}
    </div>
  )
}
