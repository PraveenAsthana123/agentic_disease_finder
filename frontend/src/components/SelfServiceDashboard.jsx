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

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'communication_scheduling', label: 'Communication & Scheduling' },
  { id: 'education_sos', label: 'Education & SOS' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SelfServiceDashboard() {
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
      axios.get(`${API_URL}/api/self-service/overview`),
      axios.get(`${API_URL}/api/self-service/breakdown`),
      axios.get(`${API_URL}/api/self-service/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Self-Service Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Self-Service / Communication / Education / Emergency Portal
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Patient self-service activity — appointments, messaging, telehealth, documents, education modules, SOS events, and daily plan completion
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
          {/* KPI Row 1 — Appointments */}
          <Card title="Total Patients">
            <KPI value={fmt(ov.total_patients)} label="Patients enrolled" color="#3b82f6" />
          </Card>
          <Card title="Total Appointments">
            <KPI value={fmt(ov.total_appointments)} label="All appointments" color="#8b5cf6" />
          </Card>
          <Card title="Upcoming Appointments">
            <KPI value={fmt(ov.upcoming_appointments)} label="Scheduled upcoming" color="#06b6d4" />
          </Card>
          <Card title="Cancelled Rate">
            <KPI value={ov.cancelled_pct != null ? `${ov.cancelled_pct}%` : '--'} label="Cancellation rate" color="#ef4444" />
          </Card>

          {/* KPI Row 2 — No-show, Messages */}
          <Card title="No-Show Rate">
            <KPI value={ov.no_show_pct != null ? `${ov.no_show_pct}%` : '--'} label="No-show rate" color="#f97316" />
          </Card>
          <Card title="Total Messages">
            <KPI value={fmt(ov.total_messages)} label="Messages sent" color="#10b981" />
          </Card>
          <Card title="Unread Messages">
            <KPI value={fmt(ov.unread_messages)} label="Awaiting response" color="#ec4899" />
          </Card>
          <Card title="Avg Response Time">
            <KPI value={ov.avg_response_time_hours != null ? `${ov.avg_response_time_hours}h` : '--'} label="Hours to respond" color="#f59e0b" />
          </Card>

          {/* KPI Row 3 — Telehealth, Satisfaction, Documents */}
          <Card title="Telehealth Sessions">
            <KPI value={fmt(ov.total_telehealth_sessions)} label="Virtual visits" color="#3b82f6" />
          </Card>
          <Card title="Avg Satisfaction">
            <KPI value={ov.avg_patient_satisfaction != null ? `${ov.avg_patient_satisfaction}/5` : '--'} label="Patient satisfaction" color="#10b981" />
          </Card>
          <Card title="Total Documents">
            <KPI value={fmt(ov.total_documents)} label="Documents on file" color="#8b5cf6" />
          </Card>
          <Card title="Documents Shared">
            <KPI value={ov.documents_shared_pct != null ? `${ov.documents_shared_pct}%` : '--'} label="Shared with patients" color="#06b6d4" />
          </Card>

          {/* KPI Row 4 — Education, SOS, Daily Plan */}
          <Card title="Education Modules">
            <KPI value={fmt(ov.total_education_modules)} label="Modules available" color="#f97316" />
          </Card>
          <Card title="Avg Education Completion">
            <KPI value={fmt(ov.avg_education_completion)} label="Completion rate" color="#ec4899" />
          </Card>
          <Card title="Total SOS Events">
            <KPI value={fmt(ov.total_sos_events)} label="Emergency triggers" color="#ef4444" />
          </Card>
          <Card title="Daily Plan Completion">
            <KPI value={fmt(ov.avg_daily_plan_completion)} label="Avg daily completion" color="#10b981" />
          </Card>

          {/* Appointment Type Distribution Bar Chart */}
          <Card title="Appointment Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.appointment_type_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Appointments" radius={[4, 4, 0, 0]}>
                  {(ov.appointment_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Appointment Status Distribution Pie Chart */}
          <Card title="Appointment Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={ov.appointment_status_distribution || []}
                  dataKey="count"
                  nameKey="status"
                  cx="50%" cy="50%"
                  outerRadius={80}
                  label={e => `${e.status}: ${e.count}`}
                  labelLine
                  fontSize={11}
                >
                  {(ov.appointment_status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Message Category Distribution Bar Chart */}
          <Card title="Message Category Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.message_category_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Messages" radius={[4, 4, 0, 0]}>
                  {(ov.message_category_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Telehealth by Type Pie Chart */}
          <Card title="Telehealth by Type" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={ov.telehealth_by_type || []}
                  dataKey="count"
                  nameKey="type"
                  cx="50%" cy="50%"
                  outerRadius={80}
                  label={e => `${e.type}: ${e.count}`}
                  labelLine
                  fontSize={11}
                >
                  {(ov.telehealth_by_type || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* COMMUNICATION & SCHEDULING TAB */}
      {tab === 'communication_scheduling' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Education Completion by Module — Horizontal Bar Chart */}
          <Card title="Education Completion by Module" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={bd.education_completion_by_module || []}
                layout="vertical"
                margin={{ left: 120 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="module" fontSize={11} width={120} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="completion_pct" name="Completion %" radius={[0, 4, 4, 0]}>
                  {(bd.education_completion_by_module || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* SOS Event Types Pie Chart */}
          <Card title="SOS Event Types" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={bd.sos_event_types || []}
                  dataKey="count"
                  nameKey="type"
                  cx="50%" cy="50%"
                  outerRadius={90}
                  label={e => `${e.type}: ${e.count}`}
                  labelLine
                  fontSize={11}
                >
                  {(bd.sos_event_types || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily Plan Completion 30-Day Trend Line Chart */}
          <Card title="Daily Plan Completion — 30-Day Trend" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={bd.daily_plan_trend_30d || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" fontSize={11} angle={-30} textAnchor="end" height={50} />
                <YAxis fontSize={11} domain={[0, 100]} unit="%" />
                <Tooltip formatter={v => `${v}%`} />
                <Line
                  type="monotone"
                  dataKey="completion_pct"
                  name="Completion %"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* EDUCATION & SOS TAB */}
      {tab === 'education_sos' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Education Completion by Module — Bar Chart */}
          <Card title="Education Completion by Module" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={bd.education_completion_by_module || []}
                layout="vertical"
                margin={{ left: 120 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="module" fontSize={11} width={120} />
                <Tooltip formatter={v => `${v}%`} />
                <Bar dataKey="completion_pct" name="Completion %" radius={[0, 4, 4, 0]}>
                  {(bd.education_completion_by_module || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* SOS Event Types Pie Chart */}
          <Card title="SOS Event Types" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={bd.sos_event_types || []}
                  dataKey="count"
                  nameKey="type"
                  cx="50%" cy="50%"
                  outerRadius={110}
                  label={e => `${e.type}: ${e.count}`}
                  labelLine
                  fontSize={11}
                >
                  {(bd.sos_event_types || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <Card title="Patient Self-Service Detail">
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
                  {[
                    'Patient ID', 'Appt Count', 'Next Appointment', 'Messages Sent',
                    'Unread', 'Telehealth Sessions', 'Documents', 'Education Progress',
                    'SOS Events', 'Daily Plan Completion'
                  ].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.appointment_count)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.next_appointment)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.messages_sent)}</td>
                    <td style={{ padding: '6px' }}>
                      <Badge
                        text={p.unread_count != null ? String(p.unread_count) : '--'}
                        color={p.unread_count > 0 ? '#ef4444' : '#10b981'}
                      />
                    </td>
                    <td style={{ padding: '6px' }}>{fmt(p.telehealth_sessions)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.documents_count)}</td>
                    <td style={{ padding: '6px' }}>
                      {p.education_progress != null ? `${p.education_progress}%` : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>
                      <Badge
                        text={p.sos_events != null ? String(p.sos_events) : '--'}
                        color={p.sos_events > 0 ? '#ef4444' : '#10b981'}
                      />
                    </td>
                    <td style={{ padding: '6px' }}>
                      {p.avg_daily_plan_completion != null ? `${p.avg_daily_plan_completion}%` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Self-Service Portal Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
