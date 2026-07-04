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

const SEV_COLORS = {
  Mild: '#10b981',
  Moderate: '#f59e0b',
  Severe: '#ef4444',
  Critical: '#7f1d1d',
  Unknown: '#94a3b8',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Summary' },
  { id: 'events', label: 'Event Log' },
  { id: 'triggers', label: 'Trigger Analysis' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SeizureDiaryDashboard() {
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
      axios.get(`${API_URL}/api/seizure-diary-dashboard/overview`),
      axios.get(`${API_URL}/api/seizure-diary-dashboard/breakdown`),
      axios.get(`${API_URL}/api/seizure-diary-dashboard/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Seizure Diary Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  const severityPieData = Object.entries(ov.severity_distribution || {}).map(([level, count]) => ({
    name: level, value: count
  }))

  const triggerBarData = Object.entries(ov.top_triggers || {}).map(([trigger, count]) => ({
    name: trigger, count
  }))

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Seizure Diary Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Patient-reported seizure event log with severity tracking, monthly trends, trigger analysis, and ER monitoring
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
          <Card title="Total Events">
            <KPI value={fmt(ov.total_events)} label="Seizure events recorded" color="#3b82f6" />
          </Card>
          <Card title="Unique Patients">
            <KPI value={fmt(ov.unique_patients)} label="Patients with diary entries" color="#8b5cf6" />
          </Card>
          <Card title="Avg Duration">
            <KPI value={ov.avg_duration_sec != null ? `${ov.avg_duration_sec}s` : '--'} label="Mean seizure duration" color="#f59e0b" />
          </Card>
          <Card title="ER Visits">
            <KPI value={fmt(ov.er_visits)} label={`${fmt(ov.er_rate_pct)}% of events`} color="#ef4444" sub={`${fmt(ov.injury_count)} injuries`} />
          </Card>

          {/* Severity Distribution Pie */}
          <Card title="Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={severityPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {severityPieData.map((entry, i) => (
                    <Cell key={i} fill={SEV_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Top Triggers Bar */}
          <Card title="Top Triggers" span={2}>
            {triggerBarData.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={triggerBarData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No trigger data recorded yet</p>
            )}
          </Card>

          {/* Monthly Trend */}
          {(ov.monthly_trend || []).length > 0 && (
            <Card title="Monthly Trend" span={4}>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={ov.monthly_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="events" name="Events" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="er_visits" name="ER Visits" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* PATIENT SUMMARY TAB */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Patient Summary (${(bd.patient_summary || []).length} patients)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Events</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Avg Duration</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>ER Visits</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Injuries</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Worst Severity</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Last Event</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Severity Breakdown</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.total_events}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.avg_duration_sec}s</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', color: p.er_visits > 0 ? '#ef4444' : undefined }}>{p.er_visits}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', color: p.injuries > 0 ? '#f59e0b' : undefined }}>{p.injuries}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <Badge text={p.worst_severity} color={SEV_COLORS[p.worst_severity] || '#64748b'} />
                      </td>
                      <td style={{ padding: '6px 10px', fontSize: 12, color: '#94a3b8' }}>{p.last_event}</td>
                      <td style={{ padding: '6px 10px' }}>
                        {Object.entries(p.severity_counts || {}).map(([sev, cnt]) => (
                          <span key={sev} style={{ marginRight: 8, fontSize: 11 }}>
                            <Badge text={`${sev}: ${cnt}`} color={SEV_COLORS[sev] || '#94a3b8'} />
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* EVENT LOG TAB */}
      {tab === 'events' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Recent Events (${(bd.event_log || []).length} shown)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Duration</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Severity</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Injury</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>ER</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Trigger</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Recovery (min)</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Rescue Med</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.event_log || []).map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: e.severity === 'Severe' || e.severity === 'Critical' ? '#fef2f210' : undefined }}>
                      <td style={{ padding: '6px 10px', fontSize: 12 }}>{e.event_date}{e.event_time ? ` ${e.event_time}` : ''}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{e.patient_id}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{e.duration_sec != null ? `${e.duration_sec}s` : '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <Badge text={e.severity || 'Unknown'} color={SEV_COLORS[e.severity] || '#94a3b8'} />
                      </td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', color: e.injury && e.injury !== 'No' ? '#ef4444' : '#94a3b8' }}>{fmt(e.injury)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', color: e.er_visit === 'Yes' ? '#ef4444' : '#94a3b8' }}>{fmt(e.er_visit)}</td>
                      <td style={{ padding: '6px 10px', fontSize: 12 }}>{fmt(e.trigger)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(e.recovery_min)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(e.rescue_med)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* TRIGGER ANALYSIS TAB */}
      {tab === 'triggers' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Trigger-Severity Breakdown">
            {(bd.trigger_analysis || []).length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Trigger</th>
                      <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Total Events</th>
                      <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Severity Breakdown</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.trigger_analysis || []).map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{t.trigger}</td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}>{t.total}</td>
                        <td style={{ padding: '6px 10px' }}>
                          {Object.entries(t.severity || {}).map(([sev, cnt]) => (
                            <span key={sev} style={{ marginRight: 8 }}>
                              <Badge text={`${sev}: ${cnt}`} color={SEV_COLORS[sev] || '#94a3b8'} />
                            </span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No trigger data recorded yet</p>
            )}
          </Card>

          {/* Trigger frequency bar chart */}
          {triggerBarData.length > 0 && (
            <Card title="Trigger Frequency">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={triggerBarData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" name="Events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={defs.title || 'Definitions'}>
            <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>{defs.description}</p>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', width: 200 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Severity Levels">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', width: 120 }}>Level</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.severity_levels || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={s.level} color={SEV_COLORS[s.level] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Data Source">
            <p style={{ fontSize: 13, color: '#475569' }}>{defs.data_source}</p>
          </Card>
        </div>
      )}
    </div>
  )
}
