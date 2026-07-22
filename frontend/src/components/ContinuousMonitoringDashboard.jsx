import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
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

function SeverityBadge({ severity }) {
  const colors = {
    Mild: { bg: '#dcfce7', fg: '#166534', border: '#bbf7d0' },
    Severe: { bg: '#fecaca', fg: '#991b1b', border: '#fca5a5' }
  }
  const c = colors[severity] || { bg: '#f1f5f9', fg: '#475569', border: '#e2e8f0' }
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, color: c.fg, background: c.bg,
      padding: '2px 8px', borderRadius: 10, border: `1px solid ${c.border}`
    }}>{(severity || 'unknown').toUpperCase()}</span>
  )
}

export default function ContinuousMonitoringDashboard() {
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
          axios.get(`${API_URL}/api/continuous-monitoring/overview`),
          axios.get(`${API_URL}/api/continuous-monitoring/breakdown`),
          axios.get(`${API_URL}/api/continuous-monitoring/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load monitoring data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading continuous monitoring data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'trends', label: 'Trends & Triggers' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'recent', label: 'Recent Events' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const sevData = overview.severity_distribution
    ? Object.entries(overview.severity_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []
  const trigData = overview.trigger_distribution
    ? Object.entries(overview.trigger_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []
  const injData = overview.injury_distribution
    ? Object.entries(overview.injury_distribution).map(([k, v]) => ({ name: k, value: v }))
    : []

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Continuous Monitoring — Seizure Diary</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Real-time seizure event tracking from patient diaries ({overview.total_events} events, {overview.unique_patients} patients)
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card><KPI label="Total Events" value={overview.total_events} color="#3b82f6" /></Card>
          <Card><KPI label="Patients Monitored" value={overview.unique_patients} color="#10b981" /></Card>
          <Card><KPI label="Avg Events/Patient" value={overview.avg_events_per_patient} color="#8b5cf6" /></Card>
          <Card><KPI label="ER Visit Rate" value={`${overview.er_rate_pct}%`} sub={`${overview.er_visits} visits`} color={overview.er_rate_pct > 15 ? '#ef4444' : '#10b981'} /></Card>
          <Card><KPI label="Severe Rate" value={`${overview.severe_rate_pct}%`} sub={`${overview.severe_count} severe`} color={overview.severe_rate_pct > 30 ? '#ef4444' : '#f59e0b'} /></Card>
          <Card><KPI label="Avg Duration" value={`${overview.avg_duration_sec}s`} sub={`max ${overview.max_duration_sec}s`} color="#06b6d4" /></Card>
          <Card><KPI label="Avg Recovery" value={`${overview.avg_recovery_min} min`} color="#64748b" /></Card>
          <Card><KPI label="Aura Reported" value={overview.aura_reported} sub="focal onset indicator" color="#8b5cf6" /></Card>

          <Card title="Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {sevData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Injury Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={injData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="value" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Trends & Triggers ─── */}
      {tab === 'trends' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Daily Seizure Events" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown.daily_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={10} angle={-30} textAnchor="end" height={60} />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Trigger Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={trigData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {trigData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Severity by Trigger">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.trigger_severity}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="trigger" fontSize={10} />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Mild" fill="#10b981" stackId="a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Severe" fill="#ef4444" stackId="a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Duration Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.duration_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bucket" fontSize={11} />
                <YAxis fontSize={11} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Patient Profiles ─── */}
      {tab === 'patients' && breakdown && (
        <Card title={`Patient Seizure Profiles (${breakdown.patient_profiles.length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Patient</th>
                  <th style={{ padding: '8px 12px' }}>Events</th>
                  <th style={{ padding: '8px 12px' }}>Worst Severity</th>
                  <th style={{ padding: '8px 12px' }}>Avg Duration (s)</th>
                  <th style={{ padding: '8px 12px' }}>ER Visits</th>
                  <th style={{ padding: '8px 12px' }}>Injuries</th>
                  <th style={{ padding: '8px 12px' }}>First Event</th>
                  <th style={{ padding: '8px 12px' }}>Last Event</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.patient_profiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{p.total_events}</td>
                    <td style={{ padding: '8px 12px' }}><SeverityBadge severity={p.worst_severity} /></td>
                    <td style={{ padding: '8px 12px' }}>{fmt(p.avg_duration_sec)}</td>
                    <td style={{ padding: '8px 12px', color: p.er_visits > 0 ? '#ef4444' : '#64748b' }}>{p.er_visits}</td>
                    <td style={{ padding: '8px 12px', color: p.injuries > 0 ? '#f59e0b' : '#64748b' }}>{p.injuries}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{p.date_range?.first || '--'}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{p.date_range?.last || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── Recent Events ─── */}
      {tab === 'recent' && breakdown && (
        <Card title="Recent Seizure Events (last 20)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Patient</th>
                  <th style={{ padding: '8px 12px' }}>Date</th>
                  <th style={{ padding: '8px 12px' }}>Duration</th>
                  <th style={{ padding: '8px 12px' }}>Severity</th>
                  <th style={{ padding: '8px 12px' }}>Trigger</th>
                  <th style={{ padding: '8px 12px' }}>Injury</th>
                  <th style={{ padding: '8px 12px' }}>ER</th>
                  <th style={{ padding: '8px 12px' }}>Aura</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.recent_events.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{e.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{e.date || '--'}</td>
                    <td style={{ padding: '8px 12px' }}>{e.duration_sec ? `${e.duration_sec}s` : '--'}</td>
                    <td style={{ padding: '8px 12px' }}><SeverityBadge severity={e.severity} /></td>
                    <td style={{ padding: '8px 12px' }}>{e.trigger || '--'}</td>
                    <td style={{ padding: '8px 12px', color: e.injury && e.injury !== 'No' ? '#f59e0b' : '#64748b' }}>{e.injury || '--'}</td>
                    <td style={{ padding: '8px 12px', color: e.er_visit === 'Yes' ? '#ef4444' : '#64748b' }}>{e.er_visit || '--'}</td>
                    <td style={{ padding: '8px 12px' }}>{e.aura || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── Definitions ─── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          {defs.sections && defs.sections.map((s, si) => (
            <Card key={si} title={s.heading}>
              <div style={{ display: 'grid', gap: 8 }}>
                {s.items.map((d, di) => (
                  <div key={di} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.term}</div>
                    <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>{d.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
