import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, bg, fg }) {
  return (
    <span style={{
      background: bg, color: fg, padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600
    }}>{text}</span>
  )
}

function EventBadge({ event }) {
  const map = {
    DOWN: { bg: '#fee2e2', fg: '#991b1b' },
    'AUTO-RESTARTED': { bg: '#dcfce7', fg: '#166534' },
    UP: { bg: '#dcfce7', fg: '#166534' },
  }
  const s = map[event] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={event} bg={s.bg} fg={s.fg} />
}

function LevelBadge({ level }) {
  const map = {
    ops: { bg: '#dbeafe', fg: '#1e40af' },
    autobuild: { bg: '#fef3c7', fg: '#92400e' },
    error: { bg: '#fee2e2', fg: '#991b1b' },
    info: { bg: '#f1f5f9', fg: '#475569' },
  }
  const s = map[level] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={level} bg={s.bg} fg={s.fg} />
}

function StatusIndicator({ status }) {
  const isUp = status === 'UP'
  return <Badge text={status} bg={isUp ? '#dcfce7' : '#fee2e2'} fg={isUp ? '#166534' : '#991b1b'} />
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v <= 1 ? (v * 100).toFixed(1) + '%' : v.toFixed(1) + '%') : String(v)
}

const TABS = ['Overview', 'Timeline', 'Heatmap & Types', 'Recovery', 'Reference']

export default function IncidentManagementDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/incident-management/overview`),
      axios.get(`${API}/incident-management/breakdown`),
      axios.get(`${API}/incident-management/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading incident management data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>
        Incident Management Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 20px' }}>
        Service incidents, auto-recovery tracking, MTTR analysis, and operational heatmaps
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#1e293b' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: 600, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} />}
      {tab === 1 && <TimelineTab ov={ov} bd={bd} />}
      {tab === 2 && <HeatmapTypesTab bd={bd} />}
      {tab === 3 && <RecoveryTab bd={bd} />}
      {tab === 4 && <ReferenceTab df={df} />}
    </div>
  )
}

/* ── Tab 1: Overview ── */
function OverviewTab({ ov }) {
  if (!ov) return null
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Incident KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Total Incidents" value={fmt(ov.total_incidents)} color="#ef4444" />
          <KPI label="Auto Recovery Rate" value={fmtPct(ov.auto_recovery_rate)} color="#10b981" />
          <div style={{ textAlign: 'center' }}>
            <div style={{ marginBottom: 4 }}><StatusIndicator status={ov.current_status || '--'} /></div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>Current Status</div>
          </div>
          <KPI label="MTTR (minutes)" value={fmt(ov.mean_time_to_recovery_minutes)} color="#3b82f6" />
        </div>
      </Card>

      <Card title="Daily Incident Counts" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={ov.daily_incident_counts || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} name="Incidents" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={ov.severity_distribution || []} dataKey="count" nameKey="level"
              cx="50%" cy="50%" outerRadius={90}
              label={({ level, count }) => `${level}: ${count}`}>
              {(ov.severity_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Current Status Summary" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '12px 0' }}>
          <KPI label="Total Recoveries" value={fmt(ov.total_recoveries)} color="#10b981" />
          <KPI label="Incidents Last 24h" value={fmt(ov.incidents_last_24h)} color="#f59e0b" />
          <KPI label="Incidents Last 7d" value={fmt(ov.incidents_last_7d)} color="#8b5cf6" />
          <KPI label="Auto Recovery %" value={fmtPct(ov.auto_recovery_rate)} color="#06b6d4" />
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 2: Timeline ── */
function TimelineTab({ ov, bd }) {
  if (!ov) return null
  const timeline = ov.incident_timeline || []
  const mttrTrend = bd?.mttr_trend || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Timeline KPIs" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <KPI label="Last 24h" value={fmt(ov.incidents_last_24h)} color="#f59e0b" />
          <KPI label="Last 7d" value={fmt(ov.incidents_last_7d)} color="#8b5cf6" />
          <KPI label="Total Recoveries" value={fmt(ov.total_recoveries)} color="#10b981" />
          <KPI label="MTTR (min)" value={fmt(ov.mean_time_to_recovery_minutes)} color="#3b82f6" />
        </div>
      </Card>

      <Card title={`Incident Timeline (${timeline.length} events)`} span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Timestamp', 'Event', 'HTTP Code', 'Recovery Time', 'TTR (min)'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {timeline.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{t.ts || '--'}</td>
                  <td style={{ padding: '8px 10px' }}><EventBadge event={t.event} /></td>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{t.http || '--'}</td>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{t.recovery_ts || '--'}</td>
                  <td style={{ padding: '8px 10px' }}>{t.ttr_minutes != null ? t.ttr_minutes.toFixed(1) : '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="MTTR Trend Over Time" span={4}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={mttrTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'MTTR (min)', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip />
            <Line type="monotone" dataKey="avg_mttr_minutes" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg MTTR (min)" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ── Tab 3: Heatmap & Types ── */
function HeatmapTypesTab({ bd }) {
  if (!bd) return null
  const heatmap = bd.hourly_heatmap || {}
  const dayLabels = heatmap.day_labels || []
  const hourLabels = heatmap.hour_labels || []
  const matrix = heatmap.matrix || []
  const topTypes = bd.top_incident_types || []
  const trackByLevel = bd.track_events_by_level || []

  function heatColor(val) {
    if (!val || val === 0) return 'transparent'
    const maxVal = Math.max(...matrix.flat(), 1)
    const intensity = Math.min(val / maxVal, 1)
    const alpha = 0.15 + intensity * 0.85
    return `rgba(239, 68, 68, ${alpha})`
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Incident Heatmap (Day x Hour)" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
            <thead>
              <tr>
                <th style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>Day</th>
                {hourLabels.map((h, i) => (
                  <th key={i} style={{ padding: '4px 6px', textAlign: 'center', color: '#64748b', fontWeight: 600, minWidth: 28 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dayLabels.map((day, di) => (
                <tr key={di}>
                  <td style={{ padding: '4px 8px', fontWeight: 600, whiteSpace: 'nowrap', color: '#334155' }}>{day}</td>
                  {(matrix[di] || []).map((val, hi) => (
                    <td key={hi} style={{
                      padding: '4px 6px', textAlign: 'center', fontWeight: val > 0 ? 600 : 400,
                      background: heatColor(val), color: val > 0 ? '#991b1b' : '#94a3b8',
                      borderRadius: 4, minWidth: 28
                    }}>
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Top Incident Types">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={topTypes}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="type" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} name="Count" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Track Events by Level">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={trackByLevel} dataKey="count" nameKey="level"
              cx="50%" cy="50%" outerRadius={90}
              label={({ level, count }) => `${level}: ${count}`}>
              {trackByLevel.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ── Tab 4: Recovery ── */
function RecoveryTab({ bd }) {
  if (!bd) return null
  const recoveryByDay = bd.recovery_success_rate_by_day || []
  const recentEvents = bd.recent_track_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Recovery Success Rate by Day">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={recoveryByDay}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} label={{ value: 'Rate %', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
            <Tooltip formatter={(v, name) => name === 'rate' ? `${v}%` : v} />
            <Legend />
            <Bar dataKey="incidents" fill="#ef4444" radius={[4, 4, 0, 0]} name="Incidents" />
            <Bar dataKey="recovered" fill="#10b981" radius={[4, 4, 0, 0]} name="Recovered" />
            <Bar dataKey="rate" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Rate %" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Recent Track Events (${recentEvents.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Timestamp', 'Level', 'Event', 'Host'].map(h =>
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {recentEvents.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>{e.ts || '--'}</td>
                  <td style={{ padding: '8px 10px' }}><LevelBadge level={e.level} /></td>
                  <td style={{ padding: '8px 10px', color: '#475569' }}>{e.event || '--'}</td>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{e.host || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Tab 5: Reference ── */
function ReferenceTab({ df }) {
  if (!df) return null
  const metrics = df.metrics || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Metric Definitions">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              {['Term', 'Definition'].map(h =>
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>{m.term}</td>
                <td style={{ padding: '8px 10px', color: '#475569' }}>{m.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
