import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line,
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined,
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

const STATUS_COLOR = { uploaded: '#10b981', pending: '#f59e0b', error: '#ef4444' }
const COLORS = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#94a3b8']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sessions', label: 'Sessions' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EcgPatchDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [sessions, setSessions] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/ecg-patch/overview`),
      axios.get(`${API_URL}/api/ecg-patch/sessions`),
      axios.get(`${API_URL}/api/ecg-patch/definitions`),
    ]).then(([o, s, d]) => {
      setOverview(o.data)
      setSessions(s.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ECG Patch data…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>ECG Patch Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Offline-first cardiac monitor — continuous 14-day recording · Lead I/II/V5 ·
          HR, HRV, QTc, ST changes, arrhythmia detection · clinic upload workflow
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontWeight: tab === t.id ? 600 : 400,
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Patches" value={k.total_patches} /></Card>
            <Card><KPI label="Recording (Pending Upload)" value={k.active_recording} color="#f59e0b" /></Card>
            <Card><KPI label="Uploaded" value={k.uploaded} color="#10b981" /></Card>
            <Card><KPI label="Upload Errors" value={k.upload_errors} color="#ef4444" /></Card>
            <Card><KPI label="Avg Battery" value={k.avg_battery_pct != null ? `${k.avg_battery_pct}%` : '--'} /></Card>
            <Card><KPI label="Total Beats Analyzed" value={k.total_beats_analyzed?.toLocaleString()} sub="across fleet" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
            {/* HR Trend */}
            <Card title="24-Hour Heart Rate Profile (fleet mean)">
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={overview?.hr_trend_24h || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="hour" tickFormatter={h => `${h}:00`} tick={{ fontSize: 11 }} />
                  <YAxis domain={[40, 140]} tick={{ fontSize: 11 }} unit=" bpm" />
                  <Tooltip formatter={v => [`${v} bpm`, 'HR']} />
                  <Line type="monotone" dataKey="hr" stroke="#ef4444" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Event distribution */}
            <Card title="Cardiac Event Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview?.event_distribution || []} dataKey="count" nameKey="type"
                    cx="50%" cy="50%" outerRadius={80} label={({ type, percent }) =>
                      percent > 0.04 ? `${(percent * 100).toFixed(0)}%` : ''
                    }>
                    {(overview?.event_distribution || []).map((e, i) => (
                      <Cell key={i} fill={e.color || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v, n) => [v.toLocaleString(), n]} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* HRV Trend */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="7-Day HRV Trend (SDNN)">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={overview?.hrv_trend_7d || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 80]} tick={{ fontSize: 11 }} unit=" ms" />
                  <Tooltip formatter={v => [`${v} ms`, 'SDNN']} />
                  <Line type="monotone" dataKey="sdnn_ms" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            {/* Fleet battery */}
            <Card title="Patch Battery Levels">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={(overview?.fleet || []).slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="patch_id" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip formatter={v => [`${v}%`, 'Battery']} />
                  <Bar dataKey="battery_pct" fill="#3b82f6" radius={[3, 3, 0, 0]}>
                    {(overview?.fleet || []).slice(0, 10).map((p, i) => (
                      <Cell key={i} fill={p.battery_pct < 20 ? '#ef4444' : p.battery_pct < 40 ? '#f59e0b' : '#10b981'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Fleet table */}
          <Card title="Device Fleet">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Patch ID', 'Patient', 'Lead', 'Duration (d)', 'Battery %', 'Total Beats', 'Status'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview?.fleet || []).map((p, i) => (
                  <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{p.patch_id}</td>
                    <td style={{ padding: '8px 10px' }}>{p.patient}</td>
                    <td style={{ padding: '8px 10px' }}>{p.lead}</td>
                    <td style={{ padding: '8px 10px' }}>{p.duration_days}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{ color: p.battery_pct < 20 ? '#ef4444' : p.battery_pct < 40 ? '#f59e0b' : '#10b981' }}>
                        {p.battery_pct}%
                      </span>
                    </td>
                    <td style={{ padding: '8px 10px' }}>{p.total_beats?.toLocaleString()}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11,
                        background: STATUS_COLOR[p.upload_status] + '22',
                        color: STATUS_COLOR[p.upload_status] || '#64748b',
                      }}>{p.upload_status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}

      {/* ── SESSIONS TAB ── */}
      {tab === 'sessions' && sessions && (
        <>
          {/* Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
            <Card><KPI label="Total Sessions" value={sessions.summary?.total_sessions} /></Card>
            <Card><KPI label="Avg Duration" value={`${sessions.summary?.avg_duration_days}d`} /></Card>
            <Card><KPI label="Avg Signal Quality" value={`${sessions.summary?.avg_signal_quality_pct}%`} color="#10b981" /></Card>
            <Card><KPI label="Avg Arrhythmia Events" value={sessions.summary?.avg_arrhythmia_events} /></Card>
          </div>

          {/* Per-session HR chart */}
          {sessions.sessions?.length > 0 && (
            <Card title={`HR Trend — ${sessions.sessions[0].session_id} (${sessions.sessions[0].patient_id})`} span={2}>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={sessions.sessions[0].hr_series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="hour" tickFormatter={h => `${h}h`} tick={{ fontSize: 10 }} interval={5} />
                  <YAxis domain={[40, 140]} tick={{ fontSize: 11 }} unit=" bpm" />
                  <Tooltip formatter={v => [`${v} bpm`, 'HR']} />
                  <Line type="monotone" dataKey="hr" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Sessions table */}
          <Card title="Recording Sessions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Session', 'Patient', 'Lead', 'Days', 'Mean HR', 'SDNN', 'RMSSD', 'pNN50', 'QTc', 'Arrhythmias', 'Signal Q%', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(sessions.sessions || []).map((s, i) => (
                    <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '7px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.session_id}</td>
                      <td style={{ padding: '7px 10px' }}>{s.patient_id}</td>
                      <td style={{ padding: '7px 10px' }}>{s.lead}</td>
                      <td style={{ padding: '7px 10px' }}>{s.duration_days}</td>
                      <td style={{ padding: '7px 10px' }}>{s.mean_hr_bpm}</td>
                      <td style={{ padding: '7px 10px' }}>{s.sdnn_ms} ms</td>
                      <td style={{ padding: '7px 10px' }}>{s.rmssd_ms} ms</td>
                      <td style={{ padding: '7px 10px' }}>{s.pnn50_pct}%</td>
                      <td style={{ padding: '7px 10px', color: s.qtc_ms > 450 ? '#ef4444' : '#1e293b' }}>{s.qtc_ms} ms</td>
                      <td style={{ padding: '7px 10px' }}>{s.arrhythmia_events}</td>
                      <td style={{ padding: '7px 10px', color: s.signal_quality_pct < 90 ? '#f59e0b' : '#10b981' }}>{s.signal_quality_pct}%</td>
                      <td style={{ padding: '7px 10px' }}>
                        <span style={{
                          padding: '2px 7px', borderRadius: 10, fontSize: 10,
                          background: STATUS_COLOR[s.upload_status] + '22',
                          color: STATUS_COLOR[s.upload_status],
                        }}>{s.upload_status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && definitions && (
        <>
          {/* Device specs */}
          <Card title="Device Specification" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {Object.entries(definitions.device || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#f8fafc', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 13, color: '#1e293b', marginTop: 3, fontWeight: 500 }}>
                    {Array.isArray(v) ? v.join(', ') : String(v)}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Epilepsy context */}
          <Card title="Epilepsy Context">
            <p style={{ margin: 0, fontSize: 13, color: '#334155', lineHeight: 1.6 }}>
              {definitions.epilepsy_context}
            </p>
          </Card>

          {/* Metrics glossary */}
          <Card title="Metric Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Metric', 'Unit', 'Normal Range', 'Clinical Note'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(definitions.metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{m.term}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.unit}</td>
                    <td style={{ padding: '8px 12px', color: '#10b981', fontFamily: 'monospace' }}>{m.normal_range}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{m.clinical_note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* References */}
          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(definitions.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{r}</li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  )
}
