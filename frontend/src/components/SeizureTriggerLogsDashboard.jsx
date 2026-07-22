import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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

function SleepBadge({ quality }) {
  const q = String(quality || '')
  const color = q === 'good' ? '#10b981' : q === 'fair' ? '#f59e0b' : q === 'poor' ? '#ef4444' : '#dc2626'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{q.replace(/_/g, ' ') || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'logs', label: 'All Logs' },
  { id: 'patients', label: 'By Patient' },
  { id: 'risk', label: 'Risk Analysis' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SeizureTriggerLogsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/seizure-trigger-logs/overview`),
      axios.get(`${API_URL}/api/seizure-trigger-logs/breakdown`),
      axios.get(`${API_URL}/api/seizure-trigger-logs/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading seizure trigger logs…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Seizure Trigger Logs Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Daily seizure diary analytics — {fmt(overview?.total_logs)} logs, {fmt(overview?.total_patients)} patients, {fmt(overview?.seizure_count)} seizure events
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'logs' && <LogsTab logs={breakdown?.logs || []} />}
      {tab === 'patients' && <PatientsTab data={breakdown} />}
      {tab === 'risk' && <RiskTab data={overview} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const triggerData = (data.trigger_distribution || []).map(d => ({ name: d.trigger.replace(/_/g, ' '), value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
          <KPI label="Total Logs" value={data.total_logs} color="#3b82f6" />
          <KPI label="Patients" value={data.total_patients} color="#8b5cf6" />
          <KPI label="Seizure Rate" value={`${data.seizure_rate}%`} sub={`${data.seizure_count} events`} color="#ef4444" />
          <KPI label="Avg Sleep" value={`${data.avg_sleep_hours}h`} color="#06b6d4" />
          <KPI label="Med Adherence" value={`${data.medication_adherence_rate}%`} color="#10b981" />
        </div>
      </Card>

      <Card title="Trigger Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={triggerData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {triggerData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Sleep Quality Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.sleep_quality_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="quality" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Logs" radius={[4, 4, 0, 0]}>
              {(data.sleep_quality_distribution || []).map((d, i) => {
                const c = d.quality === 'good' ? '#10b981' : d.quality === 'fair' ? '#f59e0b' : d.quality === 'poor' ? '#ef4444' : '#dc2626'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Seizure Type Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.seizure_type_distribution || []} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={130} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Seizures" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Trend" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="total_logs" stroke="#3b82f6" name="Total Logs" strokeWidth={2} />
            <Line type="monotone" dataKey="seizures" stroke="#ef4444" name="Seizures" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Seizure Rate by Trigger" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={(data.trigger_seizure_rate || []).map(d => ({ ...d, trigger: d.trigger.replace(/_/g, ' ') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="trigger" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis unit="%" />
            <Tooltip formatter={(v, name) => name === 'Seizure Rate' ? `${v}%` : v} />
            <Bar dataKey="rate" fill="#ef4444" name="Seizure Rate" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function LogsTab({ logs }) {
  const [sortKey, setSortKey] = useState('log_date')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = logs.filter(l => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (l.patient_id || '').toLowerCase().includes(f) ||
           (l.primary_trigger || '').toLowerCase().includes(f) ||
           (l.seizure_type || '').toLowerCase().includes(f)
  })

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir
  })

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d * -1)
    else { setSortKey(key); setSortDir(-1) }
  }

  const hdr = (label, key) => (
    <th onClick={() => toggleSort(key)} style={{
      padding: '8px 10px', cursor: 'pointer', whiteSpace: 'nowrap', fontSize: 12,
      background: '#f8fafc', borderBottom: '2px solid #e2e8f0', textAlign: 'left',
      color: sortKey === key ? '#3b82f6' : '#475569'
    }}>{label} {sortKey === key ? (sortDir > 0 ? '▲' : '▼') : ''}</th>
  )

  return (
    <Card title={`All Trigger Logs (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient, trigger, or seizure type…" value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Patient', 'patient_id')}
              {hdr('Date', 'log_date')}
              {hdr('Sleep', 'sleep_hours')}
              {hdr('Quality', 'sleep_quality')}
              {hdr('Stress', 'stress_level')}
              {hdr('Trigger', 'primary_trigger')}
              {hdr('Seizure', 'seizure_occurred')}
              {hdr('Type', 'seizure_type')}
              {hdr('Duration', 'seizure_duration_sec')}
              {hdr('Med Adh.', 'medication_adherence')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((l, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: l.seizure_occurred ? '#fef2f2' : undefined }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{l.patient_id}</td>
                <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{l.log_date}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(l.sleep_hours)}h</td>
                <td style={{ padding: '6px 10px' }}><SleepBadge quality={l.sleep_quality} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                  <span style={{ color: l.stress_level >= 7 ? '#ef4444' : l.stress_level >= 4 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                    {l.stress_level}/10
                  </span>
                </td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(l.primary_trigger || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                  {l.seizure_occurred
                    ? <span style={{ color: '#ef4444', fontWeight: 700 }}>Yes</span>
                    : <span style={{ color: '#10b981' }}>No</span>}
                </td>
                <td style={{ padding: '6px 10px', fontSize: 11 }}>{l.seizure_type || '--'}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{l.seizure_duration_sec ? `${l.seizure_duration_sec}s` : '--'}</td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                  {l.medication_adherence
                    ? <span style={{ color: '#10b981', fontWeight: 600 }}>✓</span>
                    : <span style={{ color: '#ef4444', fontWeight: 600 }}>✗</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} logs</div>}
    </Card>
  )
}

function PatientsTab({ data }) {
  if (!data) return null
  const patients = data.patient_summary || []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Patient Summary (${patients.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Logs</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Seizures</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Seizure Rate</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Sleep</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Stress</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Adherence</th>
                <th style={thStyle}>Latest Log</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => {
                const rate = p.total_logs > 0 ? ((p.seizures / p.total_logs) * 100).toFixed(0) : 0
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>{p.total_logs}</td>
                    <td style={{ ...tdStyle, textAlign: 'center', color: p.seizures > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{p.seizures}</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>{rate}%</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.avg_sleep)}h</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.avg_stress)}/10</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>
                      <span style={{ color: p.adherence_pct >= 90 ? '#10b981' : p.adherence_pct >= 70 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                        {fmt(p.adherence_pct)}%
                      </span>
                    </td>
                    <td style={tdStyle}>{p.latest_log}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {(data.top_patient_triggers || []).length > 0 && (
        <Card title="Top Patient-Trigger Combinations (seizure events)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Patient</th>
                  <th style={{ ...thStyle, color: '#dc2626' }}>Trigger</th>
                  <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Seizure Events</th>
                </tr>
              </thead>
              <tbody>
                {data.top_patient_triggers.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ ...tdStyle, fontWeight: 500 }}>{r.patient_id}</td>
                    <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{(r.primary_trigger || '').replace(/_/g, ' ')}</td>
                    <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600, color: '#ef4444' }}>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function RiskTab({ data }) {
  if (!data) return null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title="Risk Factor Comparison: Seizure vs No-Seizure Days" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data.risk_comparison || []} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="factor" tick={{ fontSize: 12 }} width={110} />
            <Tooltip />
            <Legend />
            <Bar dataKey="with_seizure" fill="#ef4444" name="With Seizure" />
            <Bar dataKey="without_seizure" fill="#10b981" name="Without Seizure" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Seizure Rate by Sleep Quality">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={(data.sleep_vs_seizure || []).map(d => ({ ...d, quality: d.quality.replace(/_/g, ' ') }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="quality" />
            <YAxis unit="%" />
            <Tooltip formatter={(v, name) => name === 'Seizure Rate' ? `${v}%` : v} />
            <Bar dataKey="rate" fill="#ef4444" name="Seizure Rate" radius={[4, 4, 0, 0]}>
              {(data.sleep_vs_seizure || []).map((d, i) => {
                const c = d.quality === 'good' ? '#10b981' : d.quality === 'fair' ? '#f59e0b' : d.quality === 'poor' ? '#ef4444' : '#dc2626'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Risk Factor Detail" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={thStyle}>Factor</th>
              <th style={{ ...thStyle, textAlign: 'center', color: '#ef4444' }}>Avg on Seizure Days</th>
              <th style={{ ...thStyle, textAlign: 'center', color: '#10b981' }}>Avg on Non-Seizure Days</th>
              <th style={{ ...thStyle, textAlign: 'center' }}>Difference</th>
            </tr>
          </thead>
          <tbody>
            {(data.risk_comparison || []).map((r, i) => {
              const diff = r.with_seizure != null && r.without_seizure != null
                ? (r.with_seizure - r.without_seizure).toFixed(1) : '--'
              const diffNum = parseFloat(diff)
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{r.factor}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{fmt(r.with_seizure)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{fmt(r.without_seizure)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600,
                    color: isNaN(diffNum) ? '#64748b' : diffNum > 0 ? '#ef4444' : diffNum < 0 ? '#10b981' : '#64748b'
                  }}>
                    {isNaN(diffNum) ? '--' : (diffNum > 0 ? '+' : '') + diff}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return <Card>No definitions available.</Card>
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>
      <Card title={data.title || 'Definitions'} span={2}>
        {(data.concepts || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
            <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.name}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</div>
          </div>
        ))}
      </Card>

      {(data.seizure_types || []).length > 0 && (
        <Card title="Seizure Types">
          {data.seizure_types.map((st, i) => (
            <div key={i} style={{ marginBottom: 10, fontSize: 13 }}>
              <strong style={{ color: '#8b5cf6', textTransform: 'capitalize' }}>{st.type}</strong>
              <div style={{ color: '#64748b', marginTop: 2 }}>{st.description}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.data_sources.map((src, i) => <li key={i} style={{ marginBottom: 4 }}>{src}</li>)}
          </ul>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
