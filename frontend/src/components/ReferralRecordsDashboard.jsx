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

function UrgencyBadge({ urgency }) {
  const u = String(urgency || '')
  const color = u === 'emergent' ? '#dc2626' : u === 'urgent' ? '#ef4444' : u === 'elective' ? '#3b82f6' : '#10b981'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{u || '--'}</span>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '').replace(/_/g, ' ')
  const color = s === 'completed' ? '#10b981' : s === 'scheduled' ? '#3b82f6'
    : s === 'triaged' ? '#8b5cf6' : s === 'in progress' ? '#06b6d4'
    : s === 'pending triage' ? '#f59e0b' : s === 'cancelled' ? '#ef4444' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'referrals', label: 'All Referrals' },
  { id: 'patients', label: 'By Patient' },
  { id: 'sources', label: 'By Source' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ReferralRecordsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/referral-records/overview`),
      axios.get(`${API_URL}/api/referral-records/breakdown`),
      axios.get(`${API_URL}/api/referral-records/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading referral records…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Referral Records Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Patient referral triage analytics — {fmt(overview?.total_referrals)} referrals, {fmt(overview?.total_patients)} patients, avg triage score {fmt(overview?.avg_triage_score)}
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
      {tab === 'referrals' && <ReferralsTab referrals={breakdown?.referrals || []} />}
      {tab === 'patients' && <PatientsTab data={breakdown} />}
      {tab === 'sources' && <SourcesTab data={breakdown} overview={overview} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const sourceData = (data.source_distribution || []).map(d => ({ name: d.source.replace(/_/g, ' '), value: d.count }))
  const reasonData = (data.reason_distribution || []).map(d => ({ name: d.reason.replace(/_/g, ' '), value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
          <KPI label="Total Referrals" value={data.total_referrals} color="#3b82f6" />
          <KPI label="Patients" value={data.total_patients} color="#8b5cf6" />
          <KPI label="Avg Triage Score" value={data.avg_triage_score} color="#06b6d4" />
          <KPI label="Urgent + Emergent" value={data.urgent_emergent_count} color="#ef4444" />
          <KPI label="Completion Rate" value={`${data.completion_rate}%`} color="#10b981" />
          <KPI label="Pending Triage" value={data.pending_count} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Referral Source Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={sourceData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {sourceData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Urgency Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.urgency_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="urgency" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Referrals" radius={[4, 4, 0, 0]}>
              {(data.urgency_distribution || []).map((d, i) => {
                const c = d.urgency === 'emergent' ? '#dc2626' : d.urgency === 'urgent' ? '#ef4444'
                  : d.urgency === 'elective' ? '#3b82f6' : '#10b981'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Triage Status Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.triage_status_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="status" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Referrals" radius={[4, 4, 0, 0]}>
              {(data.triage_status_distribution || []).map((d, i) => {
                const s = d.status
                const c = s === 'completed' ? '#10b981' : s === 'scheduled' ? '#3b82f6'
                  : s === 'triaged' ? '#8b5cf6' : s === 'in_progress' ? '#06b6d4'
                  : s === 'pending_triage' ? '#f59e0b' : s === 'cancelled' ? '#ef4444' : '#64748b'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Referral Reason Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={reasonData} layout="vertical" margin={{ left: 160 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Referrals" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Referral Trend" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="total" stroke="#3b82f6" name="Total" strokeWidth={2} />
            <Line type="monotone" dataKey="urgent_emergent" stroke="#ef4444" name="Urgent/Emergent" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Assigned Specialist Workload">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.assigned_to_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="assigned_to" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={80} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4" name="Assigned" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Avg Triage Score by Urgency">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.avg_triage_score_by_urgency || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="urgency" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="avg_score" fill="#f59e0b" name="Avg Score" radius={[4, 4, 0, 0]}>
              {(data.avg_triage_score_by_urgency || []).map((d, i) => {
                const c = d.urgency === 'emergent' ? '#dc2626' : d.urgency === 'urgent' ? '#ef4444'
                  : d.urgency === 'elective' ? '#3b82f6' : '#10b981'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ReferralsTab({ referrals }) {
  const [sortKey, setSortKey] = useState('referral_date')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = referrals.filter(r => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (r.patient_id || '').toLowerCase().includes(f) ||
           (r.referral_source || '').toLowerCase().includes(f) ||
           (r.referral_reason || '').toLowerCase().includes(f) ||
           (r.assigned_to || '').toLowerCase().includes(f)
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
    <Card title={`All Referrals (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient, source, reason, or specialist…" value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Patient', 'patient_id')}
              {hdr('Date', 'referral_date')}
              {hdr('Source', 'referral_source')}
              {hdr('Reason', 'referral_reason')}
              {hdr('Urgency', 'urgency')}
              {hdr('Status', 'triage_status')}
              {hdr('Score', 'triage_score')}
              {hdr('Assigned To', 'assigned_to')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9',
                background: r.urgency === 'emergent' ? '#fef2f2' : r.urgency === 'urgent' ? '#fffbeb' : undefined }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{r.patient_id}</td>
                <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{r.referral_date || '--'}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(r.referral_source || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(r.referral_reason || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px' }}><UrgencyBadge urgency={r.urgency} /></td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={r.triage_status} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600,
                  color: r.triage_score >= 70 ? '#ef4444' : r.triage_score >= 40 ? '#f59e0b' : '#10b981'
                }}>{fmt(r.triage_score)}</td>
                <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.assigned_to || '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} referrals</div>}
    </Card>
  )
}

function PatientsTab({ data }) {
  if (!data) return null
  const patients = data.patient_summary || []

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Patient Referral Summary (${patients.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Referrals</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Score</th>
                <th style={thStyle}>Latest Date</th>
                <th style={thStyle}>Top Source</th>
                <th style={thStyle}>Top Urgency</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{p.total_referrals}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600,
                    color: p.avg_score >= 70 ? '#ef4444' : p.avg_score >= 40 ? '#f59e0b' : '#10b981'
                  }}>{fmt(p.avg_score)}</td>
                  <td style={tdStyle}>{p.latest_date}</td>
                  <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{(p.top_source || '').replace(/_/g, ' ')}</td>
                  <td style={tdStyle}><UrgencyBadge urgency={p.top_urgency} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function SourcesTab({ data, overview }) {
  if (!data) return null
  const sources = data.by_source || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title="Referral Sources Analysis" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Source</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Count</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Score</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Completion Rate</th>
                <th style={thStyle}>Top Reason</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize' }}>{(s.source || '').replace(/_/g, ' ')}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{s.count}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600,
                    color: s.avg_score >= 70 ? '#ef4444' : s.avg_score >= 40 ? '#f59e0b' : '#10b981'
                  }}>{fmt(s.avg_score)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    <span style={{ color: s.completion_rate >= 50 ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                      {fmt(s.completion_rate)}%
                    </span>
                  </td>
                  <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{(s.top_reason || '').replace(/_/g, ' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Avg Triage Score by Source">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={overview?.avg_triage_score_by_source || []} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="source" tick={{ fontSize: 11 }} width={110}
              tickFormatter={v => v.replace(/_/g, ' ')} />
            <Tooltip />
            <Bar dataKey="avg_score" fill="#3b82f6" name="Avg Score" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
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

      {(data.urgency_levels || []).length > 0 && (
        <Card title="Urgency Levels">
          {data.urgency_levels.map((u, i) => (
            <div key={i} style={{ marginBottom: 10, fontSize: 13 }}>
              <strong style={{ textTransform: 'capitalize' }}><UrgencyBadge urgency={u.level} /></strong>
              <div style={{ color: '#64748b', marginTop: 4 }}>{u.description}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.triage_statuses || []).length > 0 && (
        <Card title="Triage Statuses">
          {data.triage_statuses.map((s, i) => (
            <div key={i} style={{ marginBottom: 10, fontSize: 13 }}>
              <StatusBadge status={s.status} />
              <div style={{ color: '#64748b', marginTop: 4 }}>{s.description}</div>
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
