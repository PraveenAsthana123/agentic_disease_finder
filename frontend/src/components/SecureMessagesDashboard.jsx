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

function DirectionBadge({ direction }) {
  const d = String(direction || '').toLowerCase()
  const color = d === 'inbound' ? '#3b82f6' : '#10b981'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{d || '--'}</span>
  )
}

function PriorityBadge({ priority }) {
  const p = String(priority || '').toLowerCase()
  const color = p === 'urgent' ? '#ef4444' : p === 'high' ? '#f97316' : p === 'normal' ? '#3b82f6' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{p || '--'}</span>
  )
}

function ReadBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'read' ? '#10b981' : '#f59e0b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'messages', label: 'All Messages' },
  { id: 'patients', label: 'By Patient' },
  { id: 'categories', label: 'By Category' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SecureMessagesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/secure-messages/overview`),
      axios.get(`${API_URL}/api/secure-messages/breakdown`),
      axios.get(`${API_URL}/api/secure-messages/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading secure messages...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Secure Messages Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Patient-provider messaging analytics — {fmt(overview?.kpis?.total_messages)} messages, {fmt(overview?.kpis?.total_patients)} patients, {fmt(overview?.kpis?.unread_count)} unread
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
      {tab === 'messages' && <MessagesTab messages={breakdown?.messages || []} />}
      {tab === 'patients' && <PatientsTab data={breakdown?.by_patient || []} />}
      {tab === 'categories' && <CategoriesTab data={breakdown?.by_category || []} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const k = data.kpis || {}
  const directionData = (data.direction_dist || []).map(d => ({ name: d.direction, value: d.count }))
  const categoryData = (data.category_dist || []).map(d => ({ name: d.category, value: d.count }))
  const priorityData = (data.priority_dist || []).map(d => ({ name: d.priority, value: d.count }))
  const readData = (data.read_status_dist || []).map(d => ({ name: d.read_status, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
          <KPI label="Total Messages" value={k.total_messages} color="#3b82f6" />
          <KPI label="Patients" value={k.total_patients} color="#8b5cf6" />
          <KPI label="Inbound" value={k.inbound_count} color="#06b6d4" />
          <KPI label="Outbound" value={k.outbound_count} color="#10b981" />
          <KPI label="Unread" value={k.unread_count} sub={`${fmt(k.unread_rate)}% rate`} color="#f59e0b" />
          <KPI label="Avg Response" value={`${fmt(k.avg_response_time_hours)}h`} color="#ef4444" />
        </div>
      </Card>

      <Card title="Direction Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={directionData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {directionData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={categoryData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" name="Messages" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {categoryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Priority Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={priorityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {priorityData.map((d, i) => {
                const c = d.name === 'urgent' ? '#ef4444' : d.name === 'high' ? '#f97316' : d.name === 'normal' ? '#3b82f6' : '#64748b'
                return <Cell key={i} fill={c} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Read Status">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={readData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {readData.map((d, i) => {
                const c = d.name === 'read' ? '#10b981' : '#f59e0b'
                return <Cell key={i} fill={c} />
              })}
            </Pie>
            <Tooltip />
          </PieChart>
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
            <Line type="monotone" dataKey="messages" stroke="#3b82f6" name="Total Messages" strokeWidth={2} />
            <Line type="monotone" dataKey="inbound" stroke="#06b6d4" name="Inbound" strokeWidth={2} />
            <Line type="monotone" dataKey="outbound" stroke="#10b981" name="Outbound" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Avg Response Time by Priority">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.avg_response_by_priority || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="priority" tick={{ fontSize: 11 }} />
            <YAxis unit="h" />
            <Tooltip formatter={(v) => `${v}h`} />
            <Bar dataKey="avg_response_time" fill="#ef4444" name="Avg Response (hrs)" radius={[4, 4, 0, 0]}>
              {(data.avg_response_by_priority || []).map((d, i) => {
                const c = d.priority === 'urgent' ? '#ef4444' : d.priority === 'high' ? '#f97316' : d.priority === 'normal' ? '#3b82f6' : '#64748b'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category by Direction" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.category_by_direction || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" angle={-20} textAnchor="end" height={70} tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Bar dataKey="inbound" stackId="a" fill="#3b82f6" name="Inbound" />
            <Bar dataKey="outbound" stackId="a" fill="#10b981" name="Outbound" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function MessagesTab({ messages }) {
  const [sortKey, setSortKey] = useState('created_at')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = messages.filter(m => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (m.patient_id || '').toLowerCase().includes(f) ||
           (m.direction || '').toLowerCase().includes(f) ||
           (m.category || '').toLowerCase().includes(f) ||
           (m.subject || '').toLowerCase().includes(f) ||
           (m.priority || '').toLowerCase().includes(f)
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
    <Card title={`All Messages (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient, direction, category, subject, or priority..." value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Patient', 'patient_id')}
              {hdr('Direction', 'direction')}
              {hdr('Category', 'category')}
              {hdr('Subject', 'subject')}
              {hdr('Priority', 'priority')}
              {hdr('Read Status', 'read_status')}
              {hdr('Response (hrs)', 'response_time_hours')}
              {hdr('Created', 'created_at')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: m.priority === 'urgent' ? '#fef2f2' : undefined }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{m.patient_id}</td>
                <td style={{ padding: '6px 10px' }}><DirectionBadge direction={m.direction} /></td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize' }}>{(m.category || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.subject || '--'}</td>
                <td style={{ padding: '6px 10px' }}><PriorityBadge priority={m.priority} /></td>
                <td style={{ padding: '6px 10px' }}><ReadBadge status={m.read_status} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(m.response_time_hours)}</td>
                <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{m.created_at || '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} messages</div>}
    </Card>
  )
}

function PatientsTab({ data }) {
  if (!data || data.length === 0) return <Card>No patient data available.</Card>

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Patient Summary (${data.length} patients)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Messages</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Inbound</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Outbound</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Unread</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Response (hrs)</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Urgent</th>
              </tr>
            </thead>
            <tbody>
              {data.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{p.messages}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{p.inbound}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{p.outbound}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: p.unread > 0 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{p.unread}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(p.avg_response_time)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: p.urgent_count > 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>{p.urgent_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function CategoriesTab({ data }) {
  if (!data || data.length === 0) return <Card>No category data available.</Card>

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Category Breakdown (${data.length} categories)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Category</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Total</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Inbound</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Outbound</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Avg Response (hrs)</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Unread</th>
              </tr>
            </thead>
            <tbody>
              {data.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b', textTransform: 'capitalize' }}>{(c.category || '').replace(/_/g, ' ')}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{c.total}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{c.inbound}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{c.outbound}</td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{fmt(c.avg_response_time)}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', color: c.unread_count > 0 ? '#f59e0b' : '#64748b', fontWeight: 600 }}>{c.unread_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
