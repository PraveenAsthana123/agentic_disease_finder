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

function StageBadge({ stage }) {
  const s = String(stage || '').toLowerCase()
  const color = s === 'published' ? '#10b981' : s === 'approved' ? '#3b82f6'
    : s === 'created' ? '#f59e0b' : s === 'expired' ? '#ef4444' : s === 'archived' ? '#64748b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{stage || '--'}</span>
  )
}

function TypeBadge({ type }) {
  const t = String(type || '')
  const color = t === 'Assessment Instrument' ? '#8b5cf6' : t === 'Clinical Analysis' ? '#3b82f6'
    : t === 'EEG Upload' ? '#06b6d4' : t === 'Expert Review' ? '#10b981'
    : t === 'Imaging Finding' ? '#f59e0b' : t === 'Medication Record' ? '#ec4899'
    : t === 'Patient Diary' ? '#f97316' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{type || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'register', label: 'Knowledge Register' },
  { id: 'patients', label: 'By Patient' },
  { id: 'lifecycle', label: 'Lifecycle Events' },
  { id: 'definitions', label: 'Definitions' },
]

export default function KnowledgeMgmtDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/knowledge-mgmt/overview`),
      axios.get(`${API_URL}/api/knowledge-mgmt/breakdown`),
      axios.get(`${API_URL}/api/knowledge-mgmt/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading knowledge management data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Knowledge Management Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Clinical knowledge lifecycle analytics — {fmt(overview?.total_knowledge_items)} items, {fmt(overview?.patients_with_knowledge)} patients, {fmt(overview?.knowledge_types_count)} types, avg confidence {fmt(overview?.avg_confidence)}
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
      {tab === 'register' && <RegisterTab items={breakdown?.knowledge_register || []} />}
      {tab === 'patients' && <PatientsTab profiles={breakdown?.patient_profiles || []} />}
      {tab === 'lifecycle' && <LifecycleTab events={breakdown?.lifecycle_events || []} stageFlow={breakdown?.stage_flow || []} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const typeData = (data.type_distribution || []).map(d => ({ name: d.type, value: d.count }))
  const sourceData = (data.source_breakdown || []).map(d => ({ name: d.source.replace(/_/g, ' '), value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
          <KPI label="Total Items" value={data.total_knowledge_items} color="#3b82f6" />
          <KPI label="Published" value={data.published_count} color="#10b981" />
          <KPI label="Approval Rate" value={`${data.approval_rate_pct}%`} color="#8b5cf6" />
          <KPI label="Avg Confidence" value={data.avg_confidence} color="#06b6d4" />
          <KPI label="Patients" value={data.patients_with_knowledge} color="#f59e0b" />
          <KPI label="Lifecycle Events" value={data.total_lifecycle_events} color="#ec4899" />
        </div>
      </Card>

      <Card title="Type Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={typeData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={95}
              label={({ name, value }) => `${name} (${value})`} labelLine={false}>
              {typeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Stage Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={data.stage_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="stage" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Items" radius={[4, 4, 0, 0]}>
              {(data.stage_distribution || []).map((d, i) => {
                const c = d.stage === 'published' ? '#10b981' : d.stage === 'approved' ? '#3b82f6'
                  : d.stage === 'created' ? '#f59e0b' : d.stage === 'expired' ? '#ef4444' : '#64748b'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Source Breakdown" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={sourceData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" name="Items" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Activity Trend" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.activity_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="events" stroke="#3b82f6" name="Events" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function RegisterTab({ items }) {
  const [sortKey, setSortKey] = useState('created_at')
  const [sortDir, setSortDir] = useState(-1)
  const [filter, setFilter] = useState('')

  const filtered = items.filter(r => {
    if (!filter) return true
    const f = filter.toLowerCase()
    return (r.patient_id || '').toLowerCase().includes(f) ||
           (r.type || '').toLowerCase().includes(f) ||
           (r.title || '').toLowerCase().includes(f) ||
           (r.stage || '').toLowerCase().includes(f) ||
           (r.source_table || '').toLowerCase().includes(f)
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
    <Card title={`Knowledge Register (${filtered.length})`}>
      <input type="text" placeholder="Filter by patient, type, title, stage, or source..." value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8,
          fontSize: 13, marginBottom: 12, boxSizing: 'border-box' }} />
      <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('ID', 'id')}
              {hdr('Patient', 'patient_id')}
              {hdr('Type', 'type')}
              {hdr('Title', 'title')}
              {hdr('Stage', 'stage')}
              {hdr('Confidence', 'confidence')}
              {hdr('Source', 'source_table')}
              {hdr('Created', 'created_at')}
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 100).map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{r.id}</td>
                <td style={{ padding: '6px 10px' }}>{r.patient_id}</td>
                <td style={{ padding: '6px 10px' }}><TypeBadge type={r.type} /></td>
                <td style={{ padding: '6px 10px', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.title || '--'}</td>
                <td style={{ padding: '6px 10px' }}><StageBadge stage={r.stage} /></td>
                <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600,
                  color: r.confidence >= 0.8 ? '#10b981' : r.confidence >= 0.5 ? '#f59e0b' : '#ef4444'
                }}>{fmt(r.confidence)}</td>
                <td style={{ padding: '6px 10px', textTransform: 'capitalize', fontSize: 11 }}>{(r.source_table || '').replace(/_/g, ' ')}</td>
                <td style={{ padding: '6px 10px', whiteSpace: 'nowrap', fontSize: 11 }}>{r.created_at || '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 100 && <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing first 100 of {sorted.length} items</div>}
    </Card>
  )
}

function PatientsTab({ profiles }) {
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title={`Patient Knowledge Profiles (${profiles.length} patients)`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>Total Items</th>
                <th style={thStyle}>Types</th>
                <th style={thStyle}>Stages</th>
              </tr>
            </thead>
            <tbody>
              {profiles.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ ...tdStyle, textAlign: 'center', fontWeight: 600 }}>{p.total_items}</td>
                  <td style={tdStyle}>
                    {(p.types || []).map((t, j) => <TypeBadge key={j} type={t} />)}
                  </td>
                  <td style={tdStyle}>
                    {(p.stages || []).map((s, j) => <StageBadge key={j} stage={s} />)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function LifecycleTab({ events, stageFlow }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
      <Card title="Stage Flow Transitions" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={stageFlow} layout="vertical" margin={{ left: 160 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="transition" tick={{ fontSize: 11 }} width={150} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Transitions" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Lifecycle Events (${events.length})`} span={2}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0 }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Event ID</th>
                <th style={thStyle}>Patient</th>
                <th style={thStyle}>Component</th>
                <th style={thStyle}>Action</th>
                <th style={thStyle}>Stage</th>
                <th style={thStyle}>Actor</th>
                <th style={thStyle}>Detail</th>
                <th style={thStyle}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{e.event_id}</td>
                  <td style={tdStyle}>{e.patient_id}</td>
                  <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{(e.component || '').replace(/_/g, ' ')}</td>
                  <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{(e.action || '').replace(/_/g, ' ')}</td>
                  <td style={tdStyle}><StageBadge stage={e.stage} /></td>
                  <td style={{ ...tdStyle, fontSize: 11 }}>{e.actor || '--'}</td>
                  <td style={{ ...tdStyle, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: 11 }}>{e.detail || '--'}</td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap', fontSize: 11 }}>{e.timestamp || '--'}</td>
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
      {(data.concepts || []).length > 0 && (
        <Card title="Concepts" span={2}>
          {data.concepts.map((c, i) => (
            <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.concepts.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.term}</div>
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.definition}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.metrics || []).length > 0 && (
        <Card title="Metrics">
          {data.metrics.map((m, i) => (
            <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < data.metrics.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{m.name}</div>
              <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{m.description}</div>
            </div>
          ))}
        </Card>
      )}

      {(data.compliance || []).length > 0 && (
        <Card title="Compliance Requirements">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.compliance.map((item, i) => <li key={i} style={{ marginBottom: 4 }}>{item}</li>)}
          </ul>
        </Card>
      )}

      {(data.remediation || []).length > 0 && (
        <Card title="Remediation Steps">
          <ul style={{ margin: 0, padding: '0 0 0 20px', fontSize: 13, color: '#64748b' }}>
            {data.remediation.map((item, i) => <li key={i} style={{ marginBottom: 4 }}>{item}</li>)}
          </ul>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
