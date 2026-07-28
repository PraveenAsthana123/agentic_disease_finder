import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
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

function RelBadge({ rel }) {
  const r = String(rel || '').toLowerCase().replace('_', ' ')
  const map = {
    spouse: '#3b82f6', parent: '#10b981', sibling: '#f59e0b', child: '#8b5cf6',
    partner: '#ec4899', friend: '#06b6d4', neighbor: '#64748b',
    grandparent: '#f97316', 'professional caregiver': '#ef4444'
  }
  const color = map[r] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{r || '--'}</span>
  )
}

function FreshnessBadge({ freshness }) {
  const f = String(freshness || '').toLowerCase()
  const color = f === 'current' ? '#10b981' : f === 'due-soon' ? '#f59e0b' : '#ef4444'
  const label = f === 'current' ? 'Current' : f === 'due-soon' ? 'Due Soon' : 'Overdue'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{label}</span>
  )
}

function YesNoBadge({ val }) {
  const yes = val === 1 || val === true
  const color = yes ? '#10b981' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{yes ? 'Yes' : 'No'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'contacts', label: 'All Contacts' },
  { id: 'relationship', label: 'By Relationship' },
  { id: 'patient', label: 'By Patient' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EmergencyContactsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')
  const [filter, setFilter] = useState('')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/emergency-contacts/overview`),
      axios.get(`${API_URL}/api/emergency-contacts/breakdown`),
      axios.get(`${API_URL}/api/emergency-contacts/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading emergency contacts...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const sorted = (rows) => {
    if (!sortCol || !rows) return rows
    return [...rows].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol]
      if (av == null && bv == null) return 0
      if (av == null) return 1
      if (bv == null) return -1
      if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av - bv : bv - av
      return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })
  }

  const doSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('asc') }
  }

  const hdr = (col, label) => (
    <th style={{ padding: '8px 10px', cursor: 'pointer', userSelect: 'none', fontSize: 12 }}
      onClick={() => doSort(col)}>
      {label} {sortCol === col ? (sortDir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )

  const filterRows = (rows) => {
    if (!filter) return rows
    const f = filter.toLowerCase()
    return rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(f)))
  }

  const kpis = overview?.kpis || {}
  const relDist = (overview?.relationship_distribution || []).map(r => ({
    name: (r.relationship || '').replace('_', ' '),
    value: r.cnt
  }))
  const notifyByRel = (overview?.notify_by_relationship || []).map(r => ({
    name: (r.relationship || '').replace('_', ' '),
    total: r.total,
    notify_enabled: r.notify_enabled,
    not_enabled: r.total - r.notify_enabled
  }))
  const freshness = (overview?.verification_freshness || []).map(r => ({
    name: r.freshness === 'current' ? 'Current' : r.freshness === 'due-soon' ? 'Due Soon' : 'Overdue',
    value: r.cnt
  }))
  const freshnessColors = { 'Current': '#10b981', 'Due Soon': '#f59e0b', 'Overdue': '#ef4444' }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Emergency Contacts Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Emergency contact registry — {fmt(kpis.total_contacts)} contacts, {fmt(kpis.patients_covered)} patients covered, {fmt(kpis.seizure_notify_pct)}% seizure notification enabled
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => { setTab(t.id); setSortCol(null); setFilter('') }}
            style={{
              padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
              background: tab === t.id ? '#3b82f6' : 'transparent',
              color: tab === t.id ? '#fff' : '#64748b',
              fontWeight: tab === t.id ? 700 : 500, fontSize: 13
            }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview ─── */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Contacts" value={kpis.total_contacts} color="#3b82f6" />
              <KPI label="Patients Covered" value={kpis.patients_covered} color="#10b981" />
              <KPI label="Primary %" value={kpis.primary_pct} sub="of contacts" color="#8b5cf6" />
              <KPI label="Seizure Notify %" value={kpis.seizure_notify_pct} sub="auto-alert enabled" color="#f59e0b" />
              <KPI label="Avg Days Since Verified" value={kpis.avg_days_since_verified} sub="freshness" color="#ef4444" />
            </div>
          </Card>

          <Card title="Relationship Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={relDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {relDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Verification Freshness">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={freshness} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {freshness.map((e, i) => <Cell key={i} fill={freshnessColors[e.name] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Seizure Notification by Relationship" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={notifyByRel} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="notify_enabled" name="Notify Enabled" fill="#10b981" stackId="a" />
                <Bar dataKey="not_enabled" name="Not Enabled" fill="#ef4444" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── All Contacts ─── */}
      {tab === 'contacts' && breakdown && (
        <Card>
          <div style={{ marginBottom: 12 }}>
            <input placeholder="Filter contacts..." value={filter} onChange={e => setFilter(e.target.value)}
              style={{ padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: 8, width: 260, fontSize: 13 }} />
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  {hdr('patient_id', 'Patient')}
                  {hdr('contact_name', 'Contact Name')}
                  {hdr('relationship', 'Relationship')}
                  {hdr('phone', 'Phone')}
                  {hdr('email', 'Email')}
                  {hdr('is_primary', 'Primary')}
                  {hdr('notify_on_seizure', 'Seizure Notify')}
                  {hdr('last_verified', 'Last Verified')}
                  {hdr('days_since_verified', 'Days Since')}
                </tr>
              </thead>
              <tbody>
                {sorted(filterRows(breakdown.all_contacts || [])).map((r, i) => (
                  <tr key={r.id || i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{r.contact_name}</td>
                    <td style={{ padding: '6px 10px' }}><RelBadge rel={r.relationship} /></td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{r.phone}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{r.email}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><YesNoBadge val={r.is_primary} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><YesNoBadge val={r.notify_on_seizure} /></td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{r.last_verified}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                      <span style={{ color: r.days_since_verified > 180 ? '#ef4444' : r.days_since_verified > 90 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {fmt(r.days_since_verified)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── By Relationship ─── */}
      {tab === 'relationship' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Relationship Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  {hdr('relationship', 'Relationship')}
                  {hdr('total', 'Total')}
                  {hdr('primary_count', 'Primary')}
                  {hdr('notify_count', 'Notify Enabled')}
                  {hdr('avg_days_since_verified', 'Avg Days Since Verified')}
                </tr>
              </thead>
              <tbody>
                {sorted(breakdown.by_relationship || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><RelBadge rel={r.relationship} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{r.total}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.primary_count}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.notify_count}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                      <span style={{ color: r.avg_days_since_verified > 180 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>
                        {fmt(r.avg_days_since_verified)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Contacts by Relationship">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={(breakdown.by_relationship || []).map(r => ({ name: (r.relationship || '').replace('_', ' '), total: r.total }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-25} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="total" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Notify Rate by Relationship">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={(breakdown.by_relationship || []).map(r => ({
                name: (r.relationship || '').replace('_', ' '),
                rate: r.total > 0 ? Math.round(r.notify_count / r.total * 100) : 0
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-25} textAnchor="end" height={60} />
                <YAxis domain={[0, 100]} unit="%" />
                <Tooltip formatter={v => v + '%'} />
                <Bar dataKey="rate" name="Notify Rate" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── By Patient ─── */}
      {tab === 'patient' && breakdown && (
        <Card title="Emergency Contact Coverage by Patient">
          <div style={{ marginBottom: 12 }}>
            <input placeholder="Filter patients..." value={filter} onChange={e => setFilter(e.target.value)}
              style={{ padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: 8, width: 260, fontSize: 13 }} />
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead style={{ background: '#f8fafc' }}>
              <tr>
                {hdr('patient_id', 'Patient')}
                {hdr('contact_count', 'Contacts')}
                {hdr('relationships', 'Relationships')}
                {hdr('notify_enabled', 'Notify Enabled')}
                {hdr('freshest_days', 'Freshest (days)')}
                {hdr('stalest_days', 'Stalest (days)')}
              </tr>
            </thead>
            <tbody>
              {sorted(filterRows(breakdown.by_patient || [])).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.contact_count}</td>
                  <td style={{ padding: '6px 10px' }}>
                    {(r.relationships || '').split(', ').map((rel, j) => (
                      <span key={j} style={{ marginRight: 4 }}><RelBadge rel={rel} /></span>
                    ))}
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.notify_enabled}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <span style={{ color: r.freshest_days > 180 ? '#ef4444' : r.freshest_days > 90 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                      {fmt(r.freshest_days)}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <span style={{ color: r.stalest_days > 180 ? '#ef4444' : r.stalest_days > 90 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                      {fmt(r.stalest_days)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* ─── Definitions ─── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Relationship Types">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.relationship_types || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><RelBadge rel={r.type} /></td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Verification Standards">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Standard</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Target</th>
                </tr>
              </thead>
              <tbody>
                {(defs.verification_standards || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.standard}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{s.target}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
