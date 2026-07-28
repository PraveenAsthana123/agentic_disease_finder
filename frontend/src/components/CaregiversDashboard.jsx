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

function RoleBadge({ role }) {
  const r = String(role || '').toLowerCase()
  const map = {
    spouse: '#3b82f6', parent: '#10b981', sibling: '#f59e0b', child: '#8b5cf6',
    friend: '#06b6d4', professional: '#ec4899'
  }
  const color = map[r] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{r || '--'}</span>
  )
}

function BurnoutBadge({ score }) {
  const s = Number(score) || 0
  const color = s <= 25 ? '#10b981' : s <= 50 ? '#f59e0b' : s <= 75 ? '#ef4444' : '#7f1d1d'
  const label = s <= 25 ? 'Low' : s <= 50 ? 'Moderate' : s <= 75 ? 'High' : 'Critical'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{label} ({s})</span>
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
  { id: 'all', label: 'All Caregivers' },
  { id: 'role', label: 'By Role' },
  { id: 'wellness', label: 'Wellness & Burnout' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CaregiversDashboard() {
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
      axios.get(`${API_URL}/api/caregivers/overview`),
      axios.get(`${API_URL}/api/caregivers/breakdown`),
      axios.get(`${API_URL}/api/caregivers/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading caregivers...</div>
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
  const roleDist = (overview?.role_distribution || []).map(r => ({ name: r.role, value: r.cnt }))
  const availDist = (overview?.availability_distribution || []).map(r => ({ name: r.availability, value: r.cnt }))
  const burnoutDist = (overview?.burnout_distribution || []).map(r => ({ name: r.tier, value: r.cnt }))
  const burnoutColors = { 'Low (0-25)': '#10b981', 'Moderate (26-50)': '#f59e0b', 'High (51-75)': '#ef4444', 'Critical (76-100)': '#7f1d1d' }
  const trainingCounts = overview?.training_counts || []
  const roleWellness = overview?.role_wellness || []

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Caregivers Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Caregiver registry &amp; wellness — {fmt(kpis.total_caregivers)} caregivers, {fmt(kpis.patients_covered)} patients covered, avg {fmt(kpis.avg_experience_years)} yrs experience, avg burnout {fmt(kpis.avg_burnout)}/100
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
              <KPI label="Total Caregivers" value={kpis.total_caregivers} color="#3b82f6" />
              <KPI label="Patients Covered" value={kpis.patients_covered} color="#10b981" />
              <KPI label="Avg Experience" value={kpis.avg_experience_years} sub="years" color="#8b5cf6" />
              <KPI label="Epilepsy Trained" value={kpis.epilepsy_trained_pct} sub="%" color="#06b6d4" />
              <KPI label="First Aid Cert" value={kpis.first_aid_pct} sub="%" color="#10b981" />
              <KPI label="Avg Burnout" value={kpis.avg_burnout} sub="/ 100" color="#ef4444" />
              <KPI label="Avg Stress" value={kpis.avg_stress} sub="/ 10" color="#f59e0b" />
              <KPI label="Avg Confidence" value={kpis.avg_confidence} sub="/ 10" color="#3b82f6" />
            </div>
          </Card>

          <Card title="Role Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={roleDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {roleDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Availability Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={availDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {availDist.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Burnout Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={burnoutDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {burnoutDist.map((e, i) => <Cell key={i} fill={burnoutColors[e.name] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Training & Certification Rates" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={trainingCounts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Certified" fill="#3b82f6" />
                <Bar dataKey="total" name="Total" fill="#e2e8f0" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Wellness by Role" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={roleWellness}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="avg_stress" name="Avg Stress (1-10)" fill="#f59e0b" />
                <Bar dataKey="avg_confidence" name="Avg Confidence (1-10)" fill="#3b82f6" />
                <Bar dataKey="avg_burnout" name="Avg Burnout (0-100)" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── All Caregivers ─── */}
      {tab === 'all' && breakdown && (
        <Card>
          <div style={{ marginBottom: 12 }}>
            <input placeholder="Filter caregivers..." value={filter} onChange={e => setFilter(e.target.value)}
              style={{ padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: 8, width: 260, fontSize: 13 }} />
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  {hdr('patient_id', 'Patient')}
                  {hdr('name', 'Caregiver')}
                  {hdr('role', 'Role')}
                  {hdr('availability', 'Availability')}
                  {hdr('experience_years', 'Exp (yrs)')}
                  {hdr('epilepsy_training_completed', 'Epilepsy Tr.')}
                  {hdr('first_aid_certified', 'First Aid')}
                  {hdr('rescue_med_trained', 'Rescue Med')}
                  {hdr('seizure_first_aid_confidence', 'Confidence')}
                  {hdr('burnout_score', 'Burnout')}
                  {hdr('caregiver_stress', 'Stress')}
                </tr>
              </thead>
              <tbody>
                {sorted(filterRows(breakdown.all_caregivers || [])).map((r, i) => (
                  <tr key={r.id || i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{r.name}</td>
                    <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                    <td style={{ padding: '6px 10px', fontSize: 12, textTransform: 'capitalize' }}>{r.availability}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.experience_years}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><YesNoBadge val={r.epilepsy_training_completed} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><YesNoBadge val={r.first_aid_certified} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><YesNoBadge val={r.rescue_med_trained} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      <span style={{ color: r.seizure_first_aid_confidence >= 7 ? '#10b981' : r.seizure_first_aid_confidence >= 4 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                        {r.seizure_first_aid_confidence}/10
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}><BurnoutBadge score={r.burnout_score} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                      <span style={{ color: r.caregiver_stress >= 7 ? '#ef4444' : r.caregiver_stress >= 4 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {r.caregiver_stress}/10
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── By Role ─── */}
      {tab === 'role' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Role Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  {hdr('role', 'Role')}
                  {hdr('total', 'Count')}
                  {hdr('avg_experience', 'Avg Exp (yrs)')}
                  {hdr('avg_burnout', 'Avg Burnout')}
                  {hdr('avg_stress', 'Avg Stress')}
                  {hdr('avg_confidence', 'Avg Confidence')}
                  {hdr('trained_count', 'Epilepsy Trained')}
                  {hdr('first_aid_count', 'First Aid')}
                </tr>
              </thead>
              <tbody>
                {sorted(breakdown.by_role || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{r.total}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(r.avg_experience)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                      <span style={{ color: r.avg_burnout > 60 ? '#ef4444' : r.avg_burnout > 40 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {fmt(r.avg_burnout)}
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(r.avg_stress)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(r.avg_confidence)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.trained_count}/{r.total}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.first_aid_count}/{r.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Availability Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  {hdr('availability', 'Availability')}
                  {hdr('total', 'Count')}
                  {hdr('avg_experience', 'Avg Exp (yrs)')}
                  {hdr('avg_burnout', 'Avg Burnout')}
                  {hdr('avg_stress', 'Avg Stress')}
                  {hdr('trained_count', 'Epilepsy Trained')}
                </tr>
              </thead>
              <tbody>
                {sorted(breakdown.by_availability || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', textTransform: 'capitalize', fontWeight: 600 }}>{r.availability}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{r.total}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(r.avg_experience)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                      <span style={{ color: r.avg_burnout > 60 ? '#ef4444' : r.avg_burnout > 40 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                        {fmt(r.avg_burnout)}
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(r.avg_stress)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.trained_count}/{r.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ─── Wellness & Burnout ─── */}
      {tab === 'wellness' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="High-Burnout Caregivers (Score > 60) — Intervention Needed">
            {(breakdown.high_burnout || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No high-burnout caregivers found.</p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead style={{ background: '#fef2f2' }}>
                    <tr>
                      {hdr('patient_id', 'Patient')}
                      {hdr('name', 'Caregiver')}
                      {hdr('role', 'Role')}
                      {hdr('availability', 'Availability')}
                      {hdr('burnout_score', 'Burnout')}
                      {hdr('caregiver_stress', 'Stress')}
                      {hdr('caregiver_sleep_quality', 'Sleep')}
                      {hdr('work_impact', 'Work Impact')}
                      {hdr('days_since_respite', 'Days Since Respite')}
                    </tr>
                  </thead>
                  <tbody>
                    {sorted(breakdown.high_burnout || []).map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #fecaca' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{r.name}</td>
                        <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                        <td style={{ padding: '6px 10px', fontSize: 12, textTransform: 'capitalize' }}>{r.availability}</td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}><BurnoutBadge score={r.burnout_score} /></td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                          <span style={{ color: '#ef4444', fontWeight: 600 }}>{r.caregiver_stress}/10</span>
                        </td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                          <span style={{ color: r.caregiver_sleep_quality <= 4 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{r.caregiver_sleep_quality}/10</span>
                        </td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                          <span style={{ color: r.work_impact >= 7 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{r.work_impact}/10</span>
                        </td>
                        <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                          <span style={{ color: r.days_since_respite > 180 ? '#ef4444' : '#f59e0b', fontWeight: 600 }}>{fmt(r.days_since_respite)}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ─── Definitions ─── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Field Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12 }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12 }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Caregiver Roles">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12 }}>Role</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12 }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.roles || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Wellness Thresholds">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ background: '#f8fafc' }}>
                <tr>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontSize: 12 }}>Metric</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#10b981' }}>Low</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#f59e0b' }}>Moderate</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#ef4444' }}>High</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontSize: 12, color: '#7f1d1d' }}>Critical</th>
                </tr>
              </thead>
              <tbody>
                {(defs.wellness_thresholds || []).map((w, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{w.metric}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{w.low}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{w.moderate}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{w.high}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{w.critical}</td>
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
