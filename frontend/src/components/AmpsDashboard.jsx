import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter,
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

function TierBadge({ tier }) {
  const t = String(tier || '')
  const map = {
    competent: '#10b981', motor_risk: '#f59e0b', process_risk: '#3b82f6', dual_risk: '#ef4444'
  }
  const color = map[t] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{t.replace(/_/g, ' ') || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'All Patients' },
  { id: 'skills', label: 'Skill Groups' },
  { id: 'scatter', label: 'Motor vs Process' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AmpsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/amps-dashboard/overview`),
      axios.get(`${API_URL}/api/amps-dashboard/breakdown`),
      axios.get(`${API_URL}/api/amps-dashboard/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AMPS data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>AMPS Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Assessment of Motor and Process Skills — {fmt(overview?.total_assessments)} assessments, {fmt(overview?.unique_patients)} patients
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
      {tab === 'patients' && <PatientsTab patients={overview?.patient_summary || []} />}
      {tab === 'skills' && <SkillsTab breakdown={breakdown} />}
      {tab === 'scatter' && <ScatterTab data={breakdown?.motor_vs_process || []} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

/* ── Overview ─────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return null
  const perfDist = data.performance_distribution || {}
  const pieData = Object.entries(perfDist).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="Total Assessments" value={data.total_assessments} color="#2563eb" />
          <KPI label="Unique Patients" value={data.unique_patients} color="#8b5cf6" />
          <KPI label="Avg Motor Logit" value={data.avg_motor_logit} sub={`range ${fmt(data.kpi?.min_motor)} – ${fmt(data.kpi?.max_motor)}`} color="#10b981" />
          <KPI label="Avg Process Logit" value={data.avg_process_logit} sub={`range ${fmt(data.kpi?.min_process)} – ${fmt(data.kpi?.max_process)}`} color="#3b82f6" />
          <KPI label="Competent" value={perfDist.competent} color="#10b981" />
          <KPI label="Dual Risk" value={perfDist.dual_risk} color="#ef4444" />
        </div>
      </Card>

      <Card title="Performance Distribution" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
              {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Performance Tier Counts" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={pieData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ── All Patients ─────────────────────────────────────── */
function PatientsTab({ patients }) {
  const [sort, setSort] = useState('patient_id')
  const [dir, setDir] = useState(1)
  const [filter, setFilter] = useState('')

  const toggle = col => { setDir(sort === col ? -dir : 1); setSort(col) }
  const arrow = col => sort === col ? (dir === 1 ? ' ▲' : ' ▼') : ''

  const filtered = patients.filter(p =>
    !filter || p.patient_id?.toLowerCase().includes(filter.toLowerCase()) ||
    p.name?.toLowerCase().includes(filter.toLowerCase()) ||
    p.overall_tier?.toLowerCase().includes(filter.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sort], bv = b[sort]
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir
  })

  const hdr = { padding: '8px 10px', fontSize: 12, color: '#64748b', cursor: 'pointer', borderBottom: '2px solid #e2e8f0', textAlign: 'left', whiteSpace: 'nowrap' }
  const cell = { padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <Card>
      <input placeholder="Filter by patient, name, or tier…" value={filter} onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', maxWidth: 360, padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 13, marginBottom: 12 }} />
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={hdr} onClick={() => toggle('patient_id')}>Patient{arrow('patient_id')}</th>
              <th style={hdr} onClick={() => toggle('name')}>Name{arrow('name')}</th>
              <th style={hdr} onClick={() => toggle('age')}>Age{arrow('age')}</th>
              <th style={hdr} onClick={() => toggle('gender')}>Gender{arrow('gender')}</th>
              <th style={hdr} onClick={() => toggle('motor_logit')}>Motor Logit{arrow('motor_logit')}</th>
              <th style={hdr} onClick={() => toggle('process_logit')}>Process Logit{arrow('process_logit')}</th>
              <th style={hdr} onClick={() => toggle('overall_tier')}>Tier{arrow('overall_tier')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                <td style={cell}>{p.patient_id}</td>
                <td style={cell}>{p.name}</td>
                <td style={cell}>{p.age}</td>
                <td style={cell}>{p.gender}</td>
                <td style={{ ...cell, fontWeight: 600 }}>{fmt(p.motor_logit)}</td>
                <td style={{ ...cell, fontWeight: 600 }}>{fmt(p.process_logit)}</td>
                <td style={cell}><TierBadge tier={p.overall_tier} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>Showing {sorted.length} of {patients.length} patients</div>
    </Card>
  )
}

/* ── Skill Groups ─────────────────────────────────────── */
function SkillsTab({ breakdown }) {
  if (!breakdown) return null
  const motor = breakdown.motor_group_summary || []
  const process = breakdown.process_group_summary || []
  const items = breakdown.item_heatmap || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title="Motor Skill Groups (avg % of max)">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={motor} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
            <YAxis dataKey="label" type="category" tick={{ fontSize: 11 }} width={120} />
            <Tooltip formatter={v => `${v}%`} />
            <Bar dataKey="pct" fill="#10b981" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Process Skill Groups (avg % of max)">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={process} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
            <YAxis dataKey="label" type="category" tick={{ fontSize: 11 }} width={120} />
            <Tooltip formatter={v => `${v}%`} />
            <Bar dataKey="pct" fill="#3b82f6" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Item Scores (lowest first)" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Item', 'Domain', 'Group', 'Avg Score', 'Max', '% of Max'].map(h => (
                  <th key={h} style={{ padding: '6px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((it, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{it.label}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: it.domain === 'motor' ? '#10b981' : '#3b82f6', fontWeight: 600 }}>{it.domain}</span>
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}>{it.group?.replace(/_/g, ' ')}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{fmt(it.avg_score)}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}>{it.max_score}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 60, height: 6, background: '#f1f5f9', borderRadius: 3 }}>
                        <div style={{ width: `${((it.avg_score / it.max_score) * 100).toFixed(0)}%`, height: 6, background: it.avg_score / it.max_score > 0.75 ? '#10b981' : it.avg_score / it.max_score > 0.5 ? '#f59e0b' : '#ef4444', borderRadius: 3 }} />
                      </div>
                      <span style={{ fontSize: 12 }}>{((it.avg_score / it.max_score) * 100).toFixed(0)}%</span>
                    </div>
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

/* ── Motor vs Process Scatter ─────────────────────────── */
function ScatterTab({ data }) {
  const tierColors = { competent: '#10b981', motor_risk: '#f59e0b', process_risk: '#3b82f6', dual_risk: '#ef4444' }
  const tiers = [...new Set(data.map(d => d.overall_tier))]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Motor vs Process Logit Scatter">
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey="motor_logit" name="Motor Logit" tick={{ fontSize: 11 }} label={{ value: 'Motor Logit', position: 'insideBottom', offset: -10, fontSize: 12 }} />
            <YAxis type="number" dataKey="process_logit" name="Process Logit" tick={{ fontSize: 11 }} label={{ value: 'Process Logit', angle: -90, position: 'insideLeft', fontSize: 12 }} />
            <Tooltip content={({ payload }) => {
              if (!payload?.length) return null
              const d = payload[0].payload
              return (
                <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, padding: 10, fontSize: 12 }}>
                  <div style={{ fontWeight: 700 }}>{d.patient_id}</div>
                  <div>Motor: {fmt(d.motor_logit)}</div>
                  <div>Process: {fmt(d.process_logit)}</div>
                  <div>Tier: <TierBadge tier={d.overall_tier} /></div>
                </div>
              )
            }} />
            <Legend />
            {tiers.map(tier => (
              <Scatter
                key={tier}
                name={tier.replace(/_/g, ' ')}
                data={data.filter(d => d.overall_tier === tier)}
                fill={tierColors[tier] || '#64748b'}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Patient Motor vs Process Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Patient', 'Motor Logit', 'Process Logit', 'Tier'].map(h => (
                  <th key={h} style={{ padding: '6px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((d, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}>{d.patient_id}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{fmt(d.motor_logit)}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{fmt(d.process_logit)}</td>
                  <td style={{ padding: '6px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }}><TierBadge tier={d.overall_tier} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ── Definitions ──────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  if (!defs) return null
  return (
    <Card title={defs.title || 'Definitions'}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', textAlign: 'left', width: '25%' }}>Term</th>
            <th style={{ padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>Definition</th>
          </tr>
        </thead>
        <tbody>
          {(defs.definitions || []).map((d, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
              <td style={{ padding: '8px 10px', fontSize: 13, fontWeight: 600, borderBottom: '1px solid #f1f5f9', verticalAlign: 'top' }}>{d.term}</td>
              <td style={{ padding: '8px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9', lineHeight: 1.5 }}>{d.definition}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
