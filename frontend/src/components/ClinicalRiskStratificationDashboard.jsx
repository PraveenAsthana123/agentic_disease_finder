import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, Radar
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const TIER_COLORS = { Critical: '#ef4444', High: '#f59e0b', Moderate: '#3b82f6', Low: '#10b981' }
const COMPONENT_COLORS = ['#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
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
  const color = TIER_COLORS[tier] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{tier || '--'}</span>
  )
}

function ScoreBar({ value, max }) {
  const pct = Math.min(100, Math.round(((value || 0) / (max || 50)) * 100))
  const color = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f59e0b' : '#10b981'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <span style={{ fontSize: 12, color: '#475569', minWidth: 32, textAlign: 'right' }}>{fmt(value)}</span>
    </div>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'All Patients' },
  { id: 'highrisk', label: 'High-Risk' },
  { id: 'components', label: 'Components' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ClinicalRiskStratificationDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/clinical-risk-stratification/overview`),
      axios.get(`${API_URL}/api/clinical-risk-stratification/breakdown`),
      axios.get(`${API_URL}/api/clinical-risk-stratification/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading clinical risk stratification…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Clinical Risk Stratification
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Composite epilepsy risk scoring from 6 data sources — {fmt(overview?.total_patients)} patients ·{' '}
        avg score {fmt(overview?.avg_composite_score)} · {fmt(overview?.high_risk_rate_pct)}% high-risk
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
      {tab === 'patients' && <PatientsTab data={breakdown} />}
      {tab === 'highrisk' && <HighRiskTab data={overview} breakdown={breakdown} />}
      {tab === 'components' && <ComponentsTab data={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null

  const tierData = (data.tier_distribution || []).map(t => ({
    name: t.tier, value: t.count, color: TIER_COLORS[t.tier] || '#64748b', pct: t.pct
  }))

  const histData = (data.score_histogram || []).map(h => ({ range: h.range, count: h.count }))

  const componentData = (data.avg_components || []).map(c => ({
    component: c.component, avg: parseFloat(fmt(c.avg)), max: c.max
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card title="Risk Summary" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="Total Patients" value={data.total_patients} color="#1e293b" />
          <KPI label="Avg Risk Score" value={data.avg_composite_score} color="#8b5cf6" sub="out of 100" />
          <KPI label="Critical" value={data.critical_count} color="#ef4444" sub="≥35 pts" />
          <KPI label="High" value={data.high_count} color="#f59e0b" sub="23–34 pts" />
          <KPI label="Moderate" value={data.moderate_count} color="#3b82f6" sub="12–22 pts" />
          <KPI label="Low" value={data.low_count} color="#10b981" sub="<12 pts" />
        </div>
      </Card>

      <Card title="Tier Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={tierData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={78}
              label={({ name, pct }) => `${name} ${pct != null ? pct + '%' : ''}`}>
              {tierData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
            </Pie>
            <Tooltip formatter={(v, n) => [v, n]} />
          </PieChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap', marginTop: 8 }}>
          {tierData.map(t => (
            <div key={t.name} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
              <div style={{ width: 10, height: 10, borderRadius: '50%', background: t.color }} />
              <span style={{ color: '#475569' }}>{t.name} ({t.value})</span>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Score Distribution (histogram)">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={histData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="range" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {histData.map((entry, i) => {
                const lo = parseInt((entry.range || '0').split('-')[0])
                const c = lo >= 35 ? '#ef4444' : lo >= 23 ? '#f59e0b' : lo >= 12 ? '#3b82f6' : '#10b981'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Avg Component Scores vs Max Points">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={componentData} layout="vertical" margin={{ top: 4, right: 16, left: 110, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="component" type="category" tick={{ fontSize: 11 }} width={110} />
            <Tooltip />
            <Legend />
            <Bar dataKey="avg" name="Avg Score" fill="#ef4444" radius={[0, 4, 4, 0]} />
            <Bar dataKey="max" name="Max Points" fill="#cbd5e1" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Top High-Risk Patients" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Patient', 'Score', 'Tier', 'Seizure Burden', 'Adherence Risk', 'Genetic Risk', 'Comorbidity', 'QoL Deficit'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.high_risk_patients || []).slice(0, 8).map((p, i) => (
              <tr key={i} style={{ borderTop: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                <td style={{ padding: '8px 12px', fontWeight: 700, color: TIER_COLORS[p.tier] || '#1e293b' }}>{fmt(p.composite_score)}</td>
                <td style={{ padding: '8px 12px' }}><TierBadge tier={p.tier} /></td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{fmt(p.seizure_burden)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f59e0b', fontWeight: 600 }}>{fmt(p.adherence_risk)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#8b5cf6', fontWeight: 600 }}>{fmt(p.genetic_risk)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#06b6d4', fontWeight: 600 }}>{fmt(p.comorbidity_burden)}</td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#ec4899', fontWeight: 600 }}>{fmt(p.qol_deficit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function PatientsTab({ data }) {
  const [sortKey, setSortKey] = useState('composite_score')
  const [sortDir, setSortDir] = useState('desc')
  const [filterTier, setFilterTier] = useState('')
  const [search, setSearch] = useState('')

  if (!data) return null

  const patients = data.all_patients || []
  const filtered = patients.filter(p =>
    (!filterTier || p.tier === filterTier) &&
    (!search || (p.patient_id || '').toLowerCase().includes(search.toLowerCase()))
  )
  const sorted = [...filtered].sort((a, b) => {
    const va = a[sortKey] ?? 0; const vb = b[sortKey] ?? 0
    return sortDir === 'asc' ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1)
  })

  function toggleSort(k) {
    if (sortKey === k) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(k); setSortDir('desc') }
  }

  const SH = ({ k, label }) => (
    <th onClick={() => toggleSort(k)} style={{
      padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569',
      fontSize: 12, cursor: 'pointer', whiteSpace: 'nowrap'
    }}>{label} {sortKey === k ? (sortDir === 'asc' ? '↑' : '↓') : ''}</th>
  )

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, marginBottom: 16, alignItems: 'center' }}>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search patient ID…"
          style={{ padding: '7px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13, width: 200 }} />
        <select value={filterTier} onChange={e => setFilterTier(e.target.value)}
          style={{ padding: '7px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
          <option value="">All Tiers</option>
          {['Critical', 'High', 'Moderate', 'Low'].map(t => <option key={t} value={t}>{t}</option>)}
        </select>
        <span style={{ color: '#64748b', fontSize: 13 }}>{sorted.length} patients</span>
      </div>
      <Card>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <SH k="patient_id" label="Patient" />
                <SH k="composite_score" label="Risk Score" />
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#475569', fontSize: 12 }}>Tier</th>
                <SH k="seizure_burden" label="Seizure" />
                <SH k="adherence_risk" label="Adherence" />
                <SH k="genetic_risk" label="Genetic" />
                <SH k="comorbidity_burden" label="Comorbidity" />
                <SH k="qol_deficit" label="QoL Deficit" />
                <SH k="age" label="Age" />
                <th style={{ padding: '8px 12px', fontWeight: 600, color: '#475569', fontSize: 12 }}>Gender</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((p, i) => (
                <tr key={i} style={{ borderTop: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ padding: '8px 12px', minWidth: 140 }}>
                    <ScoreBar value={p.composite_score} max={50} />
                  </td>
                  <td style={{ padding: '8px 12px' }}><TierBadge tier={p.tier} /></td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{fmt(p.seizure_burden)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#f59e0b', fontWeight: 600 }}>{fmt(p.adherence_risk)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#8b5cf6', fontWeight: 600 }}>{fmt(p.genetic_risk)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#06b6d4', fontWeight: 600 }}>{fmt(p.comorbidity_burden)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#ec4899', fontWeight: 600 }}>{fmt(p.qol_deficit)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#475569' }}>{fmt(p.age)}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{p.gender || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function HighRiskTab({ data, breakdown }) {
  if (!data || !breakdown) return null

  const highRisk = (breakdown.all_patients || []).filter(p => ['Critical', 'High'].includes(p.tier))

  function weakestDim(p) {
    const dims = [
      { name: 'Seizure Burden', val: (p.seizure_burden || 0) / 30 },
      { name: 'Adherence Risk', val: (p.adherence_risk || 0) / 25 },
      { name: 'Genetic Risk', val: (p.genetic_risk || 0) / 20 },
      { name: 'Comorbidity', val: (p.comorbidity_burden || 0) / 15 },
      { name: 'QoL Deficit', val: (p.qol_deficit || 0) / 10 },
    ]
    return dims.reduce((a, b) => b.val > a.val ? b : a).name
  }

  const n = highRisk.length || 1
  const radarData = [
    { dim: 'Seizure\nBurden', A: Math.round(highRisk.reduce((s, p) => s + (p.seizure_burden || 0), 0) / n / 30 * 100) },
    { dim: 'Adherence\nRisk', A: Math.round(highRisk.reduce((s, p) => s + (p.adherence_risk || 0), 0) / n / 25 * 100) },
    { dim: 'Genetic\nRisk', A: Math.round(highRisk.reduce((s, p) => s + (p.genetic_risk || 0), 0) / n / 20 * 100) },
    { dim: 'Comorbidity', A: Math.round(highRisk.reduce((s, p) => s + (p.comorbidity_burden || 0), 0) / n / 15 * 100) },
    { dim: 'QoL\nDeficit', A: Math.round(highRisk.reduce((s, p) => s + (p.qol_deficit || 0), 0) / n / 10 * 100) },
  ]

  const weakCounts = {}
  highRisk.forEach(p => { const w = weakestDim(p); weakCounts[w] = (weakCounts[w] || 0) + 1 })
  const weakData = Object.entries(weakCounts).map(([name, count]) => ({ name, count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      <Card title={`Cohort Radar — ${highRisk.length} High-Risk Patients (% of max pts)`}>
        <ResponsiveContainer width="100%" height={260}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="dim" tick={{ fontSize: 12 }} />
            <Radar name="Avg % of Max" dataKey="A" stroke="#ef4444" fill="#ef4444" fillOpacity={0.25} />
            <Tooltip formatter={v => [`${v}%`, 'Avg % of Max']} />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Primary Risk Driver per Patient">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={weakData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="High-Risk Patient Detail" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Patient', 'Score', 'Tier', 'Seizure', 'Adherence', 'Genetic', 'Comorbidity', 'QoL', 'Primary Driver'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {highRisk.map((p, i) => {
              const driver = weakestDim(p)
              const hl = (dim) => driver === dim ? { background: '#fef9c3' } : {}
              return (
                <tr key={i} style={{ borderTop: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 700, color: TIER_COLORS[p.tier] }}>{fmt(p.composite_score)}</td>
                  <td style={{ padding: '8px 12px' }}><TierBadge tier={p.tier} /></td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', ...hl('Seizure Burden') }}>
                    <span style={{ color: '#ef4444', fontWeight: driver === 'Seizure Burden' ? 700 : 400 }}>{fmt(p.seizure_burden)}</span>
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', ...hl('Adherence Risk') }}>
                    <span style={{ color: '#f59e0b', fontWeight: driver === 'Adherence Risk' ? 700 : 400 }}>{fmt(p.adherence_risk)}</span>
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', ...hl('Genetic Risk') }}>
                    <span style={{ color: '#8b5cf6', fontWeight: driver === 'Genetic Risk' ? 700 : 400 }}>{fmt(p.genetic_risk)}</span>
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', ...hl('Comorbidity') }}>
                    <span style={{ color: '#06b6d4', fontWeight: driver === 'Comorbidity' ? 700 : 400 }}>{fmt(p.comorbidity_burden)}</span>
                  </td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', ...hl('QoL Deficit') }}>
                    <span style={{ color: '#ec4899', fontWeight: driver === 'QoL Deficit' ? 700 : 400 }}>{fmt(p.qol_deficit)}</span>
                  </td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={{ fontSize: 11, color: '#ef4444', fontWeight: 600 }}>▲ {driver}</span>
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

function ComponentsTab({ data, breakdown }) {
  if (!data) return null

  const components = data.avg_components || []

  const sample = (breakdown?.all_patients || []).slice(0, 12)
  const stackData = sample.map(p => ({
    patient: (p.patient_id || '').replace('EPAT', 'P'),
    seizure: p.seizure_burden || 0,
    adherence: p.adherence_risk || 0,
    genetic: p.genetic_risk || 0,
    comorbidity: p.comorbidity_burden || 0,
    qol: p.qol_deficit || 0,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      {components.map((c, i) => {
        const pct = Math.round(((c.avg || 0) / (c.max || 1)) * 100)
        const color = COMPONENT_COLORS[i % COMPONENT_COLORS.length]
        return (
          <Card key={i} title={c.component}>
            <div style={{ fontSize: 32, fontWeight: 700, color }}>{fmt(c.avg)}</div>
            <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 12 }}>avg of max {c.max} pts</div>
            <div style={{ height: 12, background: '#f1f5f9', borderRadius: 6, overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 6 }} />
            </div>
            <div style={{ fontSize: 12, color: '#475569', marginTop: 6, textAlign: 'right' }}>
              {pct}% of maximum risk points
            </div>
          </Card>
        )
      })}

      <Card title="Stacked Risk Components — first 12 patients" span={3}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={stackData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="patient" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="seizure" name="Seizure Burden" stackId="a" fill={COMPONENT_COLORS[0]} />
            <Bar dataKey="adherence" name="Adherence Risk" stackId="a" fill={COMPONENT_COLORS[1]} />
            <Bar dataKey="genetic" name="Genetic Risk" stackId="a" fill={COMPONENT_COLORS[2]} />
            <Bar dataKey="comorbidity" name="Comorbidity" stackId="a" fill={COMPONENT_COLORS[3]} />
            <Bar dataKey="qol" name="QoL Deficit" stackId="a" fill={COMPONENT_COLORS[4]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return null

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="About This Dashboard" span={2}>
        <p style={{ color: '#475569', fontSize: 13, margin: 0, lineHeight: 1.7 }}>{data.description}</p>
      </Card>

      <Card title="Risk Tiers">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Tier', 'Score Range', 'Meaning'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.risk_tiers || []).map((t, i) => (
              <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px' }}><TierBadge tier={t.tier} /></td>
                <td style={{ padding: '8px 12px', color: '#475569', fontWeight: 600 }}>{t.range}</td>
                <td style={{ padding: '8px 12px', color: '#475569' }}>{t.meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Risk Components">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Component', 'Max Pts', 'Data Source'].map(h => (
                <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', fontSize: 12 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.risk_components || []).map((c, i) => (
              <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, color: COMPONENT_COLORS[i % COMPONENT_COLORS.length] }}>{c.component}</td>
                <td style={{ padding: '8px 12px', textAlign: 'center', color: '#475569' }}>{c.max_points}</td>
                <td style={{ padding: '8px 12px', color: '#94a3b8', fontSize: 12 }}>{(c.sources || []).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {(data.data_sources || []).length > 0 && (
        <Card title="Data Sources" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            {(data.data_sources || []).map((s, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 600, color: '#334155', fontSize: 13 }}>{s.table}</div>
                <div style={{ color: '#64748b', fontSize: 12, marginTop: 4 }}>{s.description}</div>
                {s.rows && <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>{s.rows} rows</div>}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
