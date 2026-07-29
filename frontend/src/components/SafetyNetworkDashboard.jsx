import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899']
const TIER_COLORS = { Critical: '#ef4444', 'At Risk': '#f59e0b', Adequate: '#3b82f6', Strong: '#10b981' }

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
  const color = TIER_COLORS[tier] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{tier || '--'}</span>
  )
}

function ScorePill({ score }) {
  const s = score != null ? Number(score) : null
  const color = s == null ? '#94a3b8' : s >= 80 ? '#10b981' : s >= 60 ? '#3b82f6' : s >= 40 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{
        flex: 1, height: 8, borderRadius: 4, background: '#f1f5f9', overflow: 'hidden', minWidth: 60
      }}>
        <div style={{
          width: `${Math.min(100, Math.max(0, s ?? 0))}%`, height: '100%',
          background: color, borderRadius: 4, transition: 'width .3s'
        }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color, minWidth: 32, textAlign: 'right' }}>
        {s != null ? s.toFixed(0) : '--'}
      </span>
    </div>
  )
}

function ScoreCell({ score }) {
  const s = score != null ? Number(score) : null
  const color = s == null ? '#94a3b8' : s >= 80 ? '#10b981' : s >= 60 ? '#3b82f6' : s >= 40 ? '#f59e0b' : '#ef4444'
  return (
    <span style={{ fontWeight: 600, color }}>
      {s != null ? s.toFixed(0) : '--'}
    </span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'All Patients' },
  { id: 'critical', label: 'Critical' },
  { id: 'dimensions', label: 'Dimensions' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SafetyNetworkDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/safety-network/overview`),
      axios.get(`${API_URL}/api/safety-network/breakdown`),
      axios.get(`${API_URL}/api/safety-network/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading patient safety network data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Patient Safety Network Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Composite safety scores across medication adherence, caregiver coverage, emergency readiness, wearable monitoring and IoT alert health —{' '}
        {fmt(kpis.total_patients)} patients · avg score {fmt(kpis.avg_composite_score)} · {fmt(kpis.critical_count)} critical
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
      {tab === 'patients' && <PatientsTab patients={breakdown?.per_patient || []} />}
      {tab === 'critical' && <CriticalTab patients={breakdown?.per_patient || []} />}
      {tab === 'dimensions' && <DimensionsTab data={overview} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

/* ─── Overview ─────────────────────────────────────────────────────────────── */

function OverviewTab({ data }) {
  if (!data) return null
  const kpis = data.kpis || {}
  const tierDist = data.tier_distribution || []
  const histogram = data.score_histogram || []
  const dimAvg = data.dimension_averages || []
  const criticalPts = (data.critical_patients || []).slice(0, 5)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>

      {/* 9 KPIs */}
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(9, 1fr)', gap: 12 }}>
          <KPI label="Total Patients" value={kpis.total_patients} color="#3b82f6" />
          <KPI label="Avg Composite Score" value={kpis.avg_composite_score} color="#8b5cf6" />
          <KPI label="Critical" value={kpis.critical_count} color="#ef4444" />
          <KPI label="At Risk" value={kpis.at_risk_count} color="#f59e0b" />
          <KPI label="Adequate" value={kpis.adequate_count} color="#3b82f6" />
          <KPI label="Strong" value={kpis.strong_count} color="#10b981" />
          <KPI label="With Caregiver" value={kpis.patients_with_caregiver} color="#06b6d4" />
          <KPI label="Emergency Contact" value={kpis.patients_with_emergency_contact} color="#ec4899" />
          <KPI label="Wearable Active" value={kpis.patients_with_wearable} color="#10b981" />
        </div>
      </Card>

      {/* Tier Distribution Pie */}
      <Card title="Tier Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie
              data={tierDist}
              dataKey="count"
              nameKey="tier"
              cx="50%" cy="50%"
              outerRadius={95}
              label={({ tier, count }) => `${tier} (${count})`}
              labelLine={false}
            >
              {tierDist.map((d, i) => (
                <Cell key={i} fill={TIER_COLORS[d.tier] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Score Histogram */}
      <Card title="Score Distribution">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={histogram}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
              {histogram.map((d, i) => {
                const r = d.range || ''
                const lo = parseInt(r.split('-')[0], 10)
                const c = lo >= 80 ? '#10b981' : lo >= 60 ? '#3b82f6' : lo >= 40 ? '#f59e0b' : '#ef4444'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Dimension Averages Horizontal Bar */}
      <Card title="Dimension Averages (score & weight)">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={dimAvg} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="dimension" tick={{ fontSize: 11 }} width={135} />
            <Tooltip formatter={(v, name) => name === 'score' ? `${Number(v).toFixed(1)}` : `${v}%`} />
            <Legend />
            <Bar dataKey="score" name="Avg Score" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            <Bar dataKey="weight_pct" name="Weight %" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Top 5 Critical Patients table */}
      <Card title="Top Critical Patients" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#fef2f2' }}>
                <th style={{ ...thStyle, color: '#dc2626' }}>Patient ID</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Composite Score</th>
                <th style={{ ...thStyle, color: '#dc2626' }}>Tier</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Caregiver</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Emergency</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Adherence</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Wearable</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>IoT Alert</th>
              </tr>
            </thead>
            <tbody>
              {criticalPts.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ ...tdStyle, fontWeight: 700, color: '#1e293b' }}>{p.patient_id}</td>
                  <td style={{ ...tdStyle, minWidth: 140 }}><ScorePill score={p.composite_score} /></td>
                  <td style={tdStyle}><TierBadge tier={p.tier} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.caregiver_score} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.emergency_score} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.adherence_score} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.wearable_score} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.alert_score} /></td>
                </tr>
              ))}
              {criticalPts.length === 0 && (
                <tr><td colSpan={8} style={{ ...tdStyle, textAlign: 'center', color: '#94a3b8' }}>No critical patients</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* ─── All Patients ──────────────────────────────────────────────────────────── */

function PatientsTab({ patients }) {
  const [sortKey, setSortKey] = useState('composite_score')
  const [sortDir, setSortDir] = useState(-1)
  const [filterText, setFilterText] = useState('')
  const [filterTier, setFilterTier] = useState('all')

  const filtered = patients.filter(p => {
    if (filterTier !== 'all' && p.tier !== filterTier) return false
    if (filterText) {
      const f = filterText.toLowerCase()
      return (p.patient_id || '').toLowerCase().includes(f) || (p.tier || '').toLowerCase().includes(f)
    }
    return true
  })

  const sorted = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    return (av < bv ? -1 : av > bv ? 1 : 0) * sortDir
  })

  const toggleSort = key => {
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

  const tiers = ['all', 'Critical', 'At Risk', 'Adequate', 'Strong']

  return (
    <Card title={`All Patients (${sorted.length})`}>
      <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Filter by Patient ID or Tier…"
          value={filterText}
          onChange={e => setFilterText(e.target.value)}
          style={{ flex: 1, minWidth: 200, padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 13 }}
        />
        <select
          value={filterTier}
          onChange={e => setFilterTier(e.target.value)}
          style={{ padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 13 }}
        >
          {tiers.map(t => <option key={t} value={t}>{t === 'all' ? 'All Tiers' : t}</option>)}
        </select>
      </div>
      <div style={{ overflowX: 'auto', maxHeight: 560, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0 }}>
            <tr>
              {hdr('Patient ID', 'patient_id')}
              {hdr('Composite Score', 'composite_score')}
              {hdr('Tier', 'tier')}
              {hdr('Caregiver', 'caregiver_score')}
              {hdr('Emergency', 'emergency_score')}
              {hdr('Adherence', 'adherence_score')}
              {hdr('Wearable', 'wearable_score')}
              {hdr('IoT Alert', 'alert_score')}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ ...tdStyle, fontWeight: 700, color: '#1e293b' }}>{p.patient_id}</td>
                <td style={{ ...tdStyle, minWidth: 140 }}><ScorePill score={p.composite_score} /></td>
                <td style={tdStyle}><TierBadge tier={p.tier} /></td>
                <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.caregiver_score} /></td>
                <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.emergency_score} /></td>
                <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.adherence_score} /></td>
                <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.wearable_score} /></td>
                <td style={{ ...tdStyle, textAlign: 'center' }}><ScoreCell score={p.alert_score} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length === 0 && (
        <div style={{ textAlign: 'center', color: '#94a3b8', padding: 24 }}>No patients match the current filter.</div>
      )}
    </Card>
  )
}

/* ─── Critical ──────────────────────────────────────────────────────────────── */

const DIMENSION_KEYS = [
  { key: 'adherence_score', label: 'Adherence' },
  { key: 'caregiver_score', label: 'Caregiver' },
  { key: 'emergency_score', label: 'Emergency' },
  { key: 'wearable_score', label: 'Wearable' },
  { key: 'alert_score', label: 'IoT Alert' },
]

function weakestDimension(p) {
  let minKey = null, minVal = Infinity
  for (const { key, label } of DIMENSION_KEYS) {
    const v = p[key]
    if (v != null && Number(v) < minVal) { minVal = Number(v); minKey = label }
  }
  return minKey
}

function CriticalTab({ patients }) {
  const critical = patients.filter(p => p.tier === 'Critical')

  // Radar data: avg per dimension across all critical patients
  const radarData = DIMENSION_KEYS.map(({ key, label }) => {
    const vals = critical.map(p => p[key]).filter(v => v != null)
    const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
    return { dimension: label, avg: parseFloat(avg.toFixed(1)) }
  })

  // Weakest dimension frequency
  const freq = {}
  critical.forEach(p => {
    const w = weakestDimension(p)
    if (w) freq[w] = (freq[w] || 0) + 1
  })
  const freqData = Object.entries(freq)
    .map(([dim, count]) => ({ dim, count }))
    .sort((a, b) => b.count - a.count)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>

      <Card title={`Critical Patients — ${critical.length} total`} span={2}>
        <p style={{ fontSize: 13, color: '#64748b', marginTop: 0, marginBottom: 12 }}>
          Patients with composite score below 40. The weakest dimension for each patient is highlighted.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#fef2f2' }}>
                <th style={{ ...thStyle, color: '#dc2626' }}>Patient ID</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Composite</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Adherence</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Caregiver</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Emergency</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>Wearable</th>
                <th style={{ ...thStyle, color: '#dc2626', textAlign: 'center' }}>IoT Alert</th>
                <th style={{ ...thStyle, color: '#dc2626' }}>Weakest Dimension</th>
              </tr>
            </thead>
            <tbody>
              {critical.map((p, i) => {
                const weak = weakestDimension(p)
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ ...tdStyle, fontWeight: 700, color: '#1e293b' }}>{p.patient_id}</td>
                    <td style={{ ...tdStyle, minWidth: 120 }}><ScorePill score={p.composite_score} /></td>
                    {DIMENSION_KEYS.map(({ key, label }) => {
                      const isWeak = label === weak
                      return (
                        <td key={key} style={{
                          ...tdStyle, textAlign: 'center',
                          background: isWeak ? '#fef2f2' : undefined
                        }}>
                          <ScoreCell score={p[key]} />
                          {isWeak && <span style={{ fontSize: 10, color: '#ef4444', marginLeft: 2 }}>▼</span>}
                        </td>
                      )
                    })}
                    <td style={tdStyle}>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 8,
                        background: '#fef2f2', color: '#ef4444', fontWeight: 600, fontSize: 11
                      }}>{weak || '--'}</span>
                    </td>
                  </tr>
                )
              })}
              {critical.length === 0 && (
                <tr><td colSpan={8} style={{ ...tdStyle, textAlign: 'center', color: '#94a3b8' }}>No critical patients</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Radar chart for average critical scores */}
      <Card title="Average Dimension Scores — Critical Cohort">
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={100}>
            <PolarGrid />
            <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 12 }} />
            <Radar
              name="Avg Score"
              dataKey="avg"
              stroke="#ef4444"
              fill="#ef4444"
              fillOpacity={0.25}
              strokeWidth={2}
            />
            <Tooltip formatter={v => `${v}`} />
            <Legend />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      {/* Weakest dimension frequency */}
      <Card title="Weakest Dimension Frequency (Critical Patients)">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={freqData} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis type="category" dataKey="dim" tick={{ fontSize: 12 }} width={75} />
            <Tooltip />
            <Bar dataKey="count" name="Patients" fill="#ef4444" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ─── Dimensions ────────────────────────────────────────────────────────────── */

function DimensionsTab({ data }) {
  if (!data) return null
  const dims = data.dimension_averages || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
      {dims.map((d, i) => {
        const score = d.score != null ? Number(d.score) : null
        const color = score == null ? '#94a3b8' : score >= 80 ? '#10b981' : score >= 60 ? '#3b82f6' : score >= 40 ? '#f59e0b' : '#ef4444'
        return (
          <Card key={i} title={d.dimension}>
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Avg Score</span>
                <span style={{ fontSize: 20, fontWeight: 700, color }}>{score != null ? score.toFixed(1) : '--'}</span>
              </div>
              <div style={{ height: 10, borderRadius: 5, background: '#f1f5f9', overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.min(100, score ?? 0)}%`, height: '100%',
                  background: color, borderRadius: 5, transition: 'width .4s'
                }} />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#475569', marginBottom: 4 }}>
              <span>Weight</span>
              <span style={{ fontWeight: 600, color: '#8b5cf6' }}>{d.weight_pct != null ? `${d.weight_pct}%` : '--'}</span>
            </div>
          </Card>
        )
      })}

      {/* Stacked dimension overview chart */}
      <Card title="Dimension Score Overview" span={3}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={dims} margin={{ left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dimension" tick={{ fontSize: 11 }} />
            <YAxis domain={[0, 100]} />
            <Tooltip formatter={(v, name) => [`${typeof v === 'number' ? v.toFixed(1) : v}`, name]} />
            <Legend />
            <Bar dataKey="score" name="Avg Score" radius={[4, 4, 0, 0]}>
              {dims.map((d, i) => {
                const s = d.score != null ? Number(d.score) : null
                const c = s == null ? '#94a3b8' : s >= 80 ? '#10b981' : s >= 60 ? '#3b82f6' : s >= 40 ? '#f59e0b' : '#ef4444'
                return <Cell key={i} fill={c} />
              })}
            </Bar>
            <Bar dataKey="weight_pct" name="Weight %" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* ─── Definitions ───────────────────────────────────────────────────────────── */

function DefinitionsTab({ data }) {
  if (!data) return <Card>No definitions available.</Card>
  const dims = data.dimensions || []
  const tiers = data.tiers || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 16 }}>

      <Card title={data.dashboard || 'Patient Safety Network'} span={2}>
        <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, marginTop: 0 }}>{data.description}</p>
      </Card>

      {dims.length > 0 && (
        <Card title="Score Dimensions" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={thStyle}>Dimension</th>
                  <th style={{ ...thStyle, textAlign: 'center' }}>Weight</th>
                  <th style={thStyle}>Data Source</th>
                  <th style={thStyle}>Metric</th>
                </tr>
              </thead>
              <tbody>
                {dims.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, color: '#1e293b' }}>{d.name}</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>
                      <span style={{
                        display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                        background: '#8b5cf622', color: '#8b5cf6', fontWeight: 700, fontSize: 12
                      }}>{d.weight}</span>
                    </td>
                    <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>{d.source}</td>
                    <td style={{ ...tdStyle, fontSize: 12, color: '#475569' }}>{d.metric}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tiers.length > 0 && (
        <Card title="Safety Tiers" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={thStyle}>Tier</th>
                  <th style={{ ...thStyle, textAlign: 'center' }}>Score Range</th>
                  <th style={thStyle}>Meaning</th>
                </tr>
              </thead>
              <tbody>
                {tiers.map((t, i) => {
                  const color = TIER_COLORS[t.tier] || '#64748b'
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={tdStyle}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: 14, height: 14, borderRadius: '50%', background: color, flexShrink: 0 }} />
                          <span style={{ fontWeight: 700, color }}>{t.tier}</span>
                        </div>
                      </td>
                      <td style={{ ...tdStyle, textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                          background: color + '22', color, fontWeight: 600, fontSize: 12
                        }}>{t.range}</span>
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569' }}>{t.meaning}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

const thStyle = { padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 12, color: '#475569' }
const tdStyle = { padding: '6px 10px', borderBottom: '1px solid #f1f5f9' }
