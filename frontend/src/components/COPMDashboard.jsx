import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const LEVEL_COLORS = {
  high: '#22c55e', moderate: '#3b82f6',
  low: '#eab308', very_low: '#ef4444'
}
const LEVEL_LABELS = {
  high: 'High', moderate: 'Moderate',
  low: 'Low', very_low: 'Very Low'
}
const DOMAIN_COLORS = { self_care: '#3b82f6', productivity: '#22c55e', leisure: '#f97316' }
const DOMAIN_LABELS = { self_care: 'Self-Care', productivity: 'Productivity', leisure: 'Leisure' }

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function LevelBadge({ level }) {
  const color = LEVEL_COLORS[level] || '#94a3b8'
  const label = LEVEL_LABELS[level] || level || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function ChangeBadge({ change }) {
  const significant = Math.abs(change) >= 2
  const color = change > 0 ? '#22c55e' : change < 0 ? '#ef4444' : '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color,
      border: significant ? `1px solid ${color}` : 'none'
    }}>
      {change > 0 ? '+' : ''}{fmt(change)}
      {significant && ' *'}
    </span>
  )
}

function ScoreBar({ score, max, label }) {
  const pct = Math.round(score / max * 100)
  const color = pct >= 80 ? '#22c55e' : pct >= 60 ? '#3b82f6' : pct >= 40 ? '#eab308' : '#ef4444'
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#475569' }}>{label}</span>
        <span style={{ fontWeight: 600 }}>{score}/10</span>
      </div>
      <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: '100%' }} />
      </div>
    </div>
  )
}

export default function COPMDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/copm-dashboard/overview`),
          axios.get(`${API_URL}/api/copm-dashboard/breakdown`),
          axios.get(`${API_URL}/api/copm-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading COPM data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'domains', label: 'Domains & Problems' },
    { id: 'change', label: 'Change Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const distData = overview?.performance_distribution
    ? Object.entries(overview.performance_distribution).map(([k, v]) => ({
        name: LEVEL_LABELS[k] || k, value: v, color: LEVEL_COLORS[k] || '#94a3b8'
      }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>COPM Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Canadian Occupational Performance Measure — patient-centred outcome assessment (scores 1-10)
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Avg Performance" value={fmt(overview.avg_performance)} sub="scale 1-10" />
              <KPI label="Avg Satisfaction" value={fmt(overview.avg_satisfaction)} sub="scale 1-10" />
              <KPI label="Avg Perf Change" value={`+${fmt(overview.avg_perf_change)}`} color={overview.avg_perf_change >= 2 ? '#22c55e' : '#3b82f6'} />
              <KPI label="Clinically Significant" value={overview.clinically_significant_count} sub={`${fmt(overview.clinically_significant_pct)}% of patients`} color="#22c55e" />
              <KPI label="Avg Sat Change" value={`+${fmt(overview.avg_sat_change)}`} color={overview.avg_sat_change >= 2 ? '#22c55e' : '#3b82f6'} />
            </div>
          </Card>

          {/* Performance Distribution Pie */}
          <Card title="Performance Level Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {distData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {distData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Performance vs Satisfaction per patient */}
          <Card title="Performance vs Satisfaction by Patient" span={2}>
            {overview.patient_summary && (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.patient_summary}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis domain={[0, 10]} />
                  <Tooltip />
                  <Bar dataKey="perf_initial" fill="#3b82f6" name="Performance" />
                  <Bar dataKey="sat_initial" fill="#22c55e" name="Satisfaction" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Patient Summary Table */}
          <Card title="Per-Patient COPM Scores" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Perf</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sat</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Perf Change</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sat Change</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Level</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{p.age || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700 }}>{fmt(p.perf_initial)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.sat_initial)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ChangeBadge change={p.perf_change} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ChangeBadge change={p.sat_change} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><LevelBadge level={p.level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'domains' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Domain Summary */}
          <Card title="Domain Averages — Performance & Satisfaction" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.domain_summary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 10]} />
                <Tooltip formatter={v => `${v}/10`} />
                <Bar dataKey="avg_performance" fill="#3b82f6" name="Performance" />
                <Bar dataKey="avg_satisfaction" fill="#22c55e" name="Satisfaction" />
                <Bar dataKey="avg_change" fill="#f97316" name="Change" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Problem-Level Heatmap */}
          <Card title="Problem Scores (weakest first)" span={2}>
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={breakdown.problem_heatmap} layout="vertical" margin={{ left: 220 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 10]} />
                <YAxis type="category" dataKey="label" width={210} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/10`} />
                <Bar dataKey="avg_performance" fill="#3b82f6" name="Performance" />
                <Bar dataKey="avg_satisfaction" fill="#22c55e" name="Satisfaction" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Domain Summary Cards */}
          {(breakdown.domain_summary || []).map(d => (
            <Card key={d.domain} title={d.label}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 12 }}>
                <KPI label="Avg Performance" value={fmt(d.avg_performance)} sub="/10" />
                <KPI label="Avg Satisfaction" value={fmt(d.avg_satisfaction)} sub="/10" />
                <KPI label="Avg Change" value={`+${fmt(d.avg_change)}`} color={d.avg_change >= 2 ? '#22c55e' : '#3b82f6'} />
              </div>
              {(breakdown.problem_heatmap || []).filter(p => p.domain === d.domain).map(p => (
                <ScoreBar key={p.id} score={p.avg_performance} max={10} label={p.label} />
              ))}
            </Card>
          ))}
        </div>
      )}

      {tab === 'change' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Change Analysis Bar Chart */}
          <Card title="Performance Change by Patient" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.change_analysis}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="perf_change" name="Performance Change">
                  {(breakdown.change_analysis || []).map((d, i) => (
                    <Cell key={i} fill={d.significant ? '#22c55e' : '#94a3b8'} />
                  ))}
                </Bar>
                <Bar dataKey="sat_change" name="Satisfaction Change">
                  {(breakdown.change_analysis || []).map((d, i) => (
                    <Cell key={i} fill={d.significant ? '#3b82f6' : '#cbd5e1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p style={{ fontSize: 11, color: '#64748b', marginTop: 8 }}>
              * Colored bars = clinically significant change (2+ points). Grey = not significant.
            </p>
          </Card>

          {/* Change Table */}
          <Card title="Change Summary" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Perf Change</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sat Change</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Significant</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.change_analysis || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ChangeBadge change={p.perf_change} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ChangeBadge change={p.sat_change} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        {p.significant
                          ? <span style={{ color: '#22c55e', fontWeight: 600 }}>Yes</span>
                          : <span style={{ color: '#94a3b8' }}>No</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Problem Detail">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_problems || {}).map(([pid, detail]) => (
                <div key={pid} style={{ marginBottom: 20, padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 14 }}>{pid}</span>
                      <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                        Perf: {fmt(detail.mean_performance)} | Sat: {fmt(detail.mean_satisfaction)}
                        | Change: +{fmt(detail.performance_change)}
                      </span>
                    </div>
                    {detail.clinically_significant && (
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11,
                        fontWeight: 600, background: '#22c55e22', color: '#22c55e'
                      }}>Clinically Significant</span>
                    )}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                    {(detail.problems || []).map(prob => (
                      <div key={prob.id} style={{ marginBottom: 4 }}>
                        <ScoreBar score={prob.initial_performance} max={10} label={prob.label} />
                        <div style={{ fontSize: 10, color: '#64748b', marginTop: -2, paddingLeft: 2 }}>
                          <span style={{ color: DOMAIN_COLORS[prob.domain] || '#94a3b8' }}>{prob.domain_label}</span>
                          {' | Sat: '}{prob.initial_satisfaction}
                          {' | '}<ChangeBadge change={prob.performance_change} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <Card title={defs.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(defs.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {defs.references && (
            <div style={{ marginTop: 16, padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
              <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>References</h4>
              <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                {defs.references.map((ref, i) => <li key={i} style={{ marginBottom: 4 }}>{ref}</li>)}
              </ul>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
