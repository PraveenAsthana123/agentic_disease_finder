import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const LEVEL_COLORS = {
  independent: '#22c55e', modified_independent: '#3b82f6',
  moderate_dependence: '#eab308', low_moderate: '#f97316',
  substantial: '#ef4444', total_dependence: '#7f1d1d'
}
const LEVEL_LABELS = {
  independent: 'Independent', modified_independent: 'Modified Independent',
  moderate_dependence: 'Moderate Dependence', low_moderate: 'Low-Moderate',
  substantial: 'Substantial', total_dependence: 'Total Dependence'
}
const DOMAIN_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#eab308']

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

function ScoreBar({ score, max, label }) {
  const pct = Math.round(score / max * 100)
  const color = pct >= 80 ? '#22c55e' : pct >= 60 ? '#3b82f6' : pct >= 40 ? '#eab308' : '#ef4444'
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#475569' }}>{label}</span>
        <span style={{ fontWeight: 600 }}>{score}/7</span>
      </div>
      <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: '100%' }} />
      </div>
    </div>
  )
}

export default function FIMDashboard() {
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
          axios.get(`${API_URL}/fim-dashboard/overview`),
          axios.get(`${API_URL}/fim-dashboard/breakdown`),
          axios.get(`${API_URL}/fim-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading FIM data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'domains', label: 'Domains & Items' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const distData = overview?.independence_distribution
    ? Object.entries(overview.independence_distribution).map(([k, v]) => ({
        name: LEVEL_LABELS[k] || k, value: v, color: LEVEL_COLORS[k] || '#94a3b8'
      }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>FIM Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Functional Independence Measure — 18-item functional assessment (score 18-126)
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Avg Total Score" value={fmt(overview.avg_total)} sub="range 18-126" />
              <KPI label="Avg Motor" value={fmt(overview.avg_motor)} sub="range 13-91" />
              <KPI label="Avg Cognitive" value={fmt(overview.avg_cognitive)} sub="range 5-35" />
              <KPI label="Score Range" value={`${fmt(overview.min_total)}-${fmt(overview.max_total)}`} />
            </div>
          </Card>

          {/* Independence Distribution Pie */}
          <Card title="Independence Distribution">
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

          {/* Motor vs Cognitive scatter-like bar */}
          <Card title="Motor vs Cognitive Scores" span={2}>
            {breakdown?.motor_vs_cognitive && (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={breakdown.motor_vs_cognitive}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="motor" fill="#3b82f6" name="Motor" stackId="a" />
                  <Bar dataKey="cognitive" fill="#22c55e" name="Cognitive" stackId="a" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient FIM Scores" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Motor</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Cognitive</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Level</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{p.age || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700 }}>{p.total}/126</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.motor}/91</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.cognitive}/35</td>
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
          {/* Subdomain Summary */}
          <Card title="Subdomain Average Scores" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.subdomain_summary} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 7]} />
                <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/7`} />
                <Bar dataKey="avg_score" radius={[0, 4, 4, 0]}>
                  {(breakdown.subdomain_summary || []).map((d, i) => (
                    <Cell key={i} fill={DOMAIN_COLORS[i % DOMAIN_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Item-Level Heatmap (ranked by severity) */}
          <Card title="Item Scores (weakest first)" span={2}>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={breakdown.item_heatmap} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 7]} />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/7`} />
                <Bar dataKey="avg_score" radius={[0, 4, 4, 0]}>
                  {(breakdown.item_heatmap || []).map((d, i) => {
                    const pct = d.avg_score / 7
                    const color = pct >= 0.8 ? '#22c55e' : pct >= 0.6 ? '#3b82f6' : pct >= 0.4 ? '#eab308' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Item Detail">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_items || {}).map(([pid, detail]) => (
                <div key={pid} style={{ marginBottom: 20, padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 14 }}>{pid}</span>
                      <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                        Total: {detail.total}/126 | Motor: {detail.motor}/91 | Cognitive: {detail.cognitive}/35
                      </span>
                    </div>
                    <LevelBadge level={
                      detail.total >= 109 ? 'independent' :
                      detail.total >= 90 ? 'modified_independent' :
                      detail.total >= 73 ? 'moderate_dependence' :
                      detail.total >= 55 ? 'low_moderate' :
                      detail.total >= 37 ? 'substantial' : 'total_dependence'
                    } />
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                    {(detail.items || []).map(item => (
                      <ScoreBar key={item.id} score={item.score} max={7} label={item.label} />
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
        </Card>
      )}
    </div>
  )
}
