import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const TIER_COLORS = {
  competent: '#22c55e', motor_risk: '#f97316',
  process_risk: '#8b5cf6', dual_risk: '#ef4444'
}
const TIER_LABELS = {
  competent: 'Competent', motor_risk: 'Motor Risk',
  process_risk: 'Process Risk', dual_risk: 'Dual Risk'
}
const GROUP_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#eab308', '#06b6d4', '#ec4899', '#14b8a6']

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

function TierBadge({ tier }) {
  const color = TIER_COLORS[tier] || '#94a3b8'
  const label = TIER_LABELS[tier] || tier || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function ScoreBar({ score, max, label }) {
  const pct = Math.round(score / max * 100)
  const color = pct >= 75 ? '#22c55e' : pct >= 50 ? '#3b82f6' : pct >= 25 ? '#eab308' : '#ef4444'
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#475569' }}>{label}</span>
        <span style={{ fontWeight: 600 }}>{score}/4</span>
      </div>
      <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8 }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: '100%' }} />
      </div>
    </div>
  )
}

export default function AMPSDashboard() {
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
          axios.get(`${API_URL}/amps-dashboard/overview`),
          axios.get(`${API_URL}/amps-dashboard/breakdown`),
          axios.get(`${API_URL}/amps-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AMPS data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'skills', label: 'Skills & Items' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const distData = overview?.performance_distribution
    ? Object.entries(overview.performance_distribution).map(([k, v]) => ({
        name: TIER_LABELS[k] || k, value: v, color: TIER_COLORS[k] || '#94a3b8'
      }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AMPS Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Assessment of Motor and Process Skills — OT observation-based ADL performance (36 items, Rasch logit scoring)
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
              <KPI label="Avg Motor Logit" value={fmt(overview.avg_motor_logit)} sub={`cutoff: 2.0`} color={overview.avg_motor_logit >= 2.0 ? '#22c55e' : '#ef4444'} />
              <KPI label="Avg Process Logit" value={fmt(overview.avg_process_logit)} sub={`cutoff: 1.0`} color={overview.avg_process_logit >= 1.0 ? '#22c55e' : '#ef4444'} />
              <KPI label="Motor Range" value={`${fmt(overview.min_motor_logit)} to ${fmt(overview.max_motor_logit)}`} />
              <KPI label="Process Range" value={`${fmt(overview.min_process_logit)} to ${fmt(overview.max_process_logit)}`} />
            </div>
          </Card>

          {/* Performance Distribution Pie */}
          <Card title="Performance Distribution">
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

          {/* Motor vs Process Scatter */}
          <Card title="Motor vs Process Logits" span={2}>
            {breakdown?.motor_vs_process && (
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="motor_logit" name="Motor" type="number" tick={{ fontSize: 11 }} label={{ value: 'Motor Logit', position: 'bottom', fontSize: 11 }} />
                  <YAxis dataKey="process_logit" name="Process" type="number" tick={{ fontSize: 11 }} label={{ value: 'Process Logit', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                  <Tooltip formatter={(v, name) => [v.toFixed(1), name]} labelFormatter={() => ''} />
                  <Scatter data={breakdown.motor_vs_process} fill="#3b82f6">
                    {(breakdown.motor_vs_process || []).map((d, i) => (
                      <Cell key={i} fill={TIER_COLORS[d.overall_tier] || '#94a3b8'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient AMPS Scores" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Motor Logit</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Process Logit</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{p.age || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: p.motor_logit >= 2.0 ? '#22c55e' : '#ef4444' }}>{fmt(p.motor_logit)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700, color: p.process_logit >= 1.0 ? '#22c55e' : '#ef4444' }}>{fmt(p.process_logit)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><TierBadge tier={p.overall_tier} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'skills' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Motor Group Summary */}
          <Card title="Motor Skill Groups">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={breakdown.motor_group_summary} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 4]} />
                <YAxis type="category" dataKey="label" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/4`} />
                <Bar dataKey="avg_score" radius={[0, 4, 4, 0]}>
                  {(breakdown.motor_group_summary || []).map((d, i) => (
                    <Cell key={i} fill={GROUP_COLORS[i % GROUP_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Process Group Summary */}
          <Card title="Process Skill Groups">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.process_group_summary} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 4]} />
                <YAxis type="category" dataKey="label" width={130} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/4`} />
                <Bar dataKey="avg_score" radius={[0, 4, 4, 0]}>
                  {(breakdown.process_group_summary || []).map((d, i) => (
                    <Cell key={i} fill={GROUP_COLORS[(i + 4) % GROUP_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Item-Level Heatmap (ranked by severity) */}
          <Card title="All 36 Items (weakest first)" span={2}>
            <ResponsiveContainer width="100%" height={700}>
              <BarChart data={breakdown.item_heatmap} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 4]} />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 10 }} />
                <Tooltip formatter={v => `${v}/4`} />
                <Bar dataKey="avg_score" radius={[0, 4, 4, 0]}>
                  {(breakdown.item_heatmap || []).map((d, i) => {
                    const pct = d.avg_score / 4
                    const color = pct >= 0.75 ? '#22c55e' : pct >= 0.5 ? '#3b82f6' : pct >= 0.25 ? '#eab308' : '#ef4444'
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
            <div style={{ maxHeight: 600, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_items || {}).map(([pid, detail]) => (
                <details key={pid} style={{ marginBottom: 12 }}>
                  <summary style={{ padding: '10px 16px', background: '#f8fafc', borderRadius: 8, cursor: 'pointer' }}>
                    <span style={{ fontWeight: 700, fontSize: 14 }}>{pid}</span>
                    <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                      Motor: {fmt(detail.motor_logit)} | Process: {fmt(detail.process_logit)}
                    </span>
                    <span style={{ marginLeft: 12 }}><TierBadge tier={detail.overall_tier} /></span>
                  </summary>
                  <div style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                      {(detail.items || []).map(item => (
                        <ScoreBar key={item.id} score={item.score} max={4} label={`${item.label} (${item.domain})`} />
                      ))}
                    </div>
                  </div>
                </details>
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
