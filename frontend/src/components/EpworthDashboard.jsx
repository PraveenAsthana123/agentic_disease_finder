import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const SEVERITY_COLORS = { normal: '#22c55e', mild: '#eab308', moderate: '#f97316', severe: '#ef4444' }
const COLORS = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#8b5cf6', '#3b82f6']

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

function RiskBadge({ level }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (SEVERITY_COLORS[level] || '#94a3b8') + '22', color: SEVERITY_COLORS[level] || '#64748b'
    }}>{(level || 'unknown').toUpperCase()}</span>
  )
}

export default function EpworthDashboard() {
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
          axios.get(`${API_URL}/epworth-dashboard/overview`),
          axios.get(`${API_URL}/epworth-dashboard/breakdown`),
          axios.get(`${API_URL}/epworth-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Epworth Sleepiness Scale data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const severityData = overview?.severity_distribution
    ? Object.entries(overview.severity_distribution).map(([k, v]) => ({ name: k, value: v, color: SEVERITY_COLORS[k] || '#94a3b8' }))
    : []

  const ITEM_LABELS = [
    'Sitting and reading',
    'Watching TV',
    'Sitting inactive in a public place',
    'Passenger in a car for an hour',
    'Lying down to rest in the afternoon',
    'Sitting and talking to someone',
    'Sitting quietly after lunch (no alcohol)',
    'In a car, stopped in traffic'
  ]

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Epworth Sleepiness Scale Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          ESS — 8-item self-report measure of daytime sleepiness (score 0-24)
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #ef4444' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#ef4444' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Assessments" value={fmt(overview.total_assessments)} />
              <KPI label="Unique Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Avg Score" value={fmt(overview.avg_score)} sub="range 0-24" />
              <KPI label="Excessive Sleepiness Rate" value={`${fmt(overview.excessive_sleepiness_rate_pct)}%`}
                color={overview.excessive_sleepiness_rate_pct > 50 ? '#ef4444' : overview.excessive_sleepiness_rate_pct > 25 ? '#f97316' : '#22c55e'} />
            </div>
          </Card>

          {/* Severity Distribution Pie */}
          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={severityData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {severityData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {severityData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Patient Summary Table */}
          <Card title="Patient Summary" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Score</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Severity</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Interpretation</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.latest_score}/24</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><RiskBadge level={p.severity} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{p.interpretation}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#94a3b8' }}>{(p.assessed_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'breakdown' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Item Mean Scores */}
          <Card title="Per-Item Mean Scores (8 Situations)" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(breakdown.item_scores || []).map((d, i) => ({ ...d, label: d.label || ITEM_LABELS[i] || `Item ${i + 1}` }))} layout="vertical" margin={{ left: 220 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 3]} />
                <YAxis type="category" dataKey="label" width={210} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => v.toFixed(2)} />
                <Bar dataKey="mean_score" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>
              Scale per item: 0 (would never doze) to 3 (high chance of dozing)
            </div>
          </Card>

          {/* Monthly Trend */}
          <Card title="Monthly Average Score Trend" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown.trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 24]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_score" stroke="#ef4444" strokeWidth={2} name="Avg Score" dot />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Severity Transitions */}
          <Card title="Severity Transitions (patients with 2+ assessments)" span={2}>
            {breakdown.severity_transitions?.length > 0 ? (
              <div style={{ maxHeight: 260, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>First</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}></th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Latest</th>
                      <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>#</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.severity_transitions.map((t, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '4px 8px', fontWeight: 600 }}>{t.patient_id}</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><RiskBadge level={t.first_severity} /></td>
                        <td style={{ padding: '4px 8px', textAlign: 'center', color: '#94a3b8' }}>→</td>
                        <td style={{ padding: '4px 8px', textAlign: 'center' }}><RiskBadge level={t.latest_severity} /></td>
                        <td style={{ padding: '4px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{t.assessments}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No patients with multiple assessments yet</div>
            )}
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <Card title={defs.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(defs.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 180 }}>{d.term}</td>
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
