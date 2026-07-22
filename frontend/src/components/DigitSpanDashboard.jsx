import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const TIER_COLORS = {
  'Superior': '#16a34a', 'Average': '#3b82f6', 'Low Average': '#eab308',
  'Borderline': '#f97316', 'Impaired': '#ef4444'
}
const COND_COLORS = { Forward: '#3b82f6', Backward: '#f97316', Sequencing: '#8b5cf6' }

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
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{tier}</span>
  )
}

export default function DigitSpanDashboard() {
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
          axios.get(`${API_URL}/api/digit-span-dashboard/overview`),
          axios.get(`${API_URL}/api/digit-span-dashboard/breakdown`),
          axios.get(`${API_URL}/api/digit-span-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Digit Span data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'conditions', label: 'Conditions & Spans' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const distData = overview?.performance_distribution
    ? Object.entries(overview.performance_distribution).map(([k, v]) => ({
        name: k, value: v, color: TIER_COLORS[k] || '#94a3b8'
      }))
    : []

  const alertCount = (overview?.alerts || []).length

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Digit Span Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Working Memory Assessment — Forward / Backward / Sequencing (scaled score mean=10, SD=3)
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
              <KPI label="Total Assessments" value={fmt(overview.total_assessments)} />
              <KPI label="Unique Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Mean Scaled Score" value={fmt(overview.avg_scaled_score)} sub="mean=10, SD=3" />
              <KPI label="Score Range" value={`${fmt(overview.min_scaled)}-${fmt(overview.max_scaled)}`} sub="out of 19" />
              <KPI label="Active Alerts" value={fmt(alertCount)} color={alertCount > 0 ? '#ef4444' : '#22c55e'} sub={alertCount > 0 ? 'asymmetry/impairment' : 'none'} />
            </div>
          </Card>

          {/* Performance Distribution Pie */}
          <Card title="Performance Tier Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
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

          {/* Per-Patient Condition Bars */}
          <Card title="Per-Patient Condition Scores (Forward / Backward / Sequencing)" span={2}>
            {breakdown?.patient_condition_comparison && (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={breakdown.patient_condition_comparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis domain={[0, 16]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Forward" fill={COND_COLORS.Forward} name="Forward" />
                  <Bar dataKey="Backward" fill={COND_COLORS.Backward} name="Backward" />
                  <Bar dataKey="Sequencing" fill={COND_COLORS.Sequencing} name="Sequencing" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Alerts */}
          {alertCount > 0 && (
            <Card title="Active Alerts" span={3}>
              <div style={{ maxHeight: 200, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Scaled</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Tier</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Alert</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.alerts || []).map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{a.patient_id}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>{a.scaled_score}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><TierBadge tier={a.tier} /></td>
                        <td style={{ padding: '6px 8px', color: '#b91c1c', fontSize: 11 }}>{a.alert}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient Scaled Scores (lowest first)" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Scaled</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Raw Total</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Fwd</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Bwd</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seq</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700 }}>{p.scaled_score}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.total_raw}/48</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.forward_raw}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.backward_raw}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.sequencing_raw}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><TierBadge tier={p.tier} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'conditions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Condition Profile (Group Average) */}
          <Card title="Condition Profile — Group Average Raw Scores" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.condition_profile} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 16]} />
                <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => `${v}/16`} />
                <Bar dataKey="avg_raw" radius={[0, 4, 4, 0]}>
                  {(breakdown.condition_profile || []).map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              Expected pattern: Forward &gt; Backward &gt; Sequencing
            </div>
          </Card>

          {/* Forward-Backward Asymmetry */}
          <Card title="Forward-Backward Asymmetry (clinically significant if |diff| >= 5)" span={2}>
            {breakdown.asymmetry_data && (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={breakdown.asymmetry_data} margin={{ left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis domain={[-8, 12]} />
                  <Tooltip formatter={(v, name) => [v, name === 'difference' ? 'Fwd-Bwd Diff' : name]} />
                  <Bar dataKey="difference" name="Fwd-Bwd Diff">
                    {(breakdown.asymmetry_data || []).map((d, i) => (
                      <Cell key={i} fill={d.clinically_significant ? '#ef4444' : '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Monthly Trend */}
          {breakdown.trend && breakdown.trend.length > 0 && (
            <Card title="Monthly Scaled Score Trend" span={2}>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={breakdown.trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis domain={[1, 19]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="avg_scaled" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Avg Scaled Score" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Span Length Profile */}
          <Card title="Average Longest Span by Condition" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.condition_profile} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 9]} />
                <YAxis type="category" dataKey="label" width={110} tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => `${v} digits`} />
                <Bar dataKey="avg_span" radius={[0, 4, 4, 0]} name="Avg Span">
                  {(breakdown.condition_profile || []).map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              Typical adult forward span: 6-7 digits | Backward: 4-5 | Sequencing: 4-5
            </div>
          </Card>
        </div>
      )}

      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Assessment History & Condition Profiles">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_history || {}).map(([pid, hist]) => {
                const latest = hist[hist.length - 1]
                return (
                  <div key={pid} style={{ marginBottom: 20, padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 14 }}>{pid}</span>
                        <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                          Scaled: {latest.scaled_score}/19 | Raw: {latest.total_raw}/48 |
                          Fwd: {latest.forward_raw} | Bwd: {latest.backward_raw} | Seq: {latest.sequencing_raw}
                        </span>
                      </div>
                      <TierBadge tier={latest.tier} />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                      {[
                        { label: 'Forward', val: latest.forward_raw, max: 16, color: COND_COLORS.Forward, span: latest.forward_span },
                        { label: 'Backward', val: latest.backward_raw, max: 16, color: COND_COLORS.Backward, span: latest.backward_span },
                        { label: 'Sequencing', val: latest.sequencing_raw, max: 16, color: COND_COLORS.Sequencing, span: latest.sequencing_span },
                      ].map(c => (
                        <div key={c.label} style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>{c.label}</div>
                          <div style={{ fontSize: 20, fontWeight: 700, color: c.color }}>{c.val}/{c.max}</div>
                          <div style={{ fontSize: 10, color: '#94a3b8' }}>span: {c.span}</div>
                          <div style={{
                            height: 4, borderRadius: 2, marginTop: 4,
                            background: '#f1f5f9', position: 'relative'
                          }}>
                            <div style={{
                              width: `${Math.round(c.val / c.max * 100)}%`,
                              background: c.color, borderRadius: 2, height: '100%'
                            }} />
                          </div>
                        </div>
                      ))}
                    </div>
                    {hist.length > 1 && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
                        {hist.length} assessments | First: {hist[0].date} | Latest: {latest.date}
                      </div>
                    )}
                  </div>
                )
              })}
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
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
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
