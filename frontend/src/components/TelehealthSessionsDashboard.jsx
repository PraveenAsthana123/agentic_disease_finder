import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function QualityBadge({ quality }) {
  const map = { excellent: '#10b981', good: '#3b82f6', fair: '#f59e0b', poor: '#ef4444' }
  const color = map[quality] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{quality || '--'}</span>
  )
}

function SatisfactionBadge({ score }) {
  const color = score >= 4 ? '#10b981' : score >= 3 ? '#3b82f6' : score >= 2 ? '#f59e0b' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{score != null ? `${score}/5` : '--'}</span>
  )
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

export default function TelehealthSessionsDashboard() {
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
          axios.get(`${API_URL}/api/telehealth-sessions/overview`),
          axios.get(`${API_URL}/api/telehealth-sessions/breakdown`),
          axios.get(`${API_URL}/api/telehealth-sessions/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load telehealth data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading telehealth session data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Telehealth session data not available</div>

  const tabs = ['overview', 'sessions', 'providers', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Telehealth Sessions Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Remote patient encounters — {fmt(kpis.total_sessions)} sessions across {fmt(kpis.total_patients)} patients and {fmt(kpis.total_providers)} providers
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b', fontWeight: 600, fontSize: 13
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Sessions" value={kpis.total_sessions} color="#3b82f6" />
              <KPI label="Patients" value={kpis.total_patients} color="#8b5cf6" />
              <KPI label="Providers" value={kpis.total_providers} color="#10b981" />
              <KPI label="Avg Duration" value={kpis.avg_duration} sub="minutes" color="#f59e0b" />
              <KPI label="Avg Satisfaction" value={kpis.avg_satisfaction} sub="out of 5" color="#06b6d4" />
              <KPI label="Tech Issue Rate" value={kpis.tech_issue_rate_pct} sub="%" color="#ef4444" />
            </div>
          </Card>

          {/* Session type distribution - pie */}
          <Card title="Session Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.session_type_distribution || []} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, count }) => `${name}: ${count}`}>
                  {(overview.session_type_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Platform distribution - bar */}
          <Card title="Platform Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.platform_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Connection quality - bar */}
          <Card title="Connection Quality">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.connection_quality_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {(overview.connection_quality_distribution || []).map((entry, i) => {
                    const cmap = { excellent: '#10b981', good: '#3b82f6', fair: '#f59e0b', poor: '#ef4444' }
                    return <Cell key={i} fill={cmap[entry.name] || COLORS[i]} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly trend */}
          <Card title="Monthly Session Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={overview.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" domain={[0, 5]} />
                <Tooltip />
                <Line yAxisId="left" type="monotone" dataKey="sessions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Sessions" />
                <Line yAxisId="right" type="monotone" dataKey="avg_satisfaction" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Avg Satisfaction" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Provider workload */}
          <Card title="Provider Workload" span={1}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.provider_workload || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="provider" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="sessions" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'sessions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Poor connection sessions */}
          {(breakdown.poor_connection_sessions || []).length > 0 && (
            <Card title="Poor Connection Sessions">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#fef2f2' }}>
                      {['Patient', 'Date', 'Type', 'Provider', 'Duration', 'Quality', 'Satisfaction', 'Platform'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#991b1b', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.poor_connection_sessions || []).map((s, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 500 }}>{s.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{s.session_date}</td>
                        <td style={{ padding: '6px 10px' }}>{s.session_type}</td>
                        <td style={{ padding: '6px 10px' }}>{s.provider_name}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(s.duration_minutes)} min</td>
                        <td style={{ padding: '6px 10px' }}><QualityBadge quality={s.connection_quality} /></td>
                        <td style={{ padding: '6px 10px' }}><SatisfactionBadge score={s.patient_satisfaction} /></td>
                        <td style={{ padding: '6px 10px' }}>{s.platform}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Low satisfaction sessions */}
          {(breakdown.low_satisfaction_sessions || []).length > 0 && (
            <Card title="Low Satisfaction Sessions (Score ≤ 2)">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#fffbeb' }}>
                      {['Patient', 'Date', 'Type', 'Provider', 'Duration', 'Quality', 'Satisfaction', 'Tech Issues'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#92400e', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.low_satisfaction_sessions || []).map((s, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 500 }}>{s.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{s.session_date}</td>
                        <td style={{ padding: '6px 10px' }}>{s.session_type}</td>
                        <td style={{ padding: '6px 10px' }}>{s.provider_name}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(s.duration_minutes)} min</td>
                        <td style={{ padding: '6px 10px' }}><QualityBadge quality={s.connection_quality} /></td>
                        <td style={{ padding: '6px 10px' }}><SatisfactionBadge score={s.patient_satisfaction} /></td>
                        <td style={{ padding: '6px 10px' }}>{s.technical_issues ? 'Yes' : 'No'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Recent sessions */}
          <Card title="Recent Sessions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Date', 'Type', 'Provider', 'Duration', 'Quality', 'Satisfaction', 'Platform'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_sessions || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{s.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{s.session_date}</td>
                      <td style={{ padding: '6px 10px' }}>{s.session_type}</td>
                      <td style={{ padding: '6px 10px' }}>{s.provider_name}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(s.duration_minutes)} min</td>
                      <td style={{ padding: '6px 10px' }}><QualityBadge quality={s.connection_quality} /></td>
                      <td style={{ padding: '6px 10px' }}><SatisfactionBadge score={s.patient_satisfaction} /></td>
                      <td style={{ padding: '6px 10px' }}>{s.platform}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Per-patient summary */}
          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Sessions', 'Avg Duration', 'Avg Satisfaction', 'Tech Issues', 'Primary Platform'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{p.sessions}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(p.avg_duration)} min</td>
                      <td style={{ padding: '6px 10px' }}><SatisfactionBadge score={p.avg_satisfaction} /></td>
                      <td style={{ padding: '6px 10px' }}>{p.tech_issues}</td>
                      <td style={{ padding: '6px 10px' }}>{p.most_used_platform}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'providers' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Provider by session type */}
          <Card title="Provider Session Breakdown by Type">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Provider', 'Session Type', 'Count'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.provider_by_type || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{r.provider}</td>
                      <td style={{ padding: '6px 10px' }}>{r.session_type}</td>
                      <td style={{ padding: '6px 10px' }}>{r.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Provider workload detail from overview */}
          <Card title="Provider Performance Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Provider', 'Sessions', 'Avg Duration (min)', 'Avg Satisfaction'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.provider_workload || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{p.provider}</td>
                      <td style={{ padding: '6px 10px' }}>{p.sessions}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(p.avg_duration)}</td>
                      <td style={{ padding: '6px 10px' }}><SatisfactionBadge score={p.avg_satisfaction} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Session types */}
          <Card title="Session Types">
            {Object.entries(defs.session_types || {}).map(([k, v]) => (
              <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{k}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </Card>

          {/* Connection quality levels */}
          <Card title="Connection Quality Levels">
            {Object.entries(defs.connection_quality_levels || {}).map(([k, v]) => (
              <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{k}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </Card>

          {/* Platforms */}
          <Card title="Platforms">
            {Object.entries(defs.platforms || {}).map(([k, v]) => (
              <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{k}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </Card>

          {/* Field descriptions */}
          <Card title="Field Descriptions">
            {Object.entries(defs.field_descriptions || {}).map(([k, v]) => (
              <div key={k} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{k}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </Card>

          {/* Clinical notes */}
          <Card title="Clinical Notes">
            {(defs.clinical_notes || []).map((note, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8, fontSize: 12, color: '#475569' }}>
                {note}
              </div>
            ))}
          </Card>

          {/* Glossary */}
          <Card title="Glossary">
            {(defs.glossary || []).map((g, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{g.definition}</div>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
