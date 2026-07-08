import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'

const CATEGORY_COLORS = [
  '#8b5cf6', '#3b82f6', '#22c55e', '#f59e0b', '#ec4899',
  '#ef4444', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]

const STATUS_COLORS = {
  active: '#3b82f6',
  completed: '#22c55e',
  paused: '#eab308',
  cancelled: '#ef4444',
  pending: '#94a3b8'
}

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

function progressColor(pct) {
  if (pct >= 76) return '#22c55e'
  if (pct >= 51) return '#3b82f6'
  if (pct >= 26) return '#eab308'
  return '#ef4444'
}

export default function RehabPlanDashboard() {
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
          axios.get(`${API_URL}/rehab-plan/overview`),
          axios.get(`${API_URL}/rehab-plan/breakdown`),
          axios.get(`${API_URL}/rehab-plan/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Rehab Plan data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'categories', label: 'Categories' },
    { id: 'patients', label: 'Patients' },
    { id: 'updates', label: 'Updates' },
    { id: 'glossary', label: 'Glossary' },
  ]

  /* Overview data prep */
  const kpis = overview?.kpis || {}
  const categoryDistData = overview?.category_distribution || []
  const statusDistData = overview?.status_distribution || []
  const progressTrendData = overview?.progress_trend || []
  const completionRateByCategoryData = overview?.completion_rate_by_category || []

  /* Breakdown data prep */
  const sessionAdherenceData = breakdown?.session_adherence || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Rehab Plan Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Occupational Therapist rehab plan management — goals, progress, session adherence, and outcomes
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

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Cards */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Plans" value={fmt(kpis.total_plans)} />
              <KPI label="Active Plans" value={fmt(kpis.active_plans)} color="#3b82f6" />
              <KPI label="Completed Plans" value={fmt(kpis.completed_plans)} color="#22c55e" />
              <KPI label="Avg Progress %" value={fmt(kpis.avg_progress_pct)} color={progressColor(kpis.avg_progress_pct || 0)} />
              <KPI label="Session Completion Rate" value={fmt(kpis.avg_sessions_completion_rate)} color="#8b5cf6" />
            </div>
          </Card>

          {/* Category Distribution Pie */}
          <Card title="Category Distribution">
            {categoryDistData.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data available</div>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={categoryDistData} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={80} label={({ category, count }) => `${category}: ${count}`} labelLine={false} fontSize={10}>
                      {categoryDistData.map((d, i) => <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginTop: 8 }}>
                  {categoryDistData.map((d, i) => (
                    <span key={i} style={{ fontSize: 11, color: '#475569' }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: CATEGORY_COLORS[i % CATEGORY_COLORS.length], marginRight: 4 }} />
                      {d.category}
                    </span>
                  ))}
                </div>
              </>
            )}
          </Card>

          {/* Status Distribution Pie */}
          <Card title="Status Distribution">
            {statusDistData.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data available</div>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={statusDistData} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`} labelLine={false} fontSize={10}>
                      {statusDistData.map((d, i) => <Cell key={i} fill={STATUS_COLORS[d.status?.toLowerCase()] || CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginTop: 8 }}>
                  {statusDistData.map((d, i) => (
                    <span key={i} style={{ fontSize: 11, color: '#475569' }}>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: STATUS_COLORS[d.status?.toLowerCase()] || CATEGORY_COLORS[i % CATEGORY_COLORS.length], marginRight: 4 }} />
                      {d.status}
                    </span>
                  ))}
                </div>
              </>
            )}
          </Card>

          {/* Progress Trend Line Chart */}
          <Card title="Progress Trend">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={progressTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="avg_progress" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Avg Progress %" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Categories */}
      {tab === 'categories' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Completion Rate by Category Bar Chart */}
          <Card title="Completion Rate by Category" span={2}>
            {completionRateByCategoryData.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data available</div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={completionRateByCategoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={(v) => `${fmt(v)}%`} />
                  <Bar dataKey="rate" name="Completion Rate %">
                    {completionRateByCategoryData.map((d, i) => <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Category Distribution Details */}
          <Card title="Category Distribution Details" span={2}>
            {categoryDistData.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data available</div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                {categoryDistData.map((d, i) => {
                  const color = CATEGORY_COLORS[i % CATEGORY_COLORS.length]
                  const matchingRate = completionRateByCategoryData.find(r => r.category === d.category)
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: `3px solid ${color}` }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color: '#1e293b' }}>{d.category}</div>
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{fmt(d.count)} plans</div>
                      {matchingRate && (
                        <div style={{ fontSize: 12, color: progressColor(matchingRate.rate || 0), fontWeight: 600, marginTop: 2 }}>
                          {fmt(matchingRate.rate)}% completion
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 3: Patients */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Patient Summary Table */}
          <Card title="Patient Summary">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total Plans</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Active</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Completed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Progress %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12 }}>{fmt(row.total_plans)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12, color: '#3b82f6', fontWeight: 600 }}>{fmt(row.active)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 12, color: '#22c55e', fontWeight: 600 }}>{fmt(row.completed)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 600, color: progressColor(row.avg_progress || 0) }}>
                          {fmt(row.avg_progress)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Session Adherence Bar Chart */}
          <Card title="Session Adherence by Patient">
            {sessionAdherenceData.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No data available</div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={sessionAdherenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={(v, name) => name === 'rate' ? `${fmt(v)}%` : fmt(v)} />
                  <Bar dataKey="planned" name="Planned" fill="#94a3b8" />
                  <Bar dataKey="completed" name="Completed" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>
        </div>
      )}

      {/* Tab 4: Updates */}
      {tab === 'updates' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Recent Updates Table */}
          <Card title="Recent Updates">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Goal Description</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Progress %</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Last Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_updates || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 12, color: '#475569' }}>{row.goal_category || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.goal_description || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                          background: (STATUS_COLORS[row.status?.toLowerCase()] || '#94a3b8') + '22',
                          color: STATUS_COLORS[row.status?.toLowerCase()] || '#94a3b8'
                        }}>{row.status || '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 600, color: progressColor(row.progress_pct || 0) }}>
                          {fmt(row.progress_pct)}
                        </span>
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{row.last_updated || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Upcoming Targets Table */}
          <Card title="Upcoming Targets">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Goal Description</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Target Date</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Progress %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.upcoming_targets || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 12, color: '#475569' }}>{row.goal_category || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.goal_description || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{row.target_date || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{ fontWeight: 600, color: progressColor(row.progress_pct || 0) }}>
                          {fmt(row.progress_pct)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 5: Glossary */}
      {tab === 'glossary' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Metrics Definitions */}
          {defs.metrics && defs.metrics.length > 0 && (
            <Card title="Metrics Definitions">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.metrics.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.name || d.metric}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.description || d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Categories */}
          {defs.categories && defs.categories.length > 0 && (
            <Card title="Goal Categories">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.categories.map((c, i) => {
                  const color = CATEGORY_COLORS[i % CATEGORY_COLORS.length]
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                        <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 5, background: color }} />
                        <h4 style={{ margin: 0, fontSize: 13, color: '#334155' }}>{c.name || c.category}</h4>
                      </div>
                      <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{c.description}</p>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Statuses */}
          {defs.statuses && defs.statuses.length > 0 && (
            <Card title="Plan Statuses">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                {defs.statuses.map((s, i) => {
                  const color = STATUS_COLORS[s.status?.toLowerCase()] || STATUS_COLORS[s.name?.toLowerCase()] || '#94a3b8'
                  return (
                    <div key={i} style={{ textAlign: 'center', padding: '8px 12px', background: color + '15', borderRadius: 8, border: `1px solid ${color}33` }}>
                      <div style={{ fontSize: 14, fontWeight: 700, color }}>{s.name || s.status}</div>
                      <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.description}</div>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.glossary.map((d, i) => (
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
      )}
    </div>
  )
}
