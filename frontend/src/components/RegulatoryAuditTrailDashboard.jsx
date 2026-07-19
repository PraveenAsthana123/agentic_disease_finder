import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const CATEGORY_COLORS = {
  Clinical: '#ef4444',
  Quality: '#f59e0b',
  Administrative: '#3b82f6',
  Regulatory: '#8b5cf6',
  Technical: '#10b981'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function CategoryBadge({ category }) {
  const color = CATEGORY_COLORS[category] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(category || '').replace(/_/g, ' ')}</span>
  )
}

function ActionBadge({ action }) {
  const isAlert = action === 'CAPA opened' || action === 'Deviation logged'
  const color = isAlert ? '#ef4444' : '#3b82f6'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{action || '--'}</span>
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

export default function RegulatoryAuditTrailDashboard() {
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
          axios.get(`${API_URL}/regulatory-audit-trail/overview`),
          axios.get(`${API_URL}/regulatory-audit-trail/breakdown`),
          axios.get(`${API_URL}/regulatory-audit-trail/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load regulatory audit trail data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading regulatory audit trail data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Regulatory audit trail data not available</div>

  const tabs = ['overview', 'submissions', 'actors', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Regulatory Audit Trail Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        21 CFR Part 11 compliance audit analytics — {fmt(kpis.total_actions)} actions across {fmt(kpis.total_submissions)} submissions
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Actions" value={kpis.total_actions} />
              <KPI label="Submissions" value={kpis.total_submissions} />
              <KPI label="Actors" value={kpis.total_actors} />
              <KPI label="Categories" value={kpis.total_categories} />
              <KPI label="Documents" value={kpis.total_documents} color="#3b82f6" />
            </div>
          </Card>

          <Card title="Category Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={overview.category_distribution || []} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={90} label={({ category, count }) => `${category} (${count})`}>
                  {(overview.category_distribution || []).map((entry, i) => <Cell key={i} fill={CATEGORY_COLORS[entry.category] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Action Type Breakdown" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.action_breakdown || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="action" type="category" tick={{ fontSize: 11 }} width={160} />
                <Tooltip />
                <Bar dataKey="count" name="Actions" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Actor Activity" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.actor_activity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="actor" tick={{ fontSize: 10, angle: -30 }} height={60} interval={0} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Actions" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Timeline" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={overview.monthly_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Actions" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* CAPA/Deviation Alerts */}
          {(breakdown?.alerts || []).length > 0 && (
            <Card title="CAPA & Deviation Alerts" span={4}>
              <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12, marginBottom: 8 }}>
                <span style={{ color: '#991b1b', fontWeight: 600, fontSize: 13 }}>
                  {breakdown.alerts.length} alert(s) requiring attention
                </span>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#fef2f2' }}>
                      {['Submission', 'Action', 'Actor', 'Timestamp', 'Document', 'Category'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#991b1b', borderBottom: '1px solid #fecaca', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.alerts.map((a, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 500 }}>{a.submission_id}</td>
                        <td style={{ padding: '8px 10px' }}><ActionBadge action={a.action} /></td>
                        <td style={{ padding: '8px 10px' }}>{a.actor}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{a.timestamp}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12 }}>{a.document_ref}</td>
                        <td style={{ padding: '8px 10px' }}><CategoryBadge category={a.category} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Submissions tab */}
      {tab === 'submissions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Per-Submission Summary" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Submission ID', 'Actions', 'Actors', 'Categories', 'First Action', 'Last Action'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.per_submission || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{s.submission_id}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: Math.min(s.action_count * 10, 100), height: 8, borderRadius: 4, background: '#3b82f6' }} />
                          <span>{s.action_count}</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 10px' }}>{s.actor_count}</td>
                      <td style={{ padding: '8px 10px' }}>{s.category_count}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{s.first_action}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{s.last_action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Actions" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Submission', 'Action', 'Actor', 'Timestamp', 'Document', 'Category'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.recent_actions || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 500 }}>{a.submission_id}</td>
                      <td style={{ padding: '8px 10px' }}><ActionBadge action={a.action} /></td>
                      <td style={{ padding: '8px 10px' }}>{a.actor}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{a.timestamp}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12 }}>{a.document_ref}</td>
                      <td style={{ padding: '8px 10px' }}><CategoryBadge category={a.category} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Actors tab */}
      {tab === 'actors' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Per-Actor Summary" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Actor', 'Actions', 'Submissions', 'Action Types', 'Last Activity'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.per_actor || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.actor}</td>
                      <td style={{ padding: '8px 10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ width: Math.min(a.action_count * 8, 100), height: 8, borderRadius: 4, background: '#8b5cf6' }} />
                          <span>{a.action_count}</span>
                        </div>
                      </td>
                      <td style={{ padding: '8px 10px' }}>{a.submission_count}</td>
                      <td style={{ padding: '8px 10px' }}>{a.action_types}</td>
                      <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{a.last_activity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Action Types" span={1}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs?.action_types || []).map((a, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{a.action}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{a.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Categories" span={1}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs?.categories || []).map((c, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}><CategoryBadge category={c.category} /></div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{c.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary" span={1}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs?.glossary || []).map((g, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: 12, borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{g.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes" span={1}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs?.clinical_notes || []).map((n, i) => (
                <div key={i} style={{ background: '#f8fafc', padding: 12, borderRadius: 8, fontSize: 12, color: '#475569' }}>
                  {n}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
