import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const PRIORITY_COLORS = { high: '#ef4444', medium: '#f97316', low: '#22c55e' }
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#94a3b8' }
const CAT_COLORS = {
  functional: '#3b82f6', technology: '#8b5cf6', data: '#14b8a6',
  gap: '#ef4444', architecture: '#f97316', decision_ai: '#ec4899'
}

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function PriorityBadge({ priority }) {
  const bg = PRIORITY_COLORS[priority] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {priority}
    </span>
  )
}

function StatusBadge({ status }) {
  const bg = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {status}
    </span>
  )
}

function CatBadge({ category }) {
  const bg = CAT_COLORS[category] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {category}
    </span>
  )
}

export default function FeatureGapsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/feature-gaps/overview`),
      axios.get(`${API_URL}/api/feature-gaps/breakdown`),
      axios.get(`${API_URL}/api/feature-gaps/definitions`),
    ])
      .then(([o, b, d]) => {
        setOverview(o.data)
        setBreakdown(b.data)
        setDefs(d.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Feature Gaps...</div>
  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#94a3b8' }}>Feature gaps data not available</div>

  const s = overview.summary
  const tabs = ['overview', 'by-category', 'all-gaps', 'recommendations', 'definitions']

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: '#0f172a' }}>Feature Gaps Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        Epilepsy DL Review (50 papers) vs Project Implementation — Gap Analysis
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b',
          }}>
            {t === 'overview' ? 'Overview' : t === 'by-category' ? 'By Category' :
             t === 'all-gaps' ? 'All Gaps' : t === 'recommendations' ? 'Recommendations' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 8 }}>
              <KPI label="Total Gaps" value={s.total_gaps} />
              <KPI label="Built" value={s.built} sub={`${s.built_pct}%`} />
              <KPI label="Categories" value={s.categories} />
              <KPI label="High Priority" value={s.high_priority} />
              <KPI label="Medium Priority" value={s.medium_priority} />
              <KPI label="Low Priority" value={s.low_priority} />
              <KPI label="Papers Reviewed" value={s.review_papers} />
              <KPI label="Topics Covered" value={s.review_covers_count} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Gaps by Category">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={overview.category_distribution} dataKey="value" nameKey="name"
                       cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                    {overview.category_distribution.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Gaps by Priority">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview.priority_distribution} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={70} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {overview.priority_distribution.map((d, i) => (
                      <Cell key={i} fill={PRIORITY_COLORS[d.name] || COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Gap Summary Table">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>#</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Category</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Feature</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Priority</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gap_table.map((g, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{i + 1}</td>
                      <td style={{ padding: '6px 10px' }}><CatBadge category={g.category} /></td>
                      <td style={{ padding: '6px 10px', color: '#334155' }}>{g.feature}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={g.in_project} /></td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}><PriorityBadge priority={g.priority} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ─── By Category Tab ─── */}
      {tab === 'by-category' && breakdown?.per_category && (
        <>
          <Card title="Category Breakdown">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.per_category}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="built" name="Built" fill="#22c55e" stackId="a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="total" name="Total" fill="#e2e8f0" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {breakdown.per_category.map((cat, ci) => (
            <Card key={ci} title={`${cat.category} (${cat.built}/${cat.total} built)`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Feature</th>
                      <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                      <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Priority</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Why / Evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cat.items.map((item, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', color: '#334155', fontWeight: 500 }}>{item.feature}</td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={item.in_project} /></td>
                        <td style={{ padding: '6px 10px', textAlign: 'center' }}><PriorityBadge priority={item.priority} /></td>
                        <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{item.why}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ─── All Gaps Tab ─── */}
      {tab === 'all-gaps' && breakdown?.per_category && (
        <Card title={`All Feature Gaps (${s.total_gaps})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>#</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0' }}>Priority</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Why / Evidence</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Dashboard</th>
                </tr>
              </thead>
              <tbody>
                {(() => {
                  let n = 0
                  return breakdown.per_category.flatMap(cat =>
                    cat.items.map((item, i) => {
                      n++
                      return (
                        <tr key={`${cat.category}-${i}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{n}</td>
                          <td style={{ padding: '6px 10px' }}><CatBadge category={cat.category} /></td>
                          <td style={{ padding: '6px 10px', color: '#334155', fontWeight: 500 }}>{item.feature}</td>
                          <td style={{ padding: '6px 10px', textAlign: 'center' }}><StatusBadge status={item.in_project} /></td>
                          <td style={{ padding: '6px 10px', textAlign: 'center' }}><PriorityBadge priority={item.priority} /></td>
                          <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11, maxWidth: 300 }}>{item.why}</td>
                          <td style={{ padding: '6px 10px', color: '#3b82f6', fontSize: 11 }}>{item.dashboard || '--'}</td>
                        </tr>
                      )
                    })
                  )
                })()}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── Recommendations Tab ─── */}
      {tab === 'recommendations' && breakdown && (
        <>
          <Card title="Review Coverage">
            <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 12px' }}>
              The 50-paper epilepsy DL review covers these topics:
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {(breakdown.review_covers || []).map((topic, i) => (
                <span key={i} style={{
                  background: '#f1f5f9', color: '#334155', borderRadius: 4,
                  padding: '4px 10px', fontSize: 12, fontWeight: 500
                }}>
                  {topic}
                </span>
              ))}
            </div>
          </Card>

          <Card title="Top Recommendations">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(breakdown.top_recommendations || []).map((rec, i) => (
                <li key={i} style={{ padding: '8px 0', fontSize: 13, color: '#334155', lineHeight: 1.5 }}>
                  {rec}
                </li>
              ))}
            </ol>
          </Card>
        </>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Category Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.categories || []).map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}><CatBadge category={c.name} /></td>
                    <td style={{ padding: '6px 10px', color: '#334155' }}>{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Priority Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <tbody>
                {(defs.priority_legend || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', width: 100 }}><PriorityBadge priority={p.level} /></td>
                    <td style={{ padding: '6px 10px', color: '#334155' }}>{p.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <tbody>
                {(defs.status_legend || []).map((st, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', width: 100 }}><StatusBadge status={st.status} /></td>
                    <td style={{ padding: '6px 10px', color: '#334155' }}>{st.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{g.term}</td>
                    <td style={{ padding: '6px 10px', color: '#334155' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ padding: '4px 0', fontSize: 12, color: '#64748b' }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ padding: '4px 0', fontSize: 12, color: '#334155' }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
