import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', 'not-pulled': '#94a3b8', scaffold: '#f97316', planned: '#3b82f6', unknown: '#d1d5db' }

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

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function AiTypeCoverageDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/ai-type-coverage/overview`),
      axios.get(`${API_URL}/ai-type-coverage/breakdown`),
      axios.get(`${API_URL}/ai-type-coverage/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading AI Type Coverage...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'built-types', 'not-pulled', 'coverage-matrix', 'definitions']
  const s = overview.summary || {}

  const filteredTypes = (overview.types_table || []).filter(
    t => filter === 'all' || t.status === filter
  )

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1280, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note}</p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#475569',
          }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <Card><KPI label="Total AI Types" value={s.total_types} /></Card>
            <Card><KPI label="Built" value={s.built} sub="live in platform" /></Card>
            <Card><KPI label="Not Pulled" value={s.not_pulled} sub="source project only" /></Card>
            <Card><KPI label="Scaffold" value={s.scaffold} /></Card>
            <Card><KPI label="Planned" value={s.planned} /></Card>
            <Card><KPI label="Coverage" value={`${s.coverage_pct}%`} sub="built / total" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Built Types by Category">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.category_distribution} layout="vertical" margin={{ left: 130 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="All AI Types">
            <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
              {['all', 'built', 'not-pulled', 'scaffold', 'planned'].map(f => (
                <button key={f} onClick={() => setFilter(f)} style={{
                  padding: '4px 12px', borderRadius: 4, border: 'none', cursor: 'pointer',
                  fontSize: 11, fontWeight: 600,
                  background: filter === f ? '#334155' : '#f1f5f9',
                  color: filter === f ? '#fff' : '#475569',
                }}>{f.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} ({filter === 'all' && f === 'all' ? overview.types_table.length : (overview.types_table || []).filter(t => f === 'all' || t.status === f).length})</button>
              ))}
            </div>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>AI Type</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTypes.map((t, i) => (
                    <tr key={i} style={{ background: t.status === 'built' ? '#f0fdf4' : undefined }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{t.type}</td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                      <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{t.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'built-types' && breakdown && (
        <>
          <div style={{ display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
            <Card><KPI label="Built Types" value={breakdown.built_count} /></Card>
            <Card><KPI label="Categories" value={(breakdown.per_category || []).length} /></Card>
          </div>
          {(breakdown.per_category || []).map((cat, ci) => (
            <Card key={ci} title={`${cat.category} (${cat.count})`}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 10 }}>
                {(cat.types || []).map((t, ti) => (
                  <div key={ti} style={{
                    border: '1px solid #bbf7d0', borderRadius: 6, padding: 10, background: '#f0fdf4'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{t.type}</span>
                      <StatusBadge status="built" />
                    </div>
                    {t.note && <div style={{ fontSize: 11, color: '#64748b' }}>{t.note}</div>}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {tab === 'not-pulled' && breakdown && (
        <Card title={`Not-Pulled AI Types (${breakdown.not_pulled_count})`}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            AI types cataloged in the source project (insur_project) but not applicable or not imported to epilepsy EEG.
          </p>
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>AI Type</th>
                  <th style={thStyle}>Note</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.not_pulled_list || []).map((t, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{t.type}</td>
                    <td style={{ ...tdStyle, fontSize: 11, color: '#64748b' }}>{t.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'coverage-matrix' && breakdown && (
        <>
          <Card title="Coverage Matrix — Built vs Total">
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Visual coverage showing which AI capability categories are implemented in the epilepsy EEG platform.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 10 }}>
              {(breakdown.per_category || []).map((cat, ci) => {
                const pct = Math.round(100 * cat.count / Math.max(overview.summary.built, 1))
                return (
                  <div key={ci} style={{
                    border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, background: '#f8fafc'
                  }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 6 }}>{cat.category}</div>
                    <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e' }}>{cat.count}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>built types</div>
                    <div style={{
                      marginTop: 8, height: 6, borderRadius: 3, background: '#e2e8f0', overflow: 'hidden'
                    }}>
                      <div style={{
                        height: '100%', width: `${pct}%`, background: '#22c55e', borderRadius: 3
                      }} />
                    </div>
                    <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 4 }}>
                      {pct}% of built portfolio
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>

          <Card title="Status Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              {[
                { label: 'Built', count: breakdown.built_count, color: '#22c55e', desc: 'Live endpoints + real logic' },
                { label: 'Not Pulled', count: breakdown.not_pulled_count, color: '#94a3b8', desc: 'Source project only' },
                { label: 'Scaffold', count: breakdown.scaffold_count, color: '#f97316', desc: 'Stub code exists' },
                { label: 'Planned', count: breakdown.planned_count, color: '#3b82f6', desc: 'In roadmap' },
              ].map((item, i) => (
                <div key={i} style={{
                  textAlign: 'center', padding: 16, border: `2px solid ${item.color}33`,
                  borderRadius: 8, background: `${item.color}08`
                }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: item.color }}>{item.count}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginTop: 4 }}>{item.label}</div>
                  <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{item.desc}</div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, padding: 8, border: '1px solid #e2e8f0', borderRadius: 6 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ fontSize: 12, color: '#334155' }}>{s.description}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ fontWeight: 600, fontSize: 12, color: '#1e293b' }}>{g.term}</span>
                  <span style={{ fontSize: 12, color: '#64748b' }}> — {g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
