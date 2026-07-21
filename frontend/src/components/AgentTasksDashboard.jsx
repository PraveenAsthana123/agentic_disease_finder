import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308', '#14b8a6', '#f43f5e', '#a855f7']
const STATUS_COLORS = { built: '#22c55e', planned: '#f97316', unknown: '#94a3b8' }

const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 13, fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155' }

function Card({ title, children, span }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)', gridColumn: span ? `span ${span}` : undefined }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{status}</span>
}

function CategoryBadge({ category }) {
  const hash = category.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
  const color = COLORS[hash % COLORS.length]
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{category}</span>
}

export default function AgentTasksDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [catFilter, setCatFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/agent-tasks/overview`),
      axios.get(`${API}/agent-tasks/breakdown`),
      axios.get(`${API}/agent-tasks/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Agent Tasks...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#f97316' }}>{overview.note || 'Not available'}</div>

  const tabs = ['overview', 'all-agents', 'by-category', 'modules', 'definitions']
  const k = overview.kpis || {}
  const agents = overview.agents_table || []
  const cats = (breakdown && breakdown.per_category) || []

  const filteredAgents = catFilter === 'all' ? agents : agents.filter(a => a.category === catFilter)
  const uniqueCategories = [...new Set(agents.map(a => a.category))].sort()

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{overview.title || 'Agent Tasks Registry'}</h2>
      <p style={{ margin: '0 0 18px', fontSize: 13, color: '#64748b' }}>{overview.description || 'Agent/task registry for the EEG epilepsy pipeline'}</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 500, background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569', transition: 'all .15s' }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Agents" value={k.total_agents} />
              <KPI label="Built" value={k.built} sub={`${Math.round((k.built / k.total_agents) * 100)}%`} />
              <KPI label="Planned" value={k.planned} />
              <KPI label="Categories" value={k.categories} />
              <KPI label="With Notes" value={k.with_notes} />
              <KPI label="With Needs" value={k.with_needs} />
            </div>
          </Card>

          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.status_distribution || []).map((_, i) => <Cell key={i} fill={Object.values(STATUS_COLORS)[i] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Category Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.category_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.category_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Agents per Category" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview.category_distribution} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── All Agents ── */}
      {tab === 'all-agents' && (
        <div>
          <Card>
            <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: '#475569' }}>Filter:</span>
              <button onClick={() => setCatFilter('all')} style={{ padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: catFilter === 'all' ? 700 : 400, background: catFilter === 'all' ? '#3b82f6' : '#e2e8f0', color: catFilter === 'all' ? '#fff' : '#475569' }}>All ({agents.length})</button>
              {uniqueCategories.map(c => (
                <button key={c} onClick={() => setCatFilter(c)} style={{ padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: catFilter === c ? 700 : 400, background: catFilter === c ? '#3b82f6' : '#e2e8f0', color: catFilter === c ? '#fff' : '#475569' }}>{c} ({agents.filter(a => a.category === c).length})</button>
              ))}
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Agent ID</th>
                    <th style={thStyle}>Task</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Module</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAgents.map((a, i) => (
                    <tr key={a.id + i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 12 }}>{a.id}</td>
                      <td style={{ ...tdStyle, maxWidth: 350 }}>{a.task}</td>
                      <td style={tdStyle}><StatusBadge status={a.status} /></td>
                      <td style={tdStyle}><CategoryBadge category={a.category} /></td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.module}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── By Category ── */}
      {tab === 'by-category' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {cats.map(cat => (
            <Card key={cat.category} title={`${cat.category} (${cat.total} agents — ${cat.built} built, ${cat.planned} planned)`}>
              <div style={{ display: 'grid', gap: 10 }}>
                {cat.agents.map((a, i) => (
                  <div key={a.id + i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ fontFamily: 'monospace', fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{a.id}</span>
                      <StatusBadge status={a.status} />
                    </div>
                    <div style={{ fontSize: 13, color: '#334155', marginBottom: 4 }}>{a.task}</div>
                    {a.module && a.module !== '(planned)' && (
                      <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'monospace' }}>Module: {a.module}</div>
                    )}
                    {a.needs && (
                      <div style={{ fontSize: 11, color: '#f97316', marginTop: 4 }}>Needs: {a.needs}</div>
                    )}
                    {a.note && (
                      <div style={{ fontSize: 11, color: '#64748b', marginTop: 4, fontStyle: 'italic' }}>{a.note}</div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ── Modules ── */}
      {tab === 'modules' && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="Module Inventory">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Agent</th>
                    <th style={thStyle}>Module Path</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Needs</th>
                  </tr>
                </thead>
                <tbody>
                  {agents.filter(a => a.module && a.module !== '(planned)').map((a, i) => (
                    <tr key={a.id + i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 12 }}>{a.id}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 11 }}>{a.module}</td>
                      <td style={tdStyle}><StatusBadge status={a.status} /></td>
                      <td style={{ ...tdStyle, fontSize: 12, color: a.has_needs ? '#f97316' : '#94a3b8' }}>{a.has_needs ? 'Yes' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="Status Legend">
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.status_legend || []).map(s => (
                <div key={s.status} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ fontSize: 13, color: '#334155' }}>{s.description}</span>
                </div>
              ))}
            </div>
          </Card>
          <Card title="Glossary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={g.term} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                      <td style={tdStyle}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => <li key={i} style={{ fontSize: 13, color: '#334155', marginBottom: 6 }}>{n}</li>)}
            </ul>
          </Card>
          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => <li key={i} style={{ fontSize: 13, color: '#334155', marginBottom: 6 }}><strong>{r.ref}</strong> — {r.detail}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
