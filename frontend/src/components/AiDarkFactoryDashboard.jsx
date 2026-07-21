import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308', '#14b8a6', '#f43f5e', '#a855f7']
const STATUS_COLORS = { built: '#22c55e', cataloged: '#3b82f6', planned: '#f97316', adapter: '#06b6d4', unknown: '#94a3b8' }

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

export default function AiDarkFactoryDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [catFilter, setCatFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/ai-dark-factory/overview`),
      axios.get(`${API}/ai-dark-factory/breakdown`),
      axios.get(`${API}/ai-dark-factory/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading AI Dark Factory...</div>
  if (overview.available === false) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'flow-pipeline', 'tool-catalog', 'planes-patterns', 'definitions']
  const k = overview.kpis || {}

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
          <KPI label="Flow Stages" value={k.total_stages} />
          <KPI label="Built" value={k.built} />
          <KPI label="Cataloged" value={k.cataloged} />
          <KPI label="Planned" value={k.planned} />
          <KPI label="Tools" value={k.total_tools} />
          <KPI label="Categories" value={k.tool_categories} />
          <KPI label="Planes" value={k.planes} />
          <KPI label="Patterns" value={k.patterns} />
        </div>
      </Card>

      <Card title="Flow Stage Status">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.flow_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.flow_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Tool Status">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.tool_status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.tool_status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Tools per Category" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.tools_per_category || []} layout="vertical" margin={{ left: 120, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Tools" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Flow Pipeline Summary" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Stage</th>
              <th style={thStyle}>Tool</th>
              <th style={thStyle}>Produces</th>
              <th style={thStyle}>Status</th>
            </tr></thead>
            <tbody>
              {(overview.flow_table || []).map((s, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={tdStyle}>{s.n}</td>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{s.stage}</td>
                  <td style={tdStyle}>{s.tool}</td>
                  <td style={tdStyle}>{s.produces || '-'}</td>
                  <td style={tdStyle}><StatusBadge status={s.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  /* ── Flow Pipeline Tab ── */
  const renderFlowPipeline = () => (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Full Flow Pipeline (${(overview.flow_table || []).length} stages)`}>
        <div style={{ maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Stage</th>
              <th style={thStyle}>Tool</th>
              <th style={thStyle}>Produces</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Note</th>
            </tr></thead>
            <tbody>
              {(overview.flow_table || []).map((s, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={tdStyle}>{s.n}</td>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{s.stage}</td>
                  <td style={tdStyle}>{s.tool}</td>
                  <td style={tdStyle}>{s.produces || '-'}</td>
                  <td style={tdStyle}><StatusBadge status={s.status} /></td>
                  <td style={{ ...tdStyle, fontSize: 11, color: '#64748b', maxWidth: 300 }}>{s.note || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  /* ── Tool Catalog Tab ── */
  const renderToolCatalog = () => {
    const bd = breakdown || { per_category: [] }
    const cats = bd.per_category || []
    const allCats = ['all', ...cats.map(c => c.category)]
    const filtered = catFilter === 'all' ? cats : cats.filter(c => c.category === catFilter)

    return (
      <div>
        <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {allCats.map(c => (
            <button key={c} onClick={() => setCatFilter(c)} style={{
              padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
              background: catFilter === c ? '#3b82f6' : '#fff', color: catFilter === c ? '#fff' : '#334155',
            }}>{c === 'all' ? 'All' : c}</button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 16 }}>
          {filtered.map((cat, ci) => (
            <Card key={ci} title={cat.category}>
              <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 11, color: '#64748b' }}>{cat.tools.length} tool{cat.tools.length !== 1 ? 's' : ''}</span>
              </div>
              {cat.tools.map((t, ti) => (
                <div key={ti} style={{ background: '#f8fafc', borderRadius: 6, padding: 10, border: '1px solid #e2e8f0', marginBottom: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{t.tool}</span>
                    <StatusBadge status={t.status} />
                  </div>
                  <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}>{t.for}</p>
                  {t.note && <p style={{ fontSize: 10, color: '#94a3b8', margin: '2px 0', fontStyle: 'italic' }}>{t.note}</p>}
                </div>
              ))}
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── Planes & Patterns Tab ── */
  const renderPlanesPatterns = () => {
    const bd = breakdown || { planes: [], patterns: [] }
    const planes = bd.planes || []
    const patterns = bd.patterns || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
        <Card title="Architectural Planes" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
            {planes.map((p, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{p.plane}</span>
                  {p.status && <StatusBadge status={p.status} />}
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
                  {(p.components || []).map((c, ci) => (
                    <span key={ci} style={{ background: '#e0e7ff', color: '#3730a3', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 500 }}>{c}</span>
                  ))}
                </div>
                {p.note && <p style={{ fontSize: 11, color: '#64748b', margin: '4px 0 0', fontStyle: 'italic' }}>{p.note}</p>}
              </div>
            ))}
          </div>
        </Card>

        <Card title="Architecture Patterns" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
            {patterns.map((p, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{p.id.replace(/_/g, ' ')}</span>
                  <StatusBadge status={p.status} />
                </div>
                <p style={{ fontSize: 12, color: '#334155', margin: '4px 0' }}>{p.desc}</p>
                <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Best for:</strong> {p.best_for}</p>
                <p style={{ fontSize: 11, color: '#ef4444', margin: '2px 0' }}><strong>Failure mode:</strong> {p.failure_mode}</p>
                {p.note && <p style={{ fontSize: 10, color: '#94a3b8', margin: '4px 0 0', fontStyle: 'italic' }}>{p.note}</p>}
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    const d = defs || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Description</th></tr></thead>
            <tbody>
              {(d.status_legend || []).map((s, i) => (
                <tr key={i}><td style={tdStyle}><StatusBadge status={s.status} /></td><td style={tdStyle}>{s.description}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Glossary" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
            <tbody>
              {(d.glossary || []).map((g, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                  <td style={tdStyle}>{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        {d.clinical_notes && d.clinical_notes.length > 0 && (
          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.clinical_notes.map((n, i) => <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{n}</li>)}
            </ul>
          </Card>
        )}

        {d.references && d.references.length > 0 && (
          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {d.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>
                  <strong>{r.ref}</strong> — {r.detail}
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>AI Dark Factory</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {k.total_stages} flow stages, {k.total_tools} tools cataloged, {k.planes} planes, {k.patterns} patterns — autonomous software factory reference architecture
      </p>

      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', background: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t ? 600 : 400,
            color: tab === t ? '#3b82f6' : '#64748b',
            borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            marginBottom: -2, textTransform: 'capitalize'
          }}>{t.replace(/-/g, ' ')}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'flow-pipeline' && renderFlowPipeline()}
      {tab === 'tool-catalog' && renderToolCatalog()}
      {tab === 'planes-patterns' && renderPlanesPatterns()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
