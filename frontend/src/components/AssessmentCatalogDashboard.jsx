import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', cataloged: '#3b82f6', partial: '#f97316', planned: '#8b5cf6' }
const PRIORITY_COLORS = { mandatory: '#ef4444', 'highly valuable': '#f97316', 'very valuable': '#8b5cf6', valuable: '#3b82f6', preferred: '#14b8a6' }

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

function PriorityBadge({ priority }) {
  const bg = PRIORITY_COLORS[priority] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {priority}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function AssessmentCatalogDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [catFilter, setCatFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/assessment-catalog/overview`),
      axios.get(`${API_URL}/api/assessment-catalog/breakdown`),
      axios.get(`${API_URL}/api/assessment-catalog/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Assessment Catalog...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-category', 'top-10', 'research-vars', 'definitions']
  const k = overview.kpis || {}
  const instruments = overview.instruments || []
  const categories = overview.categories || []

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
          <KPI label="Total Instruments" value={k.total_instruments} />
          <KPI label="Built" value={k.built} />
          <KPI label="Categories" value={k.categories} />
          <KPI label="Top 10 Thesis" value={k.top10_for_thesis} />
          <KPI label="Research Variables" value={k.research_variables} />
          <KPI label="Mandatory" value={k.mandatory} />
          <KPI label="Specialists" value={k.specialists} />
        </div>
      </Card>

      <Card title="Status Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.status_distribution} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={70} label={({ status, count }) => `${status} (${count})`}>
              {(overview.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Priority Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.priority_distribution} dataKey="count" nameKey="priority" cx="50%" cy="50%" outerRadius={70} label={({ priority, count }) => `${priority} (${count})`}>
              {(overview.priority_distribution || []).map((_, i) => <Cell key={i} fill={Object.values(PRIORITY_COLORS)[i] || COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Instruments per Category" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={categories} layout="vertical" margin={{ left: 140, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="category" width={130} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="total" fill="#3b82f6" name="Instruments" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Instruments" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Instrument</th>
              <th style={thStyle}>Category</th>
              <th style={thStyle}>Purpose</th>
              <th style={thStyle}>Output</th>
              <th style={thStyle}>Specialist</th>
              <th style={thStyle}>Priority</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Top 10</th>
            </tr></thead>
            <tbody>
              {instruments.map((inst, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{inst.name}</td>
                  <td style={tdStyle}>{inst.category}</td>
                  <td style={tdStyle}>{inst.purpose}</td>
                  <td style={tdStyle}>{inst.output}</td>
                  <td style={tdStyle}>{inst.specialist}</td>
                  <td style={tdStyle}><PriorityBadge priority={inst.priority} /></td>
                  <td style={tdStyle}><StatusBadge status={inst.status} /></td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{inst.in_top10 ? '\u2605' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {overview.honest_note && (
        <Card title="Honest Note" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', margin: 0 }}>{overview.honest_note}</p>
        </Card>
      )}
    </div>
  )

  /* ── By Category Tab ── */
  const renderByCategory = () => {
    const bd = breakdown || { instruments: [] }
    const allCats = [...new Set(bd.instruments.map(i => i.category))]
    const filtered = catFilter === 'all' ? bd.instruments : bd.instruments.filter(i => i.category === catFilter)

    return (
      <div>
        <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button onClick={() => setCatFilter('all')} style={{
            padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
            background: catFilter === 'all' ? '#3b82f6' : '#fff', color: catFilter === 'all' ? '#fff' : '#334155',
          }}>All</button>
          {allCats.map(c => (
            <button key={c} onClick={() => setCatFilter(c)} style={{
              padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
              background: catFilter === c ? '#3b82f6' : '#fff', color: catFilter === c ? '#fff' : '#334155',
            }}>{c}</button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16 }}>
          {filtered.map((inst, i) => (
            <Card key={i} title={inst.name}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                <StatusBadge status={inst.status} />
                <PriorityBadge priority={inst.priority} />
                {inst.in_top10 && <span style={{ background: '#fef3c7', color: '#92400e', border: '1px solid #fcd34d', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>TOP 10</span>}
              </div>
              <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}><strong>Purpose:</strong> {inst.purpose}</p>
              <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}><strong>Output:</strong> {inst.output}</p>
              <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}><strong>Specialist:</strong> {inst.specialist}</p>
              <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}><strong>Category:</strong> {inst.category}</p>
              {inst.endpoint && <p style={{ fontSize: 11, color: '#3b82f6', margin: '4px 0' }}>Endpoint: {inst.endpoint}</p>}
              {inst.api && <p style={{ fontSize: 11, color: '#3b82f6', margin: '4px 0' }}>API: {inst.api}</p>}
              {inst.frontend && <p style={{ fontSize: 11, color: '#14b8a6', margin: '4px 0' }}>Frontend: {inst.frontend}</p>}
              {inst.note && <p style={{ fontSize: 11, color: '#94a3b8', margin: '4px 0', fontStyle: 'italic' }}>{inst.note}</p>}
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── Top 10 Tab ── */
  const renderTop10 = () => {
    const top10 = overview.top10_for_thesis || []
    const top10Instruments = instruments.filter(i => i.in_top10)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Top 10 Instruments for Thesis" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Primary outcome and predictor variables selected for the EEG AI epilepsy study.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {top10.map((name, i) => {
              const inst = top10Instruments.find(t => t.name === name) || {}
              return (
                <div key={i} style={{
                  background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 8, padding: 12,
                  display: 'flex', gap: 12, alignItems: 'flex-start'
                }}>
                  <div style={{
                    background: '#f59e0b', color: '#fff', borderRadius: '50%', width: 28, height: 28,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 13, flexShrink: 0
                  }}>{i + 1}</div>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{name}</div>
                    {inst.purpose && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{inst.purpose}</div>}
                    <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                      {inst.priority && <PriorityBadge priority={inst.priority} />}
                      {inst.specialist && <span style={{ fontSize: 10, color: '#94a3b8' }}>{inst.specialist}</span>}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </Card>
      </div>
    )
  }

  /* ── Research Variables Tab ── */
  const renderResearchVars = () => {
    const bd = breakdown || { research_variables: [] }
    const vars = bd.research_variables || []
    const prCounts = {}
    vars.forEach(v => { prCounts[v.priority] = (prCounts[v.priority] || 0) + 1 })
    const prData = Object.entries(prCounts).map(([k, v]) => ({ priority: k, count: v }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Research Variables" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Minimum dataset required for retrospective EEG AI analysis.
          </p>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Variable</th>
              <th style={thStyle}>Priority</th>
            </tr></thead>
            <tbody>
              {vars.map((v, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{v.variable}</td>
                  <td style={tdStyle}><PriorityBadge priority={v.priority} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Priority Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={prData} dataKey="count" nameKey="priority" cx="50%" cy="50%" outerRadius={70} label={({ priority, count }) => `${priority} (${count})`}>
                {prData.map((_, i) => <Cell key={i} fill={Object.values(PRIORITY_COLORS)[i] || COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    if (!defs) return <div style={{ color: '#64748b' }}>Loading definitions...</div>
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Status Legend">
          {(defs.status_legend || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
              <StatusBadge status={s.status} />
              <span style={{ fontSize: 12, color: '#64748b' }}>{s.description}</span>
            </div>
          ))}
        </Card>

        <Card title="Priority Legend">
          {(defs.priority_legend || []).map((p, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
              <PriorityBadge priority={p.priority} />
              <span style={{ fontSize: 12, color: '#64748b' }}>{p.description}</span>
            </div>
          ))}
        </Card>

        <Card title="Glossary" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 8 }}>
            {(defs.glossary || []).map((g, i) => (
              <div key={i} style={{ fontSize: 12, padding: '4px 0', borderBottom: '1px solid #f1f5f9' }}>
                <strong style={{ color: '#1e293b' }}>{g.term}</strong>
                <span style={{ color: '#64748b' }}> — {g.definition}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Clinical Notes" span={2}>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.clinical_notes || []).map((n, i) => (
              <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{n}</li>
            ))}
          </ul>
        </Card>

        <Card title="References" span={2}>
          <ol style={{ margin: 0, paddingLeft: 18 }}>
            {(defs.references || []).map((r, i) => (
              <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>
                <strong>{r.ref}</strong> — {r.detail}
              </li>
            ))}
          </ol>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#1e293b' }}>{overview.title}</h2>
        <p style={{ margin: '4px 0 0', fontSize: 12, color: '#94a3b8' }}>{overview.note} &middot; Updated {overview.updated_at}</p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
            background: tab === t ? '#3b82f6' : '#fff', color: tab === t ? '#fff' : '#334155', fontWeight: tab === t ? 600 : 400,
          }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'by-category' && renderByCategory()}
      {tab === 'top-10' && renderTop10()}
      {tab === 'research-vars' && renderResearchVars()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
