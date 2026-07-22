import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', partial: '#f97316', planned: '#8b5cf6', unknown: '#94a3b8' }
const PRIORITY_COLORS = { '10/10': '#ef4444', '9/10': '#f97316', '8/10': '#3b82f6' }

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

export default function RoleSpecsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/role-specs/overview`),
      axios.get(`${API_URL}/api/role-specs/breakdown`),
      axios.get(`${API_URL}/api/role-specs/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Role Specs...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-role', 'field-analysis', 'by-priority', 'definitions']
  const s = overview.summary || {}

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1280, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note}</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t ? '2px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t ? '#eff6ff' : '#fff',
            color: tab === t ? '#2563eb' : '#64748b'
          }}>
            {t.split('-').map(w => w[0].toUpperCase() + w.slice(1)).join(' ')}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Roles" value={s.total_roles} />
              <KPI label="Fields (est.)" value={s.total_fields_estimate} />
              <KPI label="Total Sections" value={s.total_sections} />
              <KPI label="Built" value={s.built} sub="workbenches live" />
              <KPI label="Partial" value={s.partial} />
              <KPI label="Planned" value={s.planned} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Status distribution pie */}
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[overview.status_distribution[i]?.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Priority distribution pie */}
            <Card title="Priority Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.priority_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.priority_distribution || []).map((_, i) => (
                      <Cell key={i} fill={PRIORITY_COLORS[overview.priority_distribution[i]?.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Sections per role bar */}
          <Card title="Sections per Role">
            <ResponsiveContainer width="100%" height={Math.max(300, (overview.sections_per_role || []).length * 28)}>
              <BarChart data={overview.sections_per_role} layout="vertical" margin={{ left: 180, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* All roles table */}
          <Card title="All Roles Summary">
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Priority</th>
                    <th style={thStyle}>Fields</th>
                    <th style={thStyle}>Sections</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.roles_table || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{r.role}</td>
                      <td style={tdStyle}><PriorityBadge priority={r.priority} /></td>
                      <td style={tdStyle}>{r.fields}</td>
                      <td style={tdStyle}>{r.sections}</td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* BY ROLE TAB */}
      {tab === 'by-role' && breakdown && (
        <>
          {(breakdown.per_role || []).map((r, i) => (
            <Card key={i} title={r.role}>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
                <PriorityBadge priority={r.priority} />
                <StatusBadge status={r.status} />
                <span style={{ fontSize: 12, color: '#64748b' }}>Fields: {r.fields}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>Sections: {r.section_count}</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                {(r.sections || []).map((sec, j) => (
                  <span key={j} style={{
                    background: '#eff6ff', color: '#2563eb', border: '1px solid #bfdbfe',
                    borderRadius: 4, padding: '3px 8px', fontSize: 11
                  }}>
                    {sec}
                  </span>
                ))}
              </div>
              {r.endpoints && r.endpoints.length > 0 && (
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                  Endpoints: {r.endpoints.map((ep, j) => (
                    <code key={j} style={{ background: '#f1f5f9', padding: '1px 4px', borderRadius: 3, marginRight: 4 }}>{ep}</code>
                  ))}
                </div>
              )}
              {r.frontend && (
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                  Frontend: <code style={{ background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{r.frontend}</code>
                </div>
              )}
              {r.note && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 6, fontStyle: 'italic' }}>{r.note}</div>}
            </Card>
          ))}
        </>
      )}

      {/* FIELD ANALYSIS TAB */}
      {tab === 'field-analysis' && (
        <>
          <Card title="Field Count per Role (estimated midpoint)">
            <ResponsiveContainer width="100%" height={Math.max(300, (overview.field_counts || []).length * 28)}>
              <BarChart data={overview.field_counts} layout="vertical" margin={{ left: 180, right: 40 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v, _, props) => [`${v} (range: ${props.payload.range})`, 'Fields']} />
                <Bar dataKey="value" fill="#22c55e" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Field Count Details">
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Role</th>
                    <th style={thStyle}>Field Range</th>
                    <th style={thStyle}>Midpoint</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.field_counts || []).map((f, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 500 }}>{f.name}</td>
                      <td style={tdStyle}>{f.range}</td>
                      <td style={tdStyle}>{f.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* BY PRIORITY TAB */}
      {tab === 'by-priority' && breakdown && (
        <>
          {[
            { label: 'Critical (10/10)', roles: breakdown.by_priority?.critical_10, color: '#ef4444' },
            { label: 'High (9/10)', roles: breakdown.by_priority?.high_9, color: '#f97316' },
            { label: 'Standard (8/10)', roles: breakdown.by_priority?.standard, color: '#3b82f6' },
          ].map((tier, ti) => (
            <div key={ti} style={{ marginBottom: 20 }}>
              <h3 style={{ fontSize: 15, fontWeight: 600, color: tier.color, marginBottom: 10 }}>
                {tier.label} ({(tier.roles || []).length} roles)
              </h3>
              {(tier.roles || []).map((r, i) => (
                <Card key={i} title={r.role}>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
                    <StatusBadge status={r.status} />
                    <span style={{ fontSize: 12, color: '#64748b' }}>Fields: {r.fields} | Sections: {r.section_count}</span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                    {(r.sections || []).map((sec, j) => (
                      <span key={j} style={{
                        background: `${tier.color}11`, color: tier.color, border: `1px solid ${tier.color}44`,
                        borderRadius: 4, padding: '2px 7px', fontSize: 10
                      }}>
                        {sec}
                      </span>
                    ))}
                  </div>
                </Card>
              ))}
            </div>
          ))}
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            {(defs.status_legend || []).map((s, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <StatusBadge status={s.status} />
                <span style={{ fontSize: 12, color: '#475569' }}>{s.description}</span>
              </div>
            ))}
          </Card>
          <Card title="Priority Legend">
            {(defs.priority_legend || []).map((p, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <PriorityBadge priority={p.priority} />
                <span style={{ fontSize: 12, color: '#475569' }}>{p.description}</span>
              </div>
            ))}
          </Card>
          <Card title="Glossary">
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                      <td style={tdStyle}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>
          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
