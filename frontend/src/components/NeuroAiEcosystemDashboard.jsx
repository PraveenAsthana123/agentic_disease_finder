import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', installed: '#3b82f6', external: '#f97316', commercial: '#8b5cf6', unknown: '#94a3b8' }

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

function RatingStars({ count }) {
  if (!count) return <span style={{ color: '#94a3b8', fontSize: 11 }}>—</span>
  return (
    <span style={{ fontSize: 14, letterSpacing: 1 }}>
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < count ? '#eab308' : '#e2e8f0' }}>&#9733;</span>
      ))}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function NeuroAiEcosystemDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/neuro-ai-ecosystem/overview`),
      axios.get(`${API_URL}/api/neuro-ai-ecosystem/breakdown`),
      axios.get(`${API_URL}/api/neuro-ai-ecosystem/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Neuro AI Ecosystem...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-category', 'endpoints', 'recommended', 'definitions']
  const s = overview.summary || {}

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
            <Card><KPI label="Total Tools" value={s.total_tools} /></Card>
            <Card><KPI label="Categories" value={s.categories} /></Card>
            <Card><KPI label="Built" value={s.built} sub="live in platform" /></Card>
            <Card><KPI label="Installed" value={s.installed} sub="import verified" /></Card>
            <Card><KPI label="External" value={s.external} sub="separate infra" /></Card>
            <Card><KPI label="Commercial" value={s.commercial} /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview.status_distribution || []).map((_, i) => (
                      <Cell key={i} fill={STATUS_COLORS[overview.status_distribution[i]?.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Tools per Category">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.category_distribution} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="All Tools Summary">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Tool</th>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Purpose</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Rating</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.tools_table || []).map((t, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{t.name}</td>
                      <td style={tdStyle}>{t.category}</td>
                      <td style={tdStyle}>{t.purpose}</td>
                      <td style={tdStyle}><StatusBadge status={t.status} /></td>
                      <td style={tdStyle}><RatingStars count={t.rating} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'by-category' && breakdown && (
        <>
          {(breakdown.per_category || []).map((cat, ci) => (
            <Card key={ci} title={cat.category}>
              {cat.note && <p style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>{cat.note}</p>}
              <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                {Object.entries(cat.status_counts || {}).map(([st, cnt]) => (
                  <span key={st} style={{ fontSize: 11, color: '#475569' }}>
                    <StatusBadge status={st} /> {cnt}
                  </span>
                ))}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
                {(cat.tools || []).map((t, ti) => (
                  <div key={ti} style={{
                    border: '1px solid #e2e8f0', borderRadius: 6, padding: 10,
                    background: t.status === 'built' ? '#f0fdf4' : t.status === 'installed' ? '#eff6ff' : '#fff'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                      <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{t.name}</span>
                      <StatusBadge status={t.status} />
                    </div>
                    {t.purpose && <div style={{ fontSize: 11, color: '#64748b' }}>{t.purpose}</div>}
                    {t.domain && <div style={{ fontSize: 11, color: '#8b5cf6', marginTop: 2 }}>{t.domain}</div>}
                    {t.rating && <div style={{ marginTop: 4 }}><RatingStars count={t.rating} /></div>}
                    {t.endpoints && (
                      <div style={{ marginTop: 4 }}>
                        {(Array.isArray(t.endpoints) ? t.endpoints : [t.endpoints]).map((ep, ei) => (
                          <div key={ei} style={{ fontSize: 10, color: '#3b82f6', fontFamily: 'monospace' }}>{ep}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {tab === 'endpoints' && breakdown && (
        <Card title="Tools with Live API Endpoints">
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Tool</th>
                  <th style={thStyle}>Category</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Endpoints</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.tools_with_endpoints || []).map((t, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{t.name}</td>
                    <td style={tdStyle}>{t.category}</td>
                    <td style={tdStyle}><StatusBadge status={t.status} /></td>
                    <td style={tdStyle}>
                      {(Array.isArray(t.endpoints) ? t.endpoints : [t.endpoints]).map((ep, ei) => (
                        <div key={ei} style={{ fontSize: 10, color: '#3b82f6', fontFamily: 'monospace' }}>{ep}</div>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'recommended' && (
        <Card title="Recommended Open-Source Stack">
          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Minimum viable open-source neuro AI platform — one tool per clinical function.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {Object.entries(overview.recommended_stack || {}).map(([func, tool]) => (
              <div key={func} style={{
                border: '1px solid #e2e8f0', borderRadius: 6, padding: 12, background: '#f8fafc'
              }}>
                <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', fontWeight: 600, marginBottom: 4 }}>{func}</div>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#1e293b' }}>{tool}</div>
              </div>
            ))}
          </div>
        </Card>
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
