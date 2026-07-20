import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const READINESS_COLORS = { true: '#22c55e', partial: '#f97316', false: '#ef4444' }
const ROUTE_COLORS = { signal: '#3b82f6', rag: '#8b5cf6', cv: '#14b8a6', extract: '#f97316', video: '#ec4899' }

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

function ReadinessBadge({ value }) {
  const label = value === true ? 'AI Ready' : value === 'partial' ? 'Partial' : 'Not Ready'
  const bg = READINESS_COLORS[String(value)] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}

function RouteBadge({ route }) {
  const bg = ROUTE_COLORS[route] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {route}
    </span>
  )
}

function Stars({ count }) {
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

export default function EegDataFormatsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/eeg-data-formats/overview`),
      axios.get(`${API_URL}/eeg-data-formats/breakdown`),
      axios.get(`${API_URL}/eeg-data-formats/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading EEG Data Formats...</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#ef4444' }}>eeg_data_formats.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'by-format', label: 'By Format' },
    { id: 'routing', label: 'Routing' },
    { id: 'data-request', label: 'Data Request' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = overview.summary

  return (
    <div style={{ padding: '20px 24px', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>EEG Data Formats Dashboard</h2>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          {overview.title} &middot; Updated: {overview.updated_at}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', fontSize: 12, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#fff', color: tab === t.id ? '#fff' : '#475569',
            border: `1px solid ${tab === t.id ? '#3b82f6' : '#e2e8f0'}`, borderRadius: 6, cursor: 'pointer'
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 16 }}>
              <KPI label="Total Formats" value={s.total_formats} />
              <KPI label="AI Ready" value={s.ai_ready_count} />
              <KPI label="Partial" value={s.partially_ready} />
              <KPI label="Not Ready" value={s.not_ready} />
              <KPI label="Supported" value={s.supported_count} />
              <KPI label="Unsupported" value={s.unsupported_count} />
              <KPI label="Route Types" value={s.route_types} />
              <KPI label="Avg Stars" value={s.avg_stars} sub="out of 5" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="AI Readiness Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.ai_readiness_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {overview.ai_readiness_distribution.map((entry, i) => {
                      const colorKey = entry.name === 'AI Ready' ? 'true' : entry.name === 'Partial' ? 'partial' : 'false'
                      return <Cell key={i} fill={READINESS_COLORS[colorKey] || COLORS[i]} />
                    })}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Route Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.route_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {overview.route_distribution.map((entry, i) => (
                      <Cell key={i} fill={ROUTE_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Star Rating Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.star_distribution} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="value" fill="#eab308" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Formats Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Extension</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>AI Ready</th>
                    <th style={thStyle}>Stars</th>
                    <th style={thStyle}>Route</th>
                    <th style={thStyle}>Supported</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.formats_table.map((f, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontWeight: 600 }}>{f.ext}</td>
                      <td style={tdStyle}>{f.name}</td>
                      <td style={tdStyle}><ReadinessBadge value={f.ai_ready} /></td>
                      <td style={tdStyle}><Stars count={f.stars} /></td>
                      <td style={tdStyle}><RouteBadge route={f.route} /></td>
                      <td style={tdStyle}>{f.supported ? <span style={{ color: '#22c55e', fontWeight: 600 }}>Yes</span> : <span style={{ color: '#ef4444' }}>No</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'by-format' && breakdown && (
        <>
          {breakdown.per_format.map((f, i) => (
            <Card key={i}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
                <span style={{
                  background: '#1e293b', color: '#fff', borderRadius: 6,
                  padding: '4px 12px', fontSize: 13, fontWeight: 700, fontFamily: 'monospace'
                }}>
                  {f.ext}
                </span>
                <span style={{ fontSize: 15, fontWeight: 600, color: '#334155' }}>{f.name}</span>
                <Stars count={f.stars} />
                <ReadinessBadge value={f.ai_ready} />
                <RouteBadge route={f.route} />
                {f.supported ? (
                  <span style={{ background: '#dcfce7', color: '#166534', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>SUPPORTED</span>
                ) : (
                  <span style={{ background: '#fef2f2', color: '#991b1b', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>UNSUPPORTED</span>
                )}
              </div>

              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>Contains:</div>
                <div style={{ fontSize: 12, color: '#334155' }}>{f.contains}</div>
              </div>

              {f.good_for && (
                <div style={{ background: '#f0fdf4', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#166534', marginBottom: 4 }}>Good For:</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{f.good_for}</div>
                </div>
              )}

              {f.bad_for && (
                <div style={{ background: '#fef2f2', borderRadius: 6, padding: 12 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#991b1b', marginBottom: 4 }}>Bad For:</div>
                  <div style={{ fontSize: 12, color: '#334155' }}>{f.bad_for}</div>
                </div>
              )}
            </Card>
          ))}
        </>
      )}

      {tab === 'routing' && breakdown && (
        <>
          {breakdown.per_route.map((r, i) => (
            <Card key={i} title={r.route.toUpperCase() + ' Route'}>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12, marginBottom: 12 }}>
                <div style={{ fontSize: 12, color: '#334155' }}>{r.description}</div>
              </div>

              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 8 }}>Formats in this route ({r.formats.length}):</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {r.formats.map((f, j) => (
                    <div key={j} style={{
                      background: '#fff', border: '1px solid #e2e8f0', borderRadius: 6,
                      padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 8
                    }}>
                      <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 12 }}>{f.ext}</span>
                      <span style={{ fontSize: 11, color: '#64748b' }}>{f.name}</span>
                      <Stars count={f.stars} />
                      <ReadinessBadge value={f.ai_ready} />
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ background: '#dbeafe', color: '#1d4ed8', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>File Upload</span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>&rarr;</span>
                <span style={{ background: '#fef3c7', color: '#92400e', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Detect Format</span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>&rarr;</span>
                <span style={{
                  background: `${ROUTE_COLORS[r.route] || '#94a3b8'}22`,
                  color: ROUTE_COLORS[r.route] || '#94a3b8',
                  border: `1px solid ${ROUTE_COLORS[r.route] || '#94a3b8'}55`,
                  borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600
                }}>
                  {r.route.toUpperCase()} Pipeline
                </span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>&rarr;</span>
                <span style={{ background: '#dcfce7', color: '#166534', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Results</span>
              </div>
            </Card>
          ))}
        </>
      )}

      {tab === 'data-request' && breakdown && (
        <>
          <Card title="Must Have (Required Data)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Data</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.data_request.must_have || []).map((item, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{item.data}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{item.format}</td>
                      <td style={tdStyle}>{item.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Good to Have (Supplementary Data)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Data</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.data_request.good_to_have || []).map((item, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{item.data}</td>
                      <td style={{ ...tdStyle, fontFamily: 'monospace' }}>{item.format}</td>
                      <td style={tdStyle}>{item.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Minimum Dataset Requirement">
            <div style={{ background: '#eff6ff', borderRadius: 6, padding: 16, marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: '#1e40af', lineHeight: 1.6 }}>{breakdown.data_request.minimum_dbA}</div>
            </div>
          </Card>

          <Card title="Data Collection Rule">
            <div style={{ background: '#fef3c7', borderRadius: 6, padding: 16, border: '1px solid #fde68a' }}>
              <div style={{ fontSize: 12, color: '#92400e', lineHeight: 1.6, fontWeight: 500 }}>{breakdown.data_request.rule}</div>
            </div>
          </Card>
        </>
      )}

      {tab === 'definitions' && defs && (
        <>
          <Card title="Route Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Route</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.route_descriptions.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><RouteBadge route={r.route} /></td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="AI Readiness Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.ai_readiness_legend.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}>
                      <span style={{
                        background: `${r.color}22`, color: r.color, border: `1px solid ${r.color}55`,
                        borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
                      }}>
                        {r.value === 'true' ? 'AI Ready' : r.value === 'partial' ? 'Partial' : 'Not Ready'}
                      </span>
                    </td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Star Rating Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Rating</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.star_rating_legend.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><Stars count={r.stars} /></td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {defs.glossary.map((g, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {defs.clinical_notes.map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {defs.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
