import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { real_data: '#22c55e', synthetic: '#f97316', downloading: '#3b82f6' }

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
  const key = (status || '').toLowerCase()
  const bg = STATUS_COLORS[key] || '#94a3b8'
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

export default function DatasetsConfigDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/datasets-config/overview`),
      axios.get(`${API_URL}/api/datasets-config/breakdown`),
      axios.get(`${API_URL}/api/datasets-config/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Datasets Config...</div>
  if (!overview.available) return <div style={{ padding: 32 }}>{overview.note}</div>

  const tabs = ['overview', 'per-disease', 'all-datasets', 'base-paths', 'definitions']
  const kpis = overview.kpis || {}
  const charts = overview.charts || {}

  return (
    <div style={{ padding: '24px 32px', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>{overview.note}</p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569'
          }}>
            {t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'space-around' }}>
              <KPI label="Diseases" value={kpis.total_diseases} />
              <KPI label="Datasets" value={kpis.total_datasets} />
              <KPI label="Total Subjects" value={kpis.total_subjects} />
              <KPI label="All Real Data" value={kpis.all_real_data ? 'Yes' : 'No'} />
              <KPI label="Project" value={kpis.project} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Subjects by Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={charts.subjects_by_disease}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Datasets by Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={charts.datasets_by_disease}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Format Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.format_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.format_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Source Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.source_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.source_distribution || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Disease Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Subjects</th>
                    <th style={thStyle}>Accuracy</th>
                    <th style={thStyle}>Datasets</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.summary_table || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}><strong>{r.disease}</strong></td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                      <td style={tdStyle}>{r.subjects}</td>
                      <td style={tdStyle}><span style={{ color: '#22c55e', fontWeight: 600 }}>{r.accuracy}</span></td>
                      <td style={tdStyle}>{r.datasets}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'per-disease' && breakdown && (
        <>
          {(breakdown.diseases || []).map((d, i) => (
            <Card key={i} title={`${d.disease} — ${d.total_subjects} subjects — ${d.accuracy} accuracy`}>
              <div style={{ marginBottom: 8 }}>
                <StatusBadge status={d.status} />
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Dataset</th>
                    <th style={thStyle}>Subjects</th>
                    <th style={thStyle}>Channels</th>
                    <th style={thStyle}>Rate (Hz)</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Source</th>
                  </tr>
                </thead>
                <tbody>
                  {(d.datasets || []).map((ds, j) => (
                    <tr key={j} style={{ background: j % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}><strong>{ds.name}</strong></td>
                      <td style={tdStyle}>{ds.subjects}</td>
                      <td style={tdStyle}>{ds.channels}</td>
                      <td style={tdStyle}>{ds.sampling_rate}</td>
                      <td style={tdStyle}>{ds.format}</td>
                      <td style={tdStyle}>{ds.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          ))}
        </>
      )}

      {tab === 'all-datasets' && breakdown && (
        <Card title="All Datasets">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Disease</th>
                  <th style={thStyle}>Dataset</th>
                  <th style={thStyle}>Subjects</th>
                  <th style={thStyle}>Channels</th>
                  <th style={thStyle}>Rate (Hz)</th>
                  <th style={thStyle}>Format</th>
                  <th style={thStyle}>Source</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown.diseases || []).flatMap((d, di) =>
                  (d.datasets || []).map((ds, j) => (
                    <tr key={`${di}-${j}`} style={{ background: (di + j) % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={tdStyle}>{d.disease}</td>
                      <td style={tdStyle}><strong>{ds.name}</strong></td>
                      <td style={tdStyle}>{ds.subjects}</td>
                      <td style={tdStyle}>{ds.channels}</td>
                      <td style={tdStyle}>{ds.sampling_rate}</td>
                      <td style={tdStyle}>{ds.format}</td>
                      <td style={tdStyle}>{ds.source}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'base-paths' && breakdown && (
        <Card title="Base Paths">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Label</th>
                <th style={thStyle}>Path</th>
              </tr>
            </thead>
            <tbody>
              {(breakdown.base_paths || []).map((bp, i) => (
                <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                  <td style={tdStyle}><strong>{bp.label}</strong></td>
                  <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 11 }}>{bp.path}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Meaning</th></tr></thead>
              <tbody>
                {(defs.status_legend || []).map((s, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><StatusBadge status={s.status} /></td>
                    <td style={tdStyle}>{s.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><strong>{g.term}</strong></td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Source</th><th style={thStyle}>Note</th></tr></thead>
              <tbody>
                {(defs.references || []).map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><strong>{r.label}</strong></td>
                    <td style={tdStyle}>{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </div>
  )
}
