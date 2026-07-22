import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const STATUS_COLORS = {
  fresh: '#10b981',
  active: '#3b82f6',
  stale: '#f59e0b'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, color: '#fff', background: color
    }}>
      {status}
    </span>
  )
}

export default function DataVersioningDashboard() {
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
          axios.get(`${API_URL}/api/data-versioning/overview`),
          axios.get(`${API_URL}/api/data-versioning/breakdown`),
          axios.get(`${API_URL}/api/data-versioning/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Data Versioning data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Versioning data...</div>
  )
  if (error) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview?.available) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Data Versioning data unavailable</div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'analysis', label: 'Data Analysis' },
    { id: 'lineage', label: 'Lineage & History' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpi = overview.kpi || {}

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Data Catalog KPIs" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
          <KPI label="Total Datasets" value={fmt(kpi.total_datasets)} color="#3b82f6" />
          <KPI label="Total Files" value={fmt(kpi.total_files)} color="#8b5cf6" />
          <KPI label="Total Size (MB)" value={fmt(kpi.total_size_mb)} color="#10b981" />
          <KPI label="Databases" value={fmt(kpi.databases)} color="#f59e0b" />
          <KPI label="Model Artifacts" value={fmt(kpi.model_artifacts)} color="#ec4899" />
          <KPI label="Last Updated" value={kpi.last_updated || '--'} color="#64748b" />
        </div>
      </Card>

      <Card title="Dataset Catalog" span={3}>
        {(overview.datasets || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Name</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Path</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Files</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Size</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Formats</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Last Modified</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Version</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {overview.datasets.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{d.name}</td>
                    <td style={{ padding: '8px 6px', color: '#64748b', fontSize: 12, fontFamily: 'monospace' }}>{d.path}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(d.files)}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{d.size}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{Array.isArray(d.formats) ? d.formats.join(', ') : d.formats}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{d.last_modified}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{d.version}</td>
                    <td style={{ textAlign: 'center', padding: '8px 6px' }}><StatusBadge status={d.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No datasets found</div>}
      </Card>

      <Card title="File Format Distribution">
        {(overview.format_distribution || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={overview.format_distribution} dataKey="value" nameKey="name"
                cx="50%" cy="50%" outerRadius={70}
                label={({ name, value }) => `${name}: ${value}`}>
                {overview.format_distribution.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No format data</div>}
      </Card>

      <Card title="Size by Dataset" span={2}>
        {(overview.size_by_dataset || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={overview.size_by_dataset} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tickFormatter={v => `${v} MB`} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
              <Tooltip formatter={v => `${fmt(v)} MB`} />
              <Bar dataKey="size_mb" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No size data</div>}
      </Card>
    </div>
  )

  const renderAnalysis = () => {
    const bd = breakdown || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Database Info" span={2}>
          {(bd.databases || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Name</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Size</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Tables</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Path</th>
                </tr>
              </thead>
              <tbody>
                {bd.databases.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{d.name}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{d.size}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(d.tables)}</td>
                    <td style={{ padding: '8px 6px', color: '#64748b', fontSize: 12, fontFamily: 'monospace' }}>{d.path}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No database data</div>}
        </Card>

        <Card title="Model Artifacts">
          {(bd.model_artifacts || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Name</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Size</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Modified</th>
                </tr>
              </thead>
              <tbody>
                {bd.model_artifacts.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{m.size}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{m.modified}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model artifacts</div>}
        </Card>

        <Card title="Staleness Assessment" span={2}>
          {(bd.staleness || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Dataset</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Last Modified</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Age (days)</th>
                  <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {bd.staleness.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{s.name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{s.last_modified}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(s.age_days)}</td>
                    <td style={{ textAlign: 'center', padding: '8px 6px' }}><StatusBadge status={s.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No staleness data</div>}
        </Card>

        <Card title="Staleness Status Distribution">
          {(bd.staleness_distribution || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={bd.staleness_distribution} dataKey="value" nameKey="name"
                  cx="50%" cy="50%" outerRadius={65}
                  label={({ name, value }) => `${name}: ${value}`}>
                  {bd.staleness_distribution.map((s, i) => (
                    <Cell key={i} fill={STATUS_COLORS[s.name] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No distribution data</div>}
        </Card>
      </div>
    )
  }

  const renderLineage = () => {
    const bd = breakdown || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Data Lineage" span={3}>
          {(bd.lineage || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Source</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Artifact</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Pipeline</th>
                </tr>
              </thead>
              <tbody>
                {bd.lineage.map((l, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{l.source}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{l.artifact}</td>
                    <td style={{ padding: '8px 6px', color: '#64748b', fontSize: 12 }}>{l.pipeline}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No lineage data</div>}
        </Card>

        <Card title="Recent Git Changes" span={2}>
          {(bd.git_changes || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Hash</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Message</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Date</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Files</th>
                </tr>
              </thead>
              <tbody>
                {bd.git_changes.map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 12, color: '#3b82f6' }}>{g.hash}</td>
                    <td style={{ padding: '8px 6px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.message}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{g.date}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(g.files_changed)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No git change data</div>}
        </Card>

        <Card title="Data Events (track.jsonl)">
          {(bd.data_events || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Timestamp</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Event</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Agent</th>
                </tr>
              </thead>
              <tbody>
                {bd.data_events.map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontSize: 12, color: '#64748b' }}>{e.timestamp}</td>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{e.event}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{e.agent}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No event data</div>}
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    const stages = defs?.stages || []
    const metrics = defs?.metrics || []
    const concepts = defs?.concepts || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {stages.length > 0 && (
          <Card title="Data Versioning Stages">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 200 }}>Stage</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {stages.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{s.name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{s.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {metrics.length > 0 && (
          <Card title="Metric Definitions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 200 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{m.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {concepts.length > 0 && (
          <Card title="Key Concepts">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 200 }}>Concept</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {concepts.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{c.name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569' }}>{c.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 6px', fontSize: 22, color: '#1e293b' }}>Data Versioning & Catalog</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Dataset catalog, file versioning, lineage tracking, and staleness monitoring
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'analysis' && renderAnalysis()}
      {tab === 'lineage' && renderLineage()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
