import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { downloaded: '#22c55e', pending: '#f97316', true: '#22c55e', false: '#f97316' }

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
      {status === true || status === 'true' ? 'AUTO' : status === false || status === 'false' ? 'MANUAL' : status}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function DataConfigDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [diseaseFilter, setDiseaseFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/data-config/overview`),
      axios.get(`${API_URL}/api/data-config/breakdown`),
      axios.get(`${API_URL}/api/data-config/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Data Configuration...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-disease', 'features', 'validation', 'definitions']
  const k = overview.kpis || {}

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
          <KPI label="Diseases" value={k.total_diseases} />
          <KPI label="Total Datasets" value={k.total_datasets} />
          <KPI label="Primary Datasets" value={k.total_primary_datasets} />
          <KPI label="Features" value={k.total_features} />
          <KPI label="Feature Categories" value={k.feature_categories} />
          <KPI label="Validation Datasets" value={k.validation_datasets} />
          <KPI label="Platforms" value={k.platforms_supported} />
          <KPI label="Download Sources" value={k.download_sources} />
        </div>
      </Card>

      <Card title="Datasets per Disease">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.datasets_per_disease} margin={{ left: 10, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Datasets" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Subjects per Disease">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.subjects_per_disease} layout="vertical" margin={{ left: 120, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#22c55e" name="Subjects" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Format Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.format_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.format_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="License Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.license_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.license_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Auto Download Distribution">
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={overview.auto_download_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name} (${value})`}>
              {(overview.auto_download_distribution || []).map((_, i) => <Cell key={i} fill={i === 0 ? '#22c55e' : '#f97316'} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="All Diseases Summary" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Display Name</th>
              <th style={thStyle}>Datasets</th>
              <th style={thStyle}>Primary</th>
              <th style={thStyle}>Total Subjects</th>
            </tr></thead>
            <tbody>
              {(overview.diseases_summary || []).map((d, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize' }}>{d.name}</td>
                  <td style={tdStyle}>{d.display_name}</td>
                  <td style={tdStyle}>{d.datasets_count}</td>
                  <td style={tdStyle}>{d.primary_count}</td>
                  <td style={tdStyle}>{d.total_subjects}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )

  /* ── By Disease Tab ── */
  const renderByDisease = () => {
    const bd = breakdown || { diseases: [] }
    const allDiseases = bd.diseases || []
    const filtered = diseaseFilter === 'all' ? allDiseases : allDiseases.filter(d => d.name === diseaseFilter)

    return (
      <div>
        <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button onClick={() => setDiseaseFilter('all')} style={{
            padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
            background: diseaseFilter === 'all' ? '#3b82f6' : '#fff', color: diseaseFilter === 'all' ? '#fff' : '#334155',
          }}>All</button>
          {allDiseases.map(d => (
            <button key={d.name} onClick={() => setDiseaseFilter(d.name)} style={{
              padding: '4px 12px', borderRadius: 4, border: '1px solid #e2e8f0', fontSize: 12, cursor: 'pointer',
              background: diseaseFilter === d.name ? '#3b82f6' : '#fff', color: diseaseFilter === d.name ? '#fff' : '#334155',
            }}>{d.display_name}</button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
          {filtered.map((disease, di) => (
            <Card key={di} title={disease.display_name} span={disease.primary_datasets && disease.primary_datasets.length > 2 ? 2 : undefined}>
              <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 8px' }}>{disease.description}</p>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                <span style={{
                  background: '#3b82f622', color: '#3b82f6', border: '1px solid #3b82f655',
                  borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
                }}>{disease.datasets_count} datasets</span>
              </div>
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr>
                    <th style={thStyle}>ID</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Source</th>
                    <th style={thStyle}>Subjects</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>License</th>
                    <th style={thStyle}>Download</th>
                  </tr></thead>
                  <tbody>
                    {(disease.primary_datasets || []).map((ds, i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                        <td style={{ ...tdStyle, fontFamily: 'monospace', fontSize: 11 }}>{ds.id}</td>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{ds.name}</td>
                        <td style={tdStyle}>{ds.source}</td>
                        <td style={tdStyle}>{ds.subjects}</td>
                        <td style={tdStyle}>{ds.format || '-'}</td>
                        <td style={tdStyle}>{ds.license}</td>
                        <td style={tdStyle}>
                          <span style={{
                            fontSize: 10, display: 'inline-block',
                            background: ds.auto_download ? '#dcfce7' : '#fef3c7',
                            color: ds.auto_download ? '#166534' : '#92400e',
                            borderRadius: 4, padding: '2px 6px'
                          }}>{ds.auto_download ? 'Auto' : 'Manual'}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── Features Tab ── */
  const renderFeatures = () => {
    const bd = breakdown || { features: {} }
    const features = bd.features || {}
    const categories = features.categories || {}
    const featureCategories = Object.keys(categories).filter(c => !c.endsWith('_bands'))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Feature Summary" span={2}>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', marginBottom: 12 }}>
            <KPI label="Total Features" value={features.total_count || k.total_features} />
            <KPI label="Categories" value={featureCategories.length} />
          </div>
        </Card>

        {featureCategories.map((catName, ci) => {
          const catFeatures = categories[catName] || []
          const bands = categories[catName + '_bands']

          return (
            <Card key={ci} title={catName.charAt(0).toUpperCase() + catName.slice(1) + ` (${catFeatures.length})`}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: bands ? 12 : 0 }}>
                {catFeatures.map((feat, fi) => (
                  <span key={fi} style={{
                    background: `${COLORS[ci % COLORS.length]}18`,
                    color: COLORS[ci % COLORS.length],
                    border: `1px solid ${COLORS[ci % COLORS.length]}44`,
                    borderRadius: 4, padding: '3px 8px', fontSize: 11, fontWeight: 500
                  }}>{feat}</span>
                ))}
              </div>
              {bands && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 6 }}>Frequency Bands</div>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead><tr>
                      <th style={thStyle}>Band</th>
                      <th style={thStyle}>Low (Hz)</th>
                      <th style={thStyle}>High (Hz)</th>
                    </tr></thead>
                    <tbody>
                      {Object.entries(bands).map(([band, range], bi) => (
                        <tr key={bi} style={{ background: bi % 2 === 0 ? '#fff' : '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize' }}>{band}</td>
                          <td style={tdStyle}>{range[0]}</td>
                          <td style={tdStyle}>{range[1]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          )
        })}
      </div>
    )
  }

  /* ── Validation Tab ── */
  const renderValidation = () => {
    const bd = breakdown || { validation_datasets: [] }
    const vals = bd.validation_datasets || []
    const totalSize = vals.reduce((s, v) => s + (v.size_mb || 0), 0)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Validation Summary" span={2}>
          <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
            <KPI label="Validation Datasets" value={vals.length} />
            <KPI label="Total Size" value={`${totalSize.toFixed(1)} MB`} />
          </div>
        </Card>

        {vals.map((v, vi) => (
          <Card key={vi} title={v.id}>
            <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
              <StatusBadge status={v.status} />
              <span style={{ fontSize: 11, color: '#64748b' }}>{v.size_mb} MB</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{v.file_count} files</span>
            </div>
            <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0' }}>
              <strong>Path:</strong> <code style={{ fontSize: 11, background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{v.path}</code>
            </p>
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Files</div>
              {(v.files || []).map((f, fi) => (
                <div key={fi} style={{
                  fontSize: 11, color: '#64748b', fontFamily: 'monospace',
                  background: '#f8fafc', padding: '3px 6px', marginBottom: 2, borderRadius: 3
                }}>{f}</div>
              ))}
            </div>
          </Card>
        ))}
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
            <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Label</th></tr></thead>
            <tbody>
              {(d.status_legend || []).map((s, i) => (
                <tr key={i}>
                  <td style={tdStyle}>
                    <span style={{
                      background: `${s.color}22`, color: s.color, border: `1px solid ${s.color}55`,
                      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
                    }}>{s.status}</span>
                  </td>
                  <td style={tdStyle}>{s.label}</td>
                </tr>
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
              {d.references.map((r, i) => <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>{r}</li>)}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Data Configuration</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {k.total_diseases} diseases, {k.total_datasets} datasets, {k.total_primary_datasets} primary datasets, {k.total_features} features — v{overview.version} ({overview.project})
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
      {tab === 'by-disease' && renderByDisease()}
      {tab === 'features' && renderFeatures()}
      {tab === 'validation' && renderValidation()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
