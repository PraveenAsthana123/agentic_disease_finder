import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { built: '#22c55e', simulated: '#f97316', planned: '#94a3b8', unknown: '#cbd5e1' }

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

function Tag({ text, color }) {
  const c = color || '#3b82f6'
  return (
    <span style={{
      background: `${c}15`, color: c, border: `1px solid ${c}30`,
      borderRadius: 4, padding: '2px 7px', fontSize: 10, fontWeight: 500, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>
      {text}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function DataSourcesCatalogDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/data-sources-catalog/overview`),
      axios.get(`${API_URL}/data-sources-catalog/breakdown`),
      axios.get(`${API_URL}/data-sources-catalog/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Data Sources Catalog...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#94a3b8' }}>{overview.note}</div>

  const tabs = ['overview', 'internal', 'external', 'public', 'features', 'definitions']
  const kpis = overview.kpis || {}
  const charts = overview.charts || {}
  const split = overview.split || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>{overview.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{overview.note} &middot; Updated {overview.updated_at}</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t ? 700 : 500, cursor: 'pointer',
            border: 'none', borderBottom: tab === t ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t ? '#1e293b' : '#64748b', marginBottom: -2
          }}>
            {t === 'overview' ? 'Overview' : t === 'internal' ? 'Internal Datasets' : t === 'external' ? 'External Validation' : t === 'public' ? 'Public Datasets' : t === 'features' ? 'Feature Extraction' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Total Datasets" value={kpis.total_datasets} />
              <KPI label="Internal" value={kpis.internal} sub="7 diseases" />
              <KPI label="External" value={kpis.external} sub="validation" />
              <KPI label="Public" value={kpis.public} sub="benchmarks" />
              <KPI label="Internal Samples" value={kpis.internal_samples?.toLocaleString()} />
              <KPI label="Total Features" value={kpis.total_features} />
              <KPI label="Freq Bands" value={kpis.frequency_bands} />
              <KPI label="Sampling Rate" value={`${kpis.sampling_rate} Hz`} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Dataset Category Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.category_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.category_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Samples per Disease (Internal)">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={charts.samples_by_disease || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="augmented" fill="#3b82f6" name="Augmented" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="original" fill="#22c55e" name="Original" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Public Datasets by Disease">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.public_by_disease || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.public_by_disease || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Feature Type Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.feature_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.feature_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Data split summary */}
          <Card title="Data Split Strategy">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Training/Validation</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{split.train_method}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Train {(split.train_ratio * 100)}% / Val {(split.val_ratio * 100)}%</div>
              </div>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>External Validation</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{split.external_method} ({(split.external_ratio * 100)}%)</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Overfitting detection</div>
              </div>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Total</div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{split.total_samples?.toLocaleString()} samples</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{split.total_diseases} diseases &middot; {split.total_samples_per_disease}/disease</div>
              </div>
            </div>
          </Card>

          {/* Summary table */}
          <Card title="All Datasets Summary">
            <div style={{ overflowX: 'auto', maxHeight: 350 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Category</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Original</th>
                    <th style={thStyle}>Augmented</th>
                    <th style={thStyle}>Features</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.summary_table || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{r.name}</td>
                      <td style={tdStyle}><Tag text={r.category} color={r.category === 'Internal' ? '#3b82f6' : '#f97316'} /></td>
                      <td style={tdStyle}>{r.disease}</td>
                      <td style={tdStyle}>{r.samples_original}</td>
                      <td style={tdStyle}>{r.samples_augmented}</td>
                      <td style={tdStyle}>{r.features}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* INTERNAL DATASETS TAB */}
      {tab === 'internal' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {(breakdown.internal || []).map(d => (
              <Card key={d.id} title={d.name}>
                <div style={{ marginBottom: 8 }}>
                  <Tag text={`${d.samples_original} original`} color="#22c55e" />
                  <Tag text={`${d.samples_augmented} augmented`} color="#3b82f6" />
                  <Tag text={`${d.features} features`} color="#8b5cf6" />
                </div>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>
                  <strong>Classes:</strong> {Object.entries(d.classes || {}).map(([k, v]) => `${k}=${v}`).join(', ')}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                  <strong>Source:</strong> {d.source}
                </div>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                  <strong>Preprocessing:</strong> {d.preprocessing}
                </div>
                <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6 }}>
                  Path: {d.path} &middot; {d.original_file} / {d.augmented_file}
                </div>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* EXTERNAL VALIDATION TAB */}
      {tab === 'external' && breakdown && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            {(breakdown.external || []).map(d => (
              <Card key={d.id} title={d.name}>
                <div style={{ marginBottom: 8 }}>
                  <Tag text={`${d.samples} samples`} color="#3b82f6" />
                  <StatusBadge status={d.status} />
                </div>
                {d.citation && <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}><strong>Citation:</strong> {d.citation}</div>}
                <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 6 }}>Path: {d.path}</div>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* PUBLIC DATASETS TAB */}
      {tab === 'public' && breakdown && (
        <>
          <Card title="Public EEG Dataset Sizes">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={charts.public_sample_sizes || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={180} />
                <Tooltip />
                <Bar dataKey="value" fill="#8b5cf6" name="Samples" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Public Datasets">
            <div style={{ overflowX: 'auto', maxHeight: 400 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Samples</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Access</th>
                    <th style={thStyle}>Downloaded</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.public || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}>{r.name}</td>
                      <td style={tdStyle}><Tag text={r.disease} /></td>
                      <td style={tdStyle}>{r.samples?.toLocaleString()}</td>
                      <td style={tdStyle}><Tag text={r.format} color="#8b5cf6" /></td>
                      <td style={tdStyle}>{r.access}</td>
                      <td style={tdStyle}>{r.downloaded ? <Tag text="Local" color="#22c55e" /> : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* FEATURE EXTRACTION TAB */}
      {tab === 'features' && breakdown && breakdown.feature_params && (
        <>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Sampling Rate" value={`${breakdown.feature_params.sampling_rate} Hz`} />
              <KPI label="Window Size" value={`${breakdown.feature_params.window_size} samples`} />
              <KPI label="Overlap" value={`${(breakdown.feature_params.overlap * 100)}%`} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Frequency Bands">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Band</th>
                    <th style={thStyle}>Range (Hz)</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.feature_params.bands || []).map((b, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize' }}>{b.name}</td>
                      <td style={tdStyle}>{b.range_hz}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card title="Feature Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={charts.feature_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.feature_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <Card title="Time-domain Features">
              {(breakdown.feature_params.time_domain || []).map((f, i) => (
                <Tag key={i} text={f} color="#3b82f6" />
              ))}
            </Card>
            <Card title="Frequency-domain Features">
              {(breakdown.feature_params.frequency_domain || []).map((f, i) => (
                <Tag key={i} text={f} color="#8b5cf6" />
              ))}
            </Card>
            <Card title="Nonlinear Features">
              {(breakdown.feature_params.nonlinear || []).map((f, i) => (
                <Tag key={i} text={f} color="#f97316" />
              ))}
            </Card>
          </div>
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Meaning</th>
                </tr>
              </thead>
              <tbody>
                {(defs.status_legend || []).map((s, i) => (
                  <tr key={i}>
                    <td style={tdStyle}><StatusBadge status={s.status} /></td>
                    <td style={tdStyle}>{s.meaning}</td>
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
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Source</th>
                  <th style={thStyle}>Note</th>
                </tr>
              </thead>
              <tbody>
                {(defs.references || []).map((r, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.label}</td>
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
