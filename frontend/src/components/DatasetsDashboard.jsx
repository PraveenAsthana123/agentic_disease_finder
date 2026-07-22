import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = { REAL_DATA: '#22c55e', SYNTHETIC: '#f97316', PLANNED: '#8b5cf6' }

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

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function DatasetsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)
  const [diseaseFilter, setDiseaseFilter] = useState('all')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/datasets/overview`),
      axios.get(`${API_URL}/api/datasets/breakdown`),
      axios.get(`${API_URL}/api/datasets/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 24, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 24, color: '#64748b' }}>Loading Datasets Registry...</div>
  if (!overview.available) return <div style={{ padding: 24, color: '#f97316' }}>{overview.note}</div>

  const tabs = ['overview', 'by-disease', 'all-datasets', 'formats', 'definitions']
  const k = overview.kpis || {}
  const diseases = overview.diseases || []

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
          <KPI label="Diseases" value={k.total_diseases} />
          <KPI label="Datasets" value={k.total_datasets} />
          <KPI label="Total Subjects" value={k.total_subjects} />
          <KPI label="All Real Data" value={k.all_real_data ? 'Yes' : 'No'} />
          <KPI label="Avg Accuracy" value={`${k.avg_accuracy}%`} />
          <KPI label="Formats" value={k.format_count} />
        </div>
      </Card>

      <Card title="Subjects per Disease">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.subject_distribution} layout="vertical" margin={{ left: 100, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" name="Subjects" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Accuracy per Disease">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.accuracy_distribution} layout="vertical" margin={{ left: 100, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[90, 100]} />
            <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => `${v}%`} />
            <Bar dataKey="value" fill="#22c55e" name="Accuracy %" radius={[0, 4, 4, 0]} />
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

      <Card title="All Diseases Summary" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Disease</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Subjects</th>
              <th style={thStyle}>Accuracy</th>
              <th style={thStyle}>Datasets</th>
              <th style={thStyle}>Formats</th>
            </tr></thead>
            <tbody>
              {diseases.map((d, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, textTransform: 'capitalize' }}>{d.name}</td>
                  <td style={tdStyle}><StatusBadge status={d.status} /></td>
                  <td style={tdStyle}>{d.total_subjects}</td>
                  <td style={tdStyle}>{d.accuracy}%</td>
                  <td style={tdStyle}>{d.dataset_count}</td>
                  <td style={tdStyle}>{(d.formats || []).join(', ')}</td>
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
              textTransform: 'capitalize'
            }}>{d.name}</button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
          {filtered.map((disease, di) => (
            <Card key={di} title={disease.name} span={disease.datasets && disease.datasets.length > 2 ? 2 : undefined}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                <StatusBadge status={disease.status} />
                <span style={{ fontSize: 11, color: '#64748b' }}>{disease.total_subjects} subjects</span>
                <span style={{ fontSize: 11, color: '#22c55e', fontWeight: 600 }}>{disease.accuracy}% accuracy</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10, marginTop: 8 }}>
                {(disease.datasets || []).map((ds, i) => (
                  <div key={i} style={{ background: '#f8fafc', borderRadius: 6, padding: 10, border: '1px solid #e2e8f0' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{ds.name}</div>
                    <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Subjects:</strong> {ds.subjects}</p>
                    {ds.channels && <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Channels:</strong> {ds.channels}</p>}
                    {ds.sampling_rate && <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Sampling Rate:</strong> {ds.sampling_rate} Hz</p>}
                    <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Format:</strong> {ds.format}</p>
                    <p style={{ fontSize: 11, color: '#64748b', margin: '2px 0' }}><strong>Source:</strong> {ds.source}</p>
                    <span style={{
                      fontSize: 10, marginTop: 4, display: 'inline-block',
                      background: ds.is_downloaded ? '#dcfce7' : '#fef3c7',
                      color: ds.is_downloaded ? '#166534' : '#92400e',
                      borderRadius: 4, padding: '2px 6px'
                    }}>{ds.is_downloaded ? 'Downloaded' : 'Not Downloaded'}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  /* ── All Datasets Tab ── */
  const renderAllDatasets = () => {
    const bd = breakdown || { all_datasets: [] }
    const datasets = bd.all_datasets || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`All Datasets (${datasets.length})`}>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>Dataset</th>
                <th style={thStyle}>Disease</th>
                <th style={thStyle}>Subjects</th>
                <th style={thStyle}>Channels</th>
                <th style={thStyle}>Sampling Rate</th>
                <th style={thStyle}>Format</th>
                <th style={thStyle}>Source</th>
                <th style={thStyle}>Downloaded</th>
              </tr></thead>
              <tbody>
                {datasets.map((ds, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}>{i + 1}</td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{ds.name}</td>
                    <td style={{ ...tdStyle, textTransform: 'capitalize' }}>{ds.disease}</td>
                    <td style={tdStyle}>{ds.subjects}</td>
                    <td style={tdStyle}>{ds.channels || '-'}</td>
                    <td style={tdStyle}>{ds.sampling_rate ? `${ds.sampling_rate} Hz` : '-'}</td>
                    <td style={tdStyle}>{ds.format}</td>
                    <td style={{ ...tdStyle, fontSize: 11, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ds.source}</td>
                    <td style={tdStyle}>{ds.is_downloaded ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  /* ── Formats Tab ── */
  const renderFormats = () => {
    const fmts = overview.format_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Format Distribution">
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={fmts} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                {fmts.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Format Inventory">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={thStyle}>Format</th>
              <th style={thStyle}>Datasets Using</th>
            </tr></thead>
            <tbody>
              {fmts.map((f, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600 }}>{f.name}</td>
                  <td style={tdStyle}>{f.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Datasets per Disease" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={diseases} layout="vertical" margin={{ left: 100, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 11, textTransform: 'capitalize' }} />
              <Tooltip />
              <Bar dataKey="dataset_count" fill="#8b5cf6" name="Datasets" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
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
              {d.references.map((r, i) => <li key={i} style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>{r}</li>)}
            </ul>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Datasets Registry</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 16 }}>
        {k.total_diseases} diseases, {k.total_datasets} datasets, {k.total_subjects} total subjects — all from real EEG/clinical data sources
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
      {tab === 'all-datasets' && renderAllDatasets()}
      {tab === 'formats' && renderFormats()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
