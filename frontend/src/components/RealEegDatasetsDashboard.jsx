import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = {
  downloaded: '#22c55e', partial: '#f97316', downloading: '#3b82f6',
  symlinked: '#8b5cf6', failed: '#ef4444', unknown: '#94a3b8'
}

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

export default function RealEegDatasetsDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/real-eeg-datasets/overview`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/real-eeg-datasets/breakdown`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/real-eeg-datasets/definitions`).then(r => r.data).catch(() => null),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDf(d); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Real EEG Datasets...</div>
  if (!ov?.available) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>real_eeg_datasets.yaml not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'primary', label: 'Primary Datasets' },
    { id: 'additional', label: 'Additional Datasets' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = ov.kpis || {}
  const ch = ov.charts || {}

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>{ov.title}</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>{ov.note} &middot; Updated {ov.updated_at}</p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600, cursor: 'pointer',
            border: tab === t.id ? '2px solid #3b82f6' : '1px solid #e2e8f0',
            background: tab === t.id ? '#eff6ff' : '#fff',
            color: tab === t.id ? '#1d4ed8' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card title="Key Metrics">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center' }}>
              <KPI label="Primary Diseases" value={k.primary_diseases} />
              <KPI label="Additional Datasets" value={k.additional_datasets} />
              <KPI label="Total Datasets" value={k.total_datasets} />
              <KPI label="Total Subjects" value={k.total_subjects} />
              <KPI label="Total Files" value={k.total_files} />
              <KPI label="Total Size" value={k.total_size} />
              <KPI label="Downloaded" value={k.downloaded} />
              <KPI label="Partial" value={k.partial} />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340, 1fr))', gap: 16 }}>
            <Card title="Download Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={ch.status_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {(ch.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Subjects per Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ch.subjects_per_disease || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="subjects" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Files per Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ch.files_per_disease || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="files" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Data Format Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={ch.format_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {(ch.format_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Source Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={ch.source_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {(ch.source_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="All Primary Datasets">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Dataset</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Subjects</th>
                    <th style={thStyle}>Files</th>
                    <th style={thStyle}>Size</th>
                    <th style={thStyle}>Source</th>
                    <th style={thStyle}>License</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.primary_datasets || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{r.disease}</td>
                      <td style={tdStyle}>{r.format}</td>
                      <td style={{ ...tdStyle, textAlign: 'right' }}>{r.subjects}</td>
                      <td style={{ ...tdStyle, textAlign: 'right' }}>{r.files}</td>
                      <td style={tdStyle}>{r.size}</td>
                      <td style={tdStyle}>{r.source}</td>
                      <td style={tdStyle}>{r.license}</td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'primary' && bd?.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 16 }}>
          {Object.entries(bd.primary || {}).map(([key, ds]) => (
            <Card key={key} title={ds.name}>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.7 }}>
                <div><strong>Disease:</strong> {ds.disease}</div>
                <div><strong>Format:</strong> {ds.format}</div>
                <div><strong>Subjects:</strong> {ds.subjects}</div>
                <div><strong>Files:</strong> {ds.files}</div>
                <div><strong>Size:</strong> {ds.size}</div>
                <div><strong>Source:</strong> {ds.source}</div>
                <div><strong>License:</strong> {ds.license}</div>
                <div><strong>Path:</strong> <code style={{ fontSize: 11, background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{ds.path}</code></div>
                {ds.url && <div><strong>URL:</strong> <span style={{ fontSize: 11, color: '#3b82f6', wordBreak: 'break-all' }}>{ds.url}</span></div>}
                {ds.symlink_target && <div><strong>Symlink:</strong> <code style={{ fontSize: 11, background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{ds.symlink_target}</code></div>}
                {ds.notes && <div style={{ marginTop: 6, padding: '6px 8px', background: '#fffbeb', borderRadius: 4, fontSize: 11, color: '#92400e' }}>{ds.notes}</div>}
                <div style={{ marginTop: 8 }}><StatusBadge status={ds.status} /></div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {tab === 'additional' && bd?.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 16 }}>
          {Object.entries(bd.additional || {}).map(([key, ds]) => (
            <Card key={key} title={ds.name || key.replace(/_/g, ' ')}>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.7 }}>
                <div><strong>Disease:</strong> {ds.disease}</div>
                <div><strong>Format:</strong> {ds.format}</div>
                <div><strong>Subjects:</strong> {ds.subjects}</div>
                <div><strong>Files:</strong> {ds.files}</div>
                {ds.records > 0 && <div><strong>Records:</strong> {ds.records}</div>}
                <div><strong>Size:</strong> {ds.size}</div>
                <div><strong>Source:</strong> {ds.source}</div>
                <div><strong>License:</strong> {ds.license}</div>
                <div><strong>Path:</strong> <code style={{ fontSize: 11, background: '#f1f5f9', padding: '1px 4px', borderRadius: 3 }}>{ds.path}</code></div>
                {ds.url && <div><strong>URL:</strong> <span style={{ fontSize: 11, color: '#3b82f6', wordBreak: 'break-all' }}>{ds.url}</span></div>}
                {ds.notes && <div style={{ marginTop: 6, padding: '6px 8px', background: '#fffbeb', borderRadius: 4, fontSize: 11, color: '#92400e' }}>{ds.notes}</div>}
                <div style={{ marginTop: 8 }}><StatusBadge status={ds.status} /></div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {tab === 'definitions' && df?.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          <Card title="Status Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Status</th><th style={thStyle}>Meaning</th></tr></thead>
              <tbody>
                {(df.status_legend || []).map((s, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><StatusBadge status={s.status} /></td>
                    <td style={tdStyle}>{s.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Term</th><th style={thStyle}>Definition</th></tr></thead>
              <tbody>
                {(df.glossary || []).map((g, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
              {(df.clinical_notes || []).map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          </Card>

          <Card title="References">
            {(df.references || []).map((r, i) => (
              <div key={i} style={{ padding: '6px 0', borderBottom: i < df.references.length - 1 ? '1px solid #f1f5f9' : 'none', fontSize: 12 }}>
                <strong>{r.name}</strong>
                {r.path && <span style={{ color: '#64748b' }}> &mdash; {r.path}</span>}
                {r.role && <span style={{ color: '#64748b' }}> &mdash; {r.role}</span>}
                {r.note && <span style={{ color: '#64748b' }}> &mdash; {r.note}</span>}
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
