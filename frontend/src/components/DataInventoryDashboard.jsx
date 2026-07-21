import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const STATUS_COLORS = {
  complete: '#22c55e', downloaded: '#22c55e', symlinked: '#22c55e',
  partial: '#f97316', downloading: '#3b82f6',
  failed: '#ef4444', missing: '#ef4444',
  simulated: '#8b5cf6', 'cache only': '#94a3b8', 'index only': '#94a3b8',
  'n/a': '#cbd5e1'
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

export default function DataInventoryDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/data-inventory/overview`),
      axios.get(`${API_URL}/data-inventory/breakdown`),
      axios.get(`${API_URL}/data-inventory/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Data Inventory...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#94a3b8' }}>{overview.note}</div>

  const tabs = ['overview', 'diseases', 'additional', 'downloads', 'definitions']
  const kpis = overview.kpis || {}
  const charts = overview.charts || {}

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
            {t === 'overview' ? 'Overview' : t === 'diseases' ? 'Per-Disease' : t === 'additional' ? 'Additional Datasets' : t === 'downloads' ? 'Download Status' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap' }}>
              <KPI label="Diseases" value={kpis.total_diseases} sub="neurological" />
              <KPI label="Additional Datasets" value={kpis.additional_datasets} />
              <KPI label="Synthetic Records" value={kpis.total_synthetic_records} sub="augmented" />
              <KPI label="Real EEG Files" value={kpis.total_real_eeg_files} />
              <KPI label="Real EEG Size" value={kpis.total_real_eeg_size} />
              <KPI label="Total Disk" value={kpis.total_disk} />
              <KPI label="Downloaded" value={kpis.downloaded} sub="datasets" />
              <KPI label="Failed" value={kpis.failed} sub="downloads" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Download Status Pie */}
            <Card title="Download Status Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.download_status || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.download_status || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Real EEG Status Pie */}
            <Card title="Real EEG Status per Disease">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={charts.real_eeg_status || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {(charts.real_eeg_status || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Subjects per Disease Bar */}
            <Card title="Real EEG Subjects per Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={charts.subjects_per_disease || []} margin={{ left: 0, right: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="subjects" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Synthetic Records Bar */}
            <Card title="Synthetic Records per Disease">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={charts.synthetic_records || []} margin={{ left: 0, right: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="records" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Summary Table */}
          <Card title="All Diseases Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Disease</th>
                    <th style={thStyle}>Synthetic</th>
                    <th style={thStyle}>Records</th>
                    <th style={thStyle}>Real EEG</th>
                    <th style={thStyle}>Size</th>
                    <th style={thStyle}>Subjects</th>
                    <th style={thStyle}>Format</th>
                    <th style={thStyle}>Real Status</th>
                    <th style={thStyle}>Ext. Valid.</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.disease_summary || []).map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{r.disease}</td>
                      <td style={tdStyle}>{r.synthetic_files} files</td>
                      <td style={tdStyle}>{r.synthetic_records}</td>
                      <td style={tdStyle}>{r.real_files} files</td>
                      <td style={tdStyle}>{r.real_size}</td>
                      <td style={tdStyle}>{r.real_subjects}</td>
                      <td style={tdStyle}>{r.real_format}</td>
                      <td style={tdStyle}><StatusBadge status={r.real_status} /></td>
                      <td style={tdStyle}><StatusBadge status={r.ext_status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* PER-DISEASE TAB */}
      {tab === 'diseases' && breakdown && (
        <>
          {Object.entries(breakdown.diseases || {}).map(([disease, info]) => (
            <Card key={disease} title={disease.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
                {/* Synthetic */}
                {info.synthetic && Object.keys(info.synthetic).length > 0 && (
                  <div style={{ background: '#f0fdf4', borderRadius: 6, padding: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#166534', marginBottom: 6 }}>Synthetic Data</div>
                    <div style={{ fontSize: 11, color: '#334155' }}>
                      <div>Path: <code style={{ fontSize: 10 }}>{info.synthetic.path}</code></div>
                      <div>Files: {info.synthetic.files} &middot; Size: {info.synthetic.size}</div>
                      <div>Records: {info.synthetic.records}</div>
                      <div>Format: {info.synthetic.format} &middot; <StatusBadge status={info.synthetic.status} /></div>
                    </div>
                  </div>
                )}
                {/* Real EEG */}
                {info.real_eeg && Object.keys(info.real_eeg).length > 0 && (
                  <div style={{ background: '#eff6ff', borderRadius: 6, padding: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#1e40af', marginBottom: 6 }}>Real EEG</div>
                    <div style={{ fontSize: 11, color: '#334155' }}>
                      <div>Path: <code style={{ fontSize: 10 }}>{info.real_eeg.path}</code></div>
                      <div>Files: {info.real_eeg.files} &middot; Size: {info.real_eeg.size}</div>
                      <div>Source: {info.real_eeg.source}</div>
                      <div>Subjects: {info.real_eeg.subjects} &middot; Format: {info.real_eeg.format}</div>
                      {info.real_eeg.license && <div>License: {info.real_eeg.license}</div>}
                      <div><StatusBadge status={info.real_eeg.status} /></div>
                      {info.real_eeg.notes && <div style={{ marginTop: 4, fontStyle: 'italic', fontSize: 10, color: '#64748b' }}>{info.real_eeg.notes}</div>}
                    </div>
                  </div>
                )}
                {/* External Validation */}
                {info.external_validation && Object.keys(info.external_validation).length > 0 && (
                  <div style={{ background: '#faf5ff', borderRadius: 6, padding: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#7c3aed', marginBottom: 6 }}>External Validation</div>
                    <div style={{ fontSize: 11, color: '#334155' }}>
                      <div>Path: <code style={{ fontSize: 10 }}>{info.external_validation.path}</code></div>
                      <div>Files: {info.external_validation.files} &middot; Size: {info.external_validation.size}</div>
                      {info.external_validation.records && <div>Records: {info.external_validation.records}</div>}
                      <div><StatusBadge status={info.external_validation.status} /></div>
                    </div>
                  </div>
                )}
                {/* Extra datasets (eeg_datasets, ppmi, adni, etc.) */}
                {Object.entries(info).filter(([k]) => !['synthetic', 'real_eeg', 'external_validation'].includes(k) && typeof info[k] === 'object' && info[k] !== null).map(([k, v]) => (
                  <div key={k} style={{ background: '#fefce8', borderRadius: 6, padding: 12 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#854d0e', marginBottom: 6 }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                    <div style={{ fontSize: 11, color: '#334155' }}>
                      {v.path && <div>Path: <code style={{ fontSize: 10 }}>{v.path}</code></div>}
                      {v.files !== undefined && <div>Files: {v.files}{v.size ? ` \u00b7 Size: ${v.size}` : ''}</div>}
                      {v.source && <div>Source: {v.source}</div>}
                      {v.status && <div><StatusBadge status={v.status} /></div>}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ADDITIONAL DATASETS TAB */}
      {tab === 'additional' && (
        <>
          {(overview.additional_datasets || []).map((ds, i) => (
            <Card key={i} title={ds.name}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
                <div style={{ fontSize: 12, color: '#334155' }}>
                  <strong>Files:</strong> {ds.files} &middot; <strong>Size:</strong> {ds.size}
                </div>
                <div style={{ fontSize: 12, color: '#334155' }}>
                  <strong>Source:</strong> {ds.source}
                </div>
                <div style={{ fontSize: 12, color: '#334155' }}>
                  <strong>Subjects:</strong> {ds.subjects} &middot; <strong>Format:</strong> {ds.format}
                </div>
                <div><StatusBadge status={ds.status} /></div>
              </div>
            </Card>
          ))}

          {/* General datasets */}
          {breakdown && breakdown.general && Object.keys(breakdown.general).length > 0 && (
            <Card title="General / Shared Datasets">
              {Object.entries(breakdown.general).map(([k, v]) => (
                <div key={k} style={{ background: '#f8fafc', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>
                    {v.path && <span>Path: <code style={{ fontSize: 10 }}>{v.path}</code> &middot; </span>}
                    {v.records && <span>Records: {v.records} &middot; </span>}
                    {v.source && <span>Source: {v.source} &middot; </span>}
                    {v.format && <span>Format: {v.format} &middot; </span>}
                    {v.license && <span>License: {v.license} &middot; </span>}
                    {v.status && <StatusBadge status={v.status} />}
                  </div>
                </div>
              ))}
            </Card>
          )}

          {/* Disk usage */}
          {breakdown && breakdown.disk_usage && (
            <Card title="Disk Usage">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
                {Object.entries(breakdown.disk_usage).map(([k, v]) => (
                  <div key={k} style={{ background: '#f1f5f9', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                    <div style={{ fontSize: 18, fontWeight: 700, color: '#1e293b' }}>{v}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* DOWNLOAD STATUS TAB */}
      {tab === 'downloads' && breakdown && (
        <>
          {Object.entries(breakdown.download_status || {}).map(([category, items]) => (
            <Card key={category} title={category.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              {Array.isArray(items) && items.length > 0 ? (
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {items.map((item, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{item}</li>
                  ))}
                </ul>
              ) : (
                <div style={{ fontSize: 12, color: '#94a3b8' }}>None</div>
              )}
            </Card>
          ))}
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Status Legend">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                  <StatusBadge status={s.status} />
                  <span style={{ color: '#334155' }}>{s.meaning}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ fontSize: 12, color: '#334155' }}>
                  <strong style={{ color: '#1e293b' }}>{g.term}:</strong> {g.definition}
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 8 }}>
              {(defs.references || []).map((r, i) => (
                <div key={i} style={{ fontSize: 12, color: '#334155' }}>
                  <strong>{r.name}</strong> {r.path ? <code style={{ fontSize: 10 }}>{r.path}</code> : ''} {r.note ? `\u2014 ${r.note}` : ''} {r.role ? `\u2014 ${r.role}` : ''}
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
