import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function SeverityBadge({ severity }) {
  const s = String(severity || '').toLowerCase()
  const color = s === 'severe' ? '#ef4444' : s === 'moderate' ? '#f59e0b' : s === 'mild' ? '#10b981' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(severity || 'N/A')}</span>
  )
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function ArtifactAnnotationsDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/artifact-annotations/overview`),
      axios.get(`${API_URL}/artifact-annotations/breakdown`),
      axios.get(`${API_URL}/artifact-annotations/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading artifact annotations...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Artifact annotations data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'channels', label: 'Channel × Type' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = ov.kpis || {}
  const typeDist = Object.entries(ov.artifact_type_distribution || {}).map(([name, value]) => ({ name, value }))
  const sevDist = Object.entries(ov.severity_distribution || {}).map(([name, value]) => ({ name, value }))
  const chanDist = Object.entries(ov.channel_distribution || {}).map(([name, value]) => ({ name, value }))
  const sevByType = ov.severity_by_type || []

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="Artifact Annotation KPIs" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Annotations" value={k.total_annotations} color="#1e293b" />
              <KPI label="Unique Patients" value={k.unique_patients} color="#3b82f6" />
              <KPI label="Artifact Types" value={k.artifact_types} color="#8b5cf6" />
              <KPI label="Avg Duration" value={k.avg_duration_sec} sub="seconds" color="#06b6d4" />
              <KPI label="Severe" value={k.severe_pct} sub="%" color="#ef4444" />
            </div>
          </Card>

          {/* Artifact Type Distribution Pie */}
          <Card title="Artifact Type Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={typeDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {typeDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Severity Distribution Pie */}
          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={sevDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {sevDist.map((_, i) => <Cell key={i} fill={['#10b981', '#f59e0b', '#ef4444'][i] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Channel Distribution Bar */}
          <Card title="Channel Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chanDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Severity by Type Stacked Bar */}
          <Card title="Severity by Artifact Type" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={sevByType}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="mild" stackId="a" fill="#10b981" />
                <Bar dataKey="moderate" stackId="a" fill="#f59e0b" />
                <Bar dataKey="severe" stackId="a" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Duration by Type */}
          {bd && bd.duration_by_type && (
            <Card title="Duration by Artifact Type">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Type', 'Avg (s)', 'Min (s)', 'Max (s)'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bd.duration_by_type.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.type}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(d.avg_duration)}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(d.min_duration)}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(d.max_duration)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}

      {tab === 'patients' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Per-Patient Artifact Summary (${(bd.per_patient || []).length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Total', 'Avg Duration (s)', 'Mild', 'Moderate', 'Severe', 'Types'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_patient || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{p.total}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(p.avg_duration)}</td>
                      <td style={{ padding: '6px 10px' }}>{(p.severities || {}).mild || 0}</td>
                      <td style={{ padding: '6px 10px' }}>{(p.severities || {}).moderate || 0}</td>
                      <td style={{ padding: '6px 10px' }}>{(p.severities || {}).severe || 0}</td>
                      <td style={{ padding: '6px 10px', fontSize: 12 }}>{Object.keys(p.types || {}).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Recent Annotations (${(bd.recent_annotations || []).length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['ID', 'Patient', 'Type', 'Channel', 'Start (min)', 'Duration (s)', 'Severity', 'Created'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.recent_annotations || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px' }}>{a.id}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{a.artifact_type}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{a.channel}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(a.start_time_min)}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(a.duration_sec)}</td>
                      <td style={{ padding: '6px 10px' }}><SeverityBadge severity={a.severity} /></td>
                      <td style={{ padding: '6px 10px', fontSize: 12 }}>{a.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'channels' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Artifact Type × Channel Matrix">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Channel</th>
                    {['muscle', 'ECG', 'electrode_pop', 'movement', 'eye_blink', 'sweat'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.type_by_channel || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600, fontFamily: 'monospace' }}>{row.channel}</td>
                      {['muscle', 'ECG', 'electrode_pop', 'movement', 'eye_blink', 'sweat'].map(col => {
                        const val = row[col] || 0
                        const bg = val === 0 ? '#f8fafc' : val <= 2 ? '#dcfce7' : val <= 4 ? '#fef9c3' : '#fecaca'
                        return (
                          <td key={col} style={{ padding: '6px 10px', textAlign: 'center', background: bg, fontFamily: 'monospace' }}>
                            {val}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {defs.artifact_types && (
            <Card title="Artifact Types">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 140 }}>Type</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.artifact_types.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.name}</td>
                      <td style={{ padding: '6px 10px' }}>{d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {defs.severity_levels && (
            <Card title="Severity Levels">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 140 }}>Level</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.severity_levels.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px' }}><SeverityBadge severity={d.level} /></td>
                      <td style={{ padding: '6px 10px' }}>{d.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {defs.glossary && (
            <Card title="Glossary">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 160 }}>Term</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.glossary.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.term}</td>
                      <td style={{ padding: '6px 10px' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {defs.clinical_notes && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.clinical_notes.map((n, i) => (
                  <li key={i} style={{ marginBottom: 6, fontSize: 13 }}>{typeof n === 'string' ? n : n.note || JSON.stringify(n)}</li>
                ))}
              </ul>
            </Card>
          )}

          {defs.references && (
            <Card title="References">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.references.map((r, i) => (
                  <li key={i} style={{ marginBottom: 6, fontSize: 13 }}>{typeof r === 'string' ? r : r.title || JSON.stringify(r)}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
