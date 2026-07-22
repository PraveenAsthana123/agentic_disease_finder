import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function SeverityBadge({ level }) {
  const l = String(level || '').toLowerCase()
  const colors = { severe: '#dc2626', moderate: '#f59e0b', mild: '#22c55e' }
  const color = colors[l] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(level || 'N/A')}</span>
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

export default function ArtifactAnnotationDashboard() {
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
      axios.get(`${API_URL}/api/artifact-annotations/overview`),
      axios.get(`${API_URL}/api/artifact-annotations/breakdown`),
      axios.get(`${API_URL}/api/artifact-annotations/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading artifact annotations...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Artifact annotation data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = ov.kpis || {}

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
          <Card title="Artifact Annotation Summary" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Annotations" value={k.total_annotations} color="#1e293b" />
              <KPI label="Unique Patients" value={k.unique_patients} color="#3b82f6" />
              <KPI label="Artifact Types" value={k.artifact_types} color="#8b5cf6" />
              <KPI label="Avg Duration" value={k.avg_duration} sub="seconds" color="#f59e0b" />
              <KPI label="Severe %" value={k.severe_pct} sub="of total" color="#ef4444" />
            </div>
          </Card>

          <Card title="Artifact Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ov.artifact_type_distribution || []} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="type" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Severity Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={ov.severity_distribution || []} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="severity" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.severity_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Channel Distribution (Top 15)" span={2}>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={(ov.channel_distribution || []).slice(0, 15)} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="channel" width={90} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Severity by Artifact Type">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={ov.severity_by_type || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 10, angle: -30 }} height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="mild" stackId="a" fill="#22c55e" />
                <Bar dataKey="moderate" stackId="a" fill="#f59e0b" />
                <Bar dataKey="severe" stackId="a" fill="#dc2626" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Trend" span={3}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={ov.monthly_trend || []} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'breakdown' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient ID', 'Total', 'Types Breakdown', 'Avg Duration'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_patient || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'left' }}>{fmt(r.total)}</td>
                      <td style={{ padding: '6px 10px' }}>{r.types_breakdown || r.types || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(r.avg_duration)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Type-by-Channel Cross-Tab">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Artifact Type</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Channel</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.type_channel_crosstab || []).slice(0, 30).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px' }}>{r.type || r.artifact_type}</td>
                      <td style={{ padding: '6px 10px' }}>{r.channel}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{fmt(r.count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Duration by Type">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Artifact Type', 'Avg Duration', 'Min Duration', 'Max Duration'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.duration_by_type || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.type || r.artifact_type}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(r.avg)}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(r.min)}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(r.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Annotations (Last 20)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'Type', 'Channel', 'Start', 'Duration', 'Severity'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.recent_annotations || []).slice(0, 20).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.type || r.artifact_type}</td>
                      <td style={{ padding: '6px 10px' }}>{r.channel}</td>
                      <td style={{ padding: '6px 10px' }}>{r.start || r.start_time || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(r.duration)}</td>
                      <td style={{ padding: '6px 10px' }}><SeverityBadge level={r.severity} /></td>
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
          <Card title="Artifact Types">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 220 }}>Artifact Type</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(Array.isArray(defs.artifact_types) ? defs.artifact_types : Object.entries(defs.artifact_types || {})).map((item, i) => {
                  const name = Array.isArray(item) ? item[0] : (item.name || item.type)
                  const desc = Array.isArray(item) ? item[1] : (item.description || item.desc)
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{name}</td>
                      <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{desc}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>

          <Card title="Severity Levels">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 180 }}>Level</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(Array.isArray(defs.severity_levels) ? defs.severity_levels : Object.entries(defs.severity_levels || {})).map((item, i) => {
                  const name = Array.isArray(item) ? item[0] : (item.name || item.level)
                  const desc = Array.isArray(item) ? item[1] : (item.description || item.desc)
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}><SeverityBadge level={name} /></td>
                      <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{desc}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 180 }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(defs.glossary || {}).map(([k, v], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{k}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8, color: '#334155' }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.references || []).map((r, i) => (
                <li key={i} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8, color: '#334155' }}>{typeof r === 'string' ? r : r.title || r.name || JSON.stringify(r)}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
