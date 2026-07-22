import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function ActionBadge({ action }) {
  const a = String(action || '').toLowerCase()
  const color = a === 'retain' ? '#10b981' : a === 'exclude' ? '#ef4444' : a === 'review' ? '#f59e0b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(action || 'N/A')}</span>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'ok' ? '#10b981' : s === 'error' ? '#ef4444' : '#f59e0b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(status || 'N/A')}</span>
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

export default function ICALabelDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/ica-label/overview`),
      axios.get(`${API_URL}/api/ica-label/definitions`),
    ])
      .then(([o, d]) => { setOv(o.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ICLabel ICA classification...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>ICLabel data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'subjects', label: 'Per Subject' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const brainArtifactData = [
    { name: 'Brain', value: ov.total_brain || 0 },
    { name: 'Artifact', value: ov.total_artifact || 0 },
  ]

  const subjectData = (ov.per_subject || []).filter(s => s.status === 'ok').map(s => ({
    subject: s.subject,
    brain: s.brain,
    artifact: s.artifact,
    brain_ratio: s.brain_ratio,
    variance_removed_pct: s.variance_removed_pct,
  }))

  const varianceData = (ov.per_subject || []).filter(s => s.status === 'ok').map(s => ({
    name: s.subject,
    variance: s.variance_removed_pct || 0,
  }))

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
          <Card title="ICLabel ICA Classification KPIs" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Subjects Analyzed" value={ov.subjects_analyzed} color="#1e293b" />
              <KPI label="Total Components" value={ov.total_components} color="#3b82f6" />
              <KPI label="Brain Components" value={ov.total_brain} color="#10b981" />
              <KPI label="Artifact Components" value={ov.total_artifact} color="#ef4444" />
              <KPI label="Brain Ratio" value={ov.overall_brain_ratio} color="#8b5cf6" />
              <KPI label="Variance Removed" value={ov.mean_variance_removed_pct} sub="%" color="#f59e0b" />
            </div>
          </Card>

          {/* Brain vs Artifact Pie */}
          <Card title="Brain vs Artifact Components">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={brainArtifactData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  <Cell fill="#10b981" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Subject Brain/Artifact Stacked Bar */}
          <Card title="Components per Subject">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={subjectData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="brain" stackId="a" fill="#10b981" name="Brain" />
                <Bar dataKey="artifact" stackId="a" fill="#ef4444" name="Artifact" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Variance Removed Bar */}
          <Card title="Variance Removed by Subject (%)">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={varianceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="variance" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Variance %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Method Info */}
          <Card title="Classification Method" span={3}>
            <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
              <strong>Method:</strong> {ov.method || '--'}
            </p>
            {ov.note && (
              <p style={{ margin: '8px 0 0', fontSize: 13, color: '#64748b', lineHeight: 1.6 }}>
                <strong>Note:</strong> {ov.note}
              </p>
            )}
            {ov.source && (
              <p style={{ margin: '8px 0 0', fontSize: 12, color: '#94a3b8' }}>
                Source: {ov.source}
              </p>
            )}
          </Card>
        </div>
      )}

      {tab === 'subjects' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Per-Subject ICA Classification (${subjectData.length} subjects)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Subject', 'File', 'Channels', 'Components', 'Brain', 'Artifact', 'Brain Ratio', 'Variance Removed %', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(ov.per_subject || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.subject}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{s.file || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(s.n_channels)}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(s.n_components)}</td>
                      <td style={{ padding: '6px 10px', color: '#10b981', fontWeight: 600 }}>{fmt(s.brain)}</td>
                      <td style={{ padding: '6px 10px', color: '#ef4444', fontWeight: 600 }}>{fmt(s.artifact)}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(s.brain_ratio)}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace' }}>{fmt(s.variance_removed_pct)}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={s.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Summary Stats */}
          <Card title="Aggregate Statistics">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Overall Brain Ratio" value={ov.overall_brain_ratio} color="#10b981" />
              <KPI label="Mean Variance Removed" value={ov.mean_variance_removed_pct} sub="%" color="#f59e0b" />
              <KPI label="Total Brain" value={ov.total_brain} color="#10b981" />
              <KPI label="Total Artifact" value={ov.total_artifact} color="#ef4444" />
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="ICLabel Component Categories">
            <p style={{ margin: '0 0 12px', fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
              {defs.method_description}
            </p>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Category', 'Description', 'Action'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.categories || []).map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>
                      <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: c.color, marginRight: 8, verticalAlign: 'middle' }} />
                      {c.name}
                    </td>
                    <td style={{ padding: '6px 10px' }}>{c.description}</td>
                    <td style={{ padding: '6px 10px' }}><ActionBadge action={c.action} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {defs.interpretation && (
            <Card title="Interpretation Guide">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 200 }}>Metric</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Threshold / Guideline</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(defs.interpretation).map(([key, val], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{key.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '6px 10px' }}>{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
