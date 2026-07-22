import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const QUALITY_COLORS = { good: '#22c55e', moderate: '#eab308', heavy_artifact: '#ef4444' }
const COLORS = ['#3b82f6', '#22c55e', '#eab308', '#f97316', '#ef4444', '#8b5cf6']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function QualityBadge({ quality }) {
  const labels = { good: 'GOOD', moderate: 'MODERATE', heavy_artifact: 'HEAVY ARTIFACT' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (QUALITY_COLORS[quality] || '#94a3b8') + '22',
      color: QUALITY_COLORS[quality] || '#64748b'
    }}>{labels[quality] || quality}</span>
  )
}

function StageBadge({ status }) {
  const color = status === 'complete' ? '#22c55e' : (status === 'partial' ? '#eab308' : '#94a3b8')
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status.toUpperCase()}</span>
  )
}

export default function ICANoiseCleaningDashboard() {
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
          axios.get(`${API_URL}/api/ica-noise-cleaning/overview`),
          axios.get(`${API_URL}/api/ica-noise-cleaning/breakdown`),
          axios.get(`${API_URL}/api/ica-noise-cleaning/definitions`),
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ICA Noise Cleaning data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview || overview.total_files_processed === 0) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No ICA data available.'}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'files', label: 'Files & Components' },
    { id: 'pipeline', label: 'Pipeline' },
    { id: 'definitions', label: 'Definitions' },
  ]

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>ICA Noise Cleaning Pipeline</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Independent Component Analysis artifact removal — {overview.method}
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 2 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'files' && renderFiles()}
      {tab === 'pipeline' && renderPipeline()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const qualityData = Object.entries(overview.quality_distribution || {}).map(([k, v]) => ({
      name: k === 'heavy_artifact' ? 'Heavy Artifact' : k.charAt(0).toUpperCase() + k.slice(1),
      value: v,
      color: QUALITY_COLORS[k] || '#94a3b8'
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {/* KPI row */}
        <Card>
          <KPI label="Files Processed" value={overview.total_files_processed} sub="EDF recordings" />
        </Card>
        <Card>
          <KPI label="Subjects" value={overview.total_subjects} sub="CHB-MIT patients" />
        </Card>
        <Card>
          <KPI label="Mean Variance Removed" value={`${fmt(overview.mean_variance_removed_pct)}%`}
            sub="artifact signal" color={overview.mean_variance_removed_pct > 60 ? '#ef4444' : '#1e293b'} />
        </Card>
        <Card>
          <KPI label="Artifact Removal Rate" value={`${fmt(overview.artifact_removal_rate_pct)}%`}
            sub={`${overview.total_artifact_components_removed}/${overview.total_components_fitted} components`} />
        </Card>

        {/* Quality distribution pie */}
        <Card title="Signal Quality Distribution" span={2}>
          {qualityData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={qualityData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {qualityData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
        </Card>

        {/* Variance distribution bar */}
        <Card title="Variance Removed Distribution" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={overview.variance_distribution || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="range" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Files" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Per-subject table */}
        <Card title="Per-Subject Summary" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Subject', 'Files', 'Channels', 'Artifact Components', 'Avg Variance Removed', 'Quality'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.subject_summary || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.subject}</td>
                    <td style={{ padding: '8px 12px' }}>{s.n_files}</td>
                    <td style={{ padding: '8px 12px' }}>{s.n_channels}</td>
                    <td style={{ padding: '8px 12px' }}>{s.total_artifact_components}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(s.avg_variance_removed_pct)}%</td>
                    <td style={{ padding: '8px 12px' }}><QualityBadge quality={s.quality} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Method note */}
        <Card title="Method" span={4}>
          <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>{overview.method}</p>
          <p style={{ margin: '8px 0 0', fontSize: 12, color: '#94a3b8' }}>{overview.note}</p>
          <p style={{ margin: '8px 0 0', fontSize: 12, color: '#94a3b8' }}>
            Generated: {overview.generated_at} · DB analyses: {overview.db_analyses_count}
          </p>
        </Card>
      </div>
    )
  }

  function renderFiles() {
    if (!breakdown) return null
    const details = breakdown.file_details || []
    const stats = breakdown.component_stats || {}

    // Bar chart: variance removed per file
    const varChart = details.map(d => ({
      name: d.file.replace('.edf', ''),
      variance_removed: d.variance_removed_pct,
      signal_retained: d.signal_retained_pct,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {/* Component stats KPIs */}
        <Card>
          <KPI label="Avg Components" value={fmt(stats.avg_components)} sub={`range: ${stats.min_components}–${stats.max_components}`} />
        </Card>
        <Card>
          <KPI label="Avg Artifacts" value={fmt(stats.avg_artifacts)} sub={`range: ${stats.min_artifacts}–${stats.max_artifacts}`} />
        </Card>
        <Card>
          <KPI label="Channel Coverage" value={(breakdown.channel_coverage || []).map(c => `${c.n_channels}ch`).join(', ')} sub="montage" />
        </Card>
        <Card>
          <KPI label="Total Files" value={details.length} sub="processed" />
        </Card>

        {/* Variance per file bar chart */}
        <Card title="Variance Removed vs Signal Retained per File" span={4}>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={varChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
              <Tooltip formatter={v => `${v}%`} />
              <Bar dataKey="signal_retained" fill="#22c55e" name="Signal Retained %" radius={[4, 4, 0, 0]} />
              <Bar dataKey="variance_removed" fill="#ef4444" name="Variance Removed %" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Per-file detail table */}
        <Card title="Per-File ICA Detail" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['File', 'Subject', 'Channels', 'Components', 'Artifacts', 'Retained', 'Variance Removed', 'Signal Retained'].map(h => (
                    <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {details.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{d.file}</td>
                    <td style={{ padding: '8px 12px' }}>{d.subject}</td>
                    <td style={{ padding: '8px 12px' }}>{d.n_channels}</td>
                    <td style={{ padding: '8px 12px' }}>{d.n_components}</td>
                    <td style={{ padding: '8px 12px', color: '#ef4444', fontWeight: 600 }}>{d.artifact_components}</td>
                    <td style={{ padding: '8px 12px', color: '#22c55e', fontWeight: 600 }}>{d.retained_components}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(d.variance_removed_pct)}%</td>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(d.signal_retained_pct)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderPipeline() {
    if (!breakdown) return null
    const stages = breakdown.pipeline_stages || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="ICA Pipeline Stages" span={2}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {stages.map((s, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 16, padding: '12px 16px',
                background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0'
              }}>
                <div style={{
                  width: 32, height: 32, borderRadius: '50%', background: '#3b82f6',
                  color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 700, fontSize: 14, flexShrink: 0
                }}>{s.step}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{s.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.description}</div>
                </div>
                <StageBadge status={s.status} />
              </div>
            ))}
          </div>
        </Card>

        <Card title="Artifact Types Targeted" span={1}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { type: 'Eye Blinks (EOG)', desc: 'Detected via high kurtosis — sparse, large-amplitude frontal events', icon: '👁️' },
              { type: 'Muscle (EMG)', desc: 'High-frequency broadband activity, typically temporal/frontal channels', icon: '💪' },
              { type: 'Line Noise (60 Hz)', desc: 'Power line interference — narrowband, affects all channels', icon: '⚡' },
              { type: 'Channel Noise', desc: 'Single-channel high-variance components from poor electrode contact', icon: '📡' },
            ].map((a, i) => (
              <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 6 }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{a.icon} {a.type}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{a.desc}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Quality Thresholds" span={1}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { label: 'Good', range: '<40% variance removed', color: '#22c55e', desc: 'Clean recording, minimal artifact' },
              { label: 'Moderate', range: '40-60% variance removed', color: '#eab308', desc: 'Some artifact, review recommended' },
              { label: 'Heavy Artifact', range: '>60% variance removed', color: '#ef4444', desc: 'Heavily contaminated, manual review required' },
            ].map((q, i) => (
              <div key={i} style={{
                padding: '8px 12px', background: q.color + '11', borderRadius: 6,
                borderLeft: `3px solid ${q.color}`
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: q.color }}>{q.label}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{q.range}</div>
                <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{q.desc}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!defs) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={defs.title}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {(defs.definitions || []).map((d, i) => (
              <div key={i} style={{ padding: '10px 14px', background: i % 2 === 0 ? '#f8fafc' : '#fff', borderRadius: 6 }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{d.term}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4, lineHeight: 1.5 }}>{d.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }
}
