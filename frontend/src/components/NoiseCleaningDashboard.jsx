import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : v.toFixed(1) + '%' }

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
  const colorMap = {
    minimal: '#22c55e', low: '#22c55e', ok: '#22c55e',
    moderate: '#3b82f6',
    significant: '#f97316', warning: '#eab308',
    heavy: '#ef4444', extreme: '#ef4444', critical: '#ef4444',
    clean: '#22c55e', artifact: '#ef4444'
  }
  const color = colorMap[status?.toLowerCase()] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function NoiseCleaningDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/noise-cleaning/overview`),
          axios.get(`${API_URL}/noise-cleaning/breakdown`),
          axios.get(`${API_URL}/noise-cleaning/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Noise Cleaning data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'subjects', label: 'Subject Comparison' },
    { id: 'files', label: 'Per-File Details' },
    { id: 'quality', label: 'Quality Tiers' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  const kpis = overview?.kpis || []
  const varianceDist = overview?.variance_distribution || []
  const perSubject = overview?.per_subject_summary || []
  const timeline = overview?.timeline || []
  const perFile = breakdown?.per_file_details || []
  const channelStats = breakdown?.channel_stats || {}
  const componentStats = breakdown?.component_stats || {}
  const subjectComparison = breakdown?.subject_comparison || []
  const qualityTiers = breakdown?.quality_tiers || []
  const defs = definitions?.metrics || []
  const methodology = definitions?.methodology || ''
  const qualityNotes = definitions?.quality_notes || []

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>EEG Noise Cleaning Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          ICA artifact removal &amp; denoising — {overview?.method || 'MNE ICA pipeline'}
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${kpis.length || 4}, 1fr)`, gap: 16 }}>
              {kpis.map((k, i) => (
                <KPI key={i} label={k.label} value={k.unit === '%' ? fmtPct(k.value) : fmt(k.value)} sub={k.unit !== '%' ? k.unit : undefined} color={COLORS[i % COLORS.length]} />
              ))}
            </div>
          </Card>

          <Card title="Variance Removed Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={varianceDist} dataKey="count" nameKey="range" cx="50%" cy="50%" outerRadius={80} label={({ range, count }) => `${range}: ${count}`}>
                  {varianceDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Subject Average Variance Removed" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={perSubject}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <YAxis unit="%" />
                <Tooltip formatter={(v) => fmtPct(v)} />
                <Bar dataKey="avg_variance_removed" fill="#3b82f6" name="Avg Variance Removed %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Artifact Components by File">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={timeline}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="file" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="artifact_components_removed" fill="#f97316" name="Artifacts Removed" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'subjects' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Component & Channel Stats" span={1}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <KPI label="Avg Channels" value={fmt(channelStats.avg_channels)} color="#3b82f6" />
              <KPI label="Avg ICA Components" value={fmt(componentStats.avg_components)} color="#8b5cf6" />
              <KPI label="Avg Artifacts Removed" value={fmt(componentStats.avg_artifacts)} color="#f97316" />
              <KPI label="Artifact Ratio" value={fmtPct(componentStats.artifact_ratio_pct)} color="#ef4444" />
            </div>
          </Card>

          <Card title="Subject Comparison — Variance Removed" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={subjectComparison} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" unit="%" />
                <YAxis type="category" dataKey="subject" width={80} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => fmtPct(v)} />
                <Legend />
                <Bar dataKey="variance_removed_pct" fill="#8b5cf6" name="Variance Removed %" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Artifact Components Per Subject" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={subjectComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="artifact_components" fill="#ef4444" name="Artifact Components" radius={[4, 4, 0, 0]} />
                <Bar dataKey="n_files" fill="#22c55e" name="Files Processed" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'files' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Per-File Details (${perFile.length} files)`}>
            <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>File</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Subject</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Channels</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>ICA Components</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Artifacts Removed</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right' }}>Variance Removed</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center' }}>Quality Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {perFile.map((f, i) => {
                    const vr = f.variance_removed_pct || 0
                    const tier = vr < 20 ? 'Minimal' : vr < 40 ? 'Moderate' : vr < 60 ? 'Significant' : vr < 80 ? 'Heavy' : 'Extreme'
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.file}</td>
                        <td style={{ padding: '8px 10px' }}>{f.subject}</td>
                        <td style={{ padding: '8px 10px', textAlign: 'right' }}>{fmt(f.n_channels)}</td>
                        <td style={{ padding: '8px 10px', textAlign: 'right' }}>{fmt(f.n_components)}</td>
                        <td style={{ padding: '8px 10px', textAlign: 'right', color: '#f97316' }}>{fmt(f.artifact_components_removed)}</td>
                        <td style={{ padding: '8px 10px', textAlign: 'right' }}>{fmtPct(vr)}</td>
                        <td style={{ padding: '8px 10px', textAlign: 'center' }}><StatusBadge status={tier} /></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Variance Removed per File">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={perFile}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="file" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis unit="%" />
                <Tooltip formatter={(v) => fmtPct(v)} />
                <Area type="monotone" dataKey="variance_removed_pct" stroke="#8b5cf6" fill="#8b5cf622" strokeWidth={2} name="Variance Removed %" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'quality' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Quality Tier Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={qualityTiers}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="tier" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {qualityTiers.map((_, i) => <Cell key={i} fill={['#22c55e','#3b82f6','#f97316','#ef4444','#7f1d1d'][i] || COLORS[i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Tier Descriptions" span={2}>
            <div style={{ display: 'grid', gap: 10 }}>
              {qualityTiers.map((t, i) => (
                <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: `3px solid ${['#22c55e','#3b82f6','#f97316','#ef4444','#7f1d1d'][i] || '#94a3b8'}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <span style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{t.tier}</span>
                    <span style={{ fontSize: 12, color: '#64748b', marginLeft: 12 }}>{t.description}</span>
                  </div>
                  <span style={{ fontWeight: 700, fontSize: 18, color: '#334155' }}>{t.count}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Variance Removed Histogram" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={varianceDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Methodology">
            <div style={{ padding: '12px 16px', background: '#f0f9ff', borderRadius: 8, borderLeft: '3px solid #3b82f6', fontSize: 13, color: '#1e3a5f', lineHeight: 1.6 }}>
              {methodology}
            </div>
          </Card>

          <Card title="Metric Definitions">
            <div style={{ display: 'grid', gap: 12 }}>
              {defs.map((d, i) => (
                <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>
                    {d.name} {d.unit && <span style={{ fontSize: 11, color: '#94a3b8', fontWeight: 400 }}>({d.unit})</span>}
                  </div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{d.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Notes & Caveats">
            <div style={{ display: 'grid', gap: 8 }}>
              {qualityNotes.map((n, i) => (
                <div key={i} style={{ padding: '10px 14px', background: '#fffbeb', borderRadius: 8, borderLeft: '3px solid #eab308', fontSize: 13, color: '#78350f', lineHeight: 1.5 }}>
                  {n}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
