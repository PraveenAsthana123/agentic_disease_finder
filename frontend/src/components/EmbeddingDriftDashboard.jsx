import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function driftColor(value) {
  if (value == null) return '#94a3b8'
  if (value < 0.05) return '#4caf50'
  if (value < 0.15) return '#ff9800'
  return '#f44336'
}

function statusLabel(value) {
  if (value == null) return 'unknown'
  if (value < 0.05) return 'stable'
  if (value < 0.15) return 'moderate'
  return 'drifted'
}

function fmt(v, decimals = 4) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function EmbeddingDriftDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/embedding-drift/overview`),
          axios.get(`${API_URL}/api/embedding-drift/breakdown`),
          axios.get(`${API_URL}/api/embedding-drift/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load embedding drift data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Analyzing embedding drift across vector space...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Embedding drift monitoring not available. Ensure reference and current embeddings are present.'}
    </div>
  )

  const summary = overview.summary || {}
  const metadata = overview.metadata || {}
  const driftTimeline = breakdown?.drift_over_time || []
  const dimensionDrift = breakdown?.dimension_drift || []
  const segments = breakdown?.corpus_segments || []
  const distribution = breakdown?.drift_distribution || []
  const staleVectors = breakdown?.stale_vectors || []

  const meanDrift = summary.mean_cosine_drift
  const maxDriftDim = summary.max_drift_dimension
  const pctDrifted = summary.pct_vectors_drifted
  const refCorpusSize = summary.reference_corpus_size

  const kpiItems = [
    { label: 'Mean Cosine Drift', value: meanDrift, icon: '&#9878;', suffix: '' },
    { label: 'Max Drift Dimension', value: maxDriftDim, icon: '&#9878;', suffix: '', isLabel: true },
    { label: '% Vectors Drifted', value: pctDrifted, icon: '&#9878;', suffix: '%', pctMode: true },
    { label: 'Reference Corpus Size', value: refCorpusSize, icon: '&#9878;', suffix: '', isCount: true },
  ]

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
    border: '1px solid #e5e7eb',
  }

  const kpiStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    minWidth: 150,
    flex: 1,
  })

  const defsList = Array.isArray(defs) ? defs : (defs ? Object.values(defs).filter(d => typeof d === 'object' && d.name) : [])

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Embedding Drift Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Vector embedding drift monitoring
          {metadata.model_name ? ` — ${metadata.model_name}` : ''}
          {metadata.dimensions ? ` | ${metadata.dimensions} dimensions` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => {
          const color = kpi.isLabel
            ? '#1e88e5'
            : kpi.isCount
              ? '#7c4dff'
              : kpi.pctMode
                ? driftColor((kpi.value != null ? kpi.value : 0) / 100)
                : driftColor(kpi.value)
          return (
            <div key={kpi.label} style={kpiStyle(color)}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
              <div style={{ fontSize: 26, fontWeight: 700, color }}>
                {kpi.isLabel
                  ? (kpi.value != null ? String(kpi.value) : '--')
                  : kpi.isCount
                    ? (kpi.value != null ? kpi.value.toLocaleString() : '--')
                    : kpi.pctMode
                      ? (kpi.value != null ? kpi.value.toFixed(1) + '%' : '--')
                      : fmt(kpi.value)}
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                {kpi.isLabel ? 'highest drift dim' : kpi.isCount ? 'vectors in reference' : kpi.pctMode ? 'above threshold' : 'cosine distance'}
              </div>
            </div>
          )
        })}
      </div>

      {/* Drift Over Time */}
      {driftTimeline.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Drift Over Time
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={driftTimeline} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="week" tick={{ fontSize: 12, fill: '#475569' }} />
              <YAxis yAxisId="left" tick={{ fontSize: 12, fill: '#475569' }} label={{ value: 'Cosine Drift', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#475569' } }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12, fill: '#475569' }} label={{ value: '% Drifted', angle: 90, position: 'insideRight', style: { fontSize: 12, fill: '#475569' } }} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val, name) => [val != null ? val.toFixed(4) : '--', name]}
              />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="cosine_drift" stroke={COLORS[0]} strokeWidth={2} dot={{ r: 3 }} name="Cosine Drift" />
              <Line yAxisId="right" type="monotone" dataKey="pct_drifted" stroke={COLORS[4]} strokeWidth={2} dot={{ r: 3 }} name="% Drifted" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Two-column: Dimension Drift + Corpus Segments */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Dimension Drift */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Dimension Drift (Top Drifting)
          </h3>
          {dimensionDrift.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dimensionDrift} layout="vertical" margin={{ top: 10, right: 20, left: 60, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} />
                <YAxis type="category" dataKey="dimension" tick={{ fontSize: 11, fill: '#475569' }} width={50} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                  formatter={(val) => val != null ? val.toFixed(4) : '--'}
                />
                <Bar dataKey="drift_score" fill={COLORS[1]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No dimension drift data available</div>
          )}
        </div>

        {/* Corpus Segments */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Corpus Segments
          </h3>
          {segments.length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Segment</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Vectors</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Avg Drift</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Max Drift</th>
                  <th style={{ textAlign: 'center', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {segments.map((seg, i) => {
                  const color = driftColor(seg.avg_drift)
                  const status = statusLabel(seg.avg_drift)
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>{seg.segment}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#334155' }}>
                        {seg.vector_count != null ? seg.vector_count.toLocaleString() : '--'}
                      </td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#334155' }}>
                        {fmt(seg.avg_drift)}
                      </td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#334155' }}>
                        {fmt(seg.max_drift)}
                      </td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                          fontSize: 11, fontWeight: 600, color: '#fff', background: color,
                        }}>
                          {status}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No segment data available</div>
          )}
        </div>
      </div>

      {/* Drift Distribution */}
      {distribution.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Drift Score Distribution
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={distribution} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="bin" tick={{ fontSize: 12, fill: '#475569' }} />
              <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val) => val != null ? val.toLocaleString() : '--'}
              />
              <Bar dataKey="count" fill={COLORS[5]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Stale Vectors */}
      {staleVectors.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Stale Vectors Needing Refresh
          </h3>
          <div style={{ maxHeight: 320, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Doc ID</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Last Updated</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Drift Score</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {staleVectors.map((vec, i) => {
                  const color = driftColor(vec.drift_score)
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500, fontFamily: 'monospace' }}>
                        {vec.doc_id || '--'}
                      </td>
                      <td style={{ padding: '8px 10px', color: '#475569' }}>
                        {vec.last_updated || '--'}
                      </td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color, fontWeight: 600 }}>
                        {fmt(vec.drift_score)}
                      </td>
                      <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>
                        {vec.recommendation || '--'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} Embedding Drift Definitions & Methodology
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Metric Name</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Threshold</th>
                </tr>
              </thead>
              <tbody>
                {defsList.map((def, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {def.name || def.metric_name || `Metric ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '--'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.threshold || def.clinical_relevance || '--'}
                    </td>
                  </tr>
                ))}
                {defsList.length === 0 && (
                  <tr>
                    <td colSpan={3} style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No definitions available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
