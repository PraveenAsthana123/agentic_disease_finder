import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#8bc34a']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function ChunkingDashboard() {
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
          axios.get(`${API_URL}/chunking/overview`),
          axios.get(`${API_URL}/chunking/breakdown`),
          axios.get(`${API_URL}/chunking/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load chunking data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading chunking data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Chunking data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const prodCfg = overview.production_config || {}
  const docDist = overview.doc_type_distribution || []
  const sizeDist = overview.size_distribution || []
  const strategies = breakdown?.strategies || []
  const patientChunks = breakdown?.patient_chunks || []
  const typeStats = breakdown?.type_stats || []
  const defsList = defs?.metrics || []

  const kpiItems = [
    { label: 'Total Chunks', value: s.total_chunks, color: COLORS[0] },
    { label: 'Avg Length', value: s.avg_chunk_length, color: COLORS[1], unit: ' chars', decimals: 0 },
    { label: 'Overlap Ratio', value: s.overlap_ratio_pct, color: COLORS[2], unit: '%', decimals: 1 },
    { label: 'Patients Covered', value: s.patients_covered, color: COLORS[3] },
    { label: 'Doc Types', value: s.document_types, color: COLORS[4] },
    { label: 'Storage', value: s.storage, color: COLORS[5], raw: true },
  ]

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
    border: '1px solid #e5e7eb',
  }

  const kpiCardStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    flex: 1,
    minWidth: 130,
    padding: 16,
  })

  const sectionHeading = { fontSize: 16, fontWeight: 700, margin: '20px 0 12px', color: '#334155' }

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Chunking Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          {s.total_chunks} chunks across {s.patients_covered} patients
          {' \u2014 '}{prodCfg.strategy} (size={prodCfg.chunk_size}, overlap={prodCfg.chunk_overlap})
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiCardStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: kpi.color }}>
              {kpi.raw ? kpi.value : fmt(kpi.value, kpi.decimals || 0)}{!kpi.raw && kpi.unit ? kpi.unit : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Chunk Size Distribution + Doc Type Pie side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Size Distribution */}
        {sizeDist.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Chunk Size Distribution</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={sizeDist} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="range" tick={{ fontSize: 12, fill: '#475569' }} />
                <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                  formatter={(val) => [val, 'Chunks']} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {sizeDist.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Doc Type Pie */}
        {docDist.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Chunks by Document Type</h3>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={docDist}
                  dataKey="chunks"
                  nameKey="type"
                  cx="50%"
                  cy="50%"
                  outerRadius={85}
                  label={({ type, chunks }) => `${type}: ${chunks}`}
                >
                  {docDist.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Production Config Card */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={sectionHeading}>Production Chunking Config</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          {[
            { label: 'Strategy', val: prodCfg.strategy },
            { label: 'Chunk Size', val: `${prodCfg.chunk_size} chars` },
            { label: 'Overlap', val: `${prodCfg.chunk_overlap} chars` },
            { label: 'Median Length', val: `${fmt(s.median_chunk_length)} chars` },
          ].map(c => (
            <div key={c.label} style={{ padding: 12, background: '#f1f5f9', borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{c.label}</div>
              <div style={{ fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{c.val}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Available Strategies Table */}
      {strategies.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Available Chunking Strategies</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Strategy</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Parameters</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Use Case</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((st, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>{st.name}</td>
                  <td style={{ padding: '8px 10px', color: '#475569' }}>{st.description}</td>
                  <td style={{ padding: '8px 10px', color: '#1e88e5', fontFamily: 'monospace', fontSize: 12 }}>
                    {Object.entries(st.params || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')}
                  </td>
                  <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{st.use_case}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Per-Type Stats + Per-Patient Chunks side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Per-Type Stats */}
        {typeStats.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Chunk Stats by Document Type</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569', fontWeight: 600 }}>Type</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569', fontWeight: 600 }}>Count</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569', fontWeight: 600 }}>Avg Len</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569', fontWeight: 600 }}>Min</th>
                  <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569', fontWeight: 600 }}>Max</th>
                </tr>
              </thead>
              <tbody>
                {typeStats.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', color: '#1e293b', fontWeight: 500 }}>{t.type}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', color: '#1e88e5', fontWeight: 600 }}>{t.count}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', color: '#475569' }}>{fmt(t.avg_length, 0)}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', color: '#94a3b8' }}>{t.min_length}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'right', color: '#94a3b8' }}>{t.max_length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Per-Patient Chunks Bar */}
        {patientChunks.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Chunks per Patient</h3>
            <ResponsiveContainer width="100%" height={Math.max(200, patientChunks.length * 28)}>
              <BarChart
                data={patientChunks.slice(0, 15)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} />
                <YAxis type="category" dataKey="patient_id" tick={{ fontSize: 11, fill: '#475569' }} width={75} />
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                  formatter={(val) => [val, 'Chunks']} />
                <Bar dataKey="chunks" radius={[0, 4, 4, 0]}>
                  {patientChunks.slice(0, 15).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} Metric Definitions
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Unit</th>
                </tr>
              </thead>
              <tbody>
                {defsList.map((def, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {def.name || `Metric ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '--'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.unit || '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {defs.clinical_relevance && (
              <p style={{ marginTop: 16, padding: '12px 14px', background: '#f0fdf4', borderRadius: 8, fontSize: 13, color: '#166534', lineHeight: 1.5 }}>
                <strong>Clinical Relevance:</strong> {defs.clinical_relevance}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
