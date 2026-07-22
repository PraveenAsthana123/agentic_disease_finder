import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function fmt(v, decimals = 2) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function InferenceGPUDashboard() {
  const [overview, setOverview] = useState(null)
  const [models, setModels] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, md, df] = await Promise.all([
          axios.get(`${API_URL}/api/inference-gpu/overview`),
          axios.get(`${API_URL}/api/inference-gpu/models`),
          axios.get(`${API_URL}/api/inference-gpu/definitions`)
        ])
        setOverview(ov.data)
        setModels(md.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load inference/GPU data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading inference and GPU data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'GPU data not available. Ensure nvidia-smi is accessible.'}
    </div>
  )

  const gpu = overview.gpu || {}
  const inference = overview.inference_summary || {}
  const system = overview.system || {}
  const modelList = Array.isArray(models?.models) ? models.models : []
  const defsList = Array.isArray(defs?.metrics) ? defs.metrics : []

  const kpiItems = [
    { label: 'GPU Utilization', value: gpu.utilization_gpu_pct, color: COLORS[0], unit: '%', decimals: 0 },
    { label: 'VRAM Used', value: gpu.memory_used_mb, color: COLORS[1], unit: 'MB', decimals: 0 },
    { label: 'Temperature', value: gpu.temperature_c, color: gpu.temperature_c > 80 ? COLORS[4] : gpu.temperature_c > 60 ? COLORS[3] : COLORS[2], unit: '\u00b0C', decimals: 0 },
    { label: 'Power Draw', value: gpu.power_draw_w, color: COLORS[3], unit: 'W', decimals: 1 },
  ]

  const vramData = [
    { name: 'Used', value: gpu.memory_used_mb || 0 },
    { name: 'Free', value: gpu.memory_free_mb || 0 },
  ]

  const ramUsedGb = system.ram_used_gb || 0
  const ramTotalGb = system.ram_total_gb || 1
  const ramPct = (ramUsedGb / ramTotalGb) * 100

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
    minWidth: 150,
    padding: 16,
  })

  const sectionHeading = { fontSize: 16, fontWeight: 700, margin: '20px 0 12px', color: '#334155' }

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Inference &amp; GPU Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          {gpu.name || 'GPU'} — {inference.total_inferences != null ? `${inference.total_inferences} total inferences` : 'Monitoring GPU resources'}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiCardStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: kpi.color }}>
              {fmt(kpi.value, kpi.decimals || 0)}{kpi.unit ? ` ${kpi.unit}` : ''}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{kpi.unit || 'count'}</div>
          </div>
        ))}
      </div>

      {/* GPU Memory */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={sectionHeading}>GPU Memory (VRAM)</h3>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart
            data={vramData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} tickFormatter={v => `${v} MB`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: '#475569' }} width={55} />
            <Tooltip
              contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
              formatter={(val) => [`${val} MB`, 'VRAM']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {vramData.map((entry, i) => (
                <Cell key={i} fill={i === 0 ? COLORS[1] : COLORS[2]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 6, textAlign: 'center' }}>
          {fmt(gpu.memory_used_mb, 0)} / {fmt(gpu.memory_total_mb, 0)} MB ({fmt((gpu.memory_used_mb / (gpu.memory_total_mb || 1)) * 100, 1)}% used)
        </div>
      </div>

      {/* System Resources */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* CPU */}
        <div style={cardStyle}>
          <h3 style={{ ...sectionHeading, margin: '0 0 14px' }}>CPU Cores</h3>
          <div style={{ fontSize: 36, fontWeight: 700, color: COLORS[0] }}>
            {system.cpu_count || '--'}
          </div>
          <div style={{ marginTop: 10, fontSize: 12, color: '#94a3b8' }}>
            Python {system.python_version || '--'}
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: '#94a3b8' }}>
            PyTorch {system.torch_version || '--'}
          </div>
          <div style={{ marginTop: 4, fontSize: 12, color: system.cuda_available ? '#4caf50' : '#ff9800' }}>
            CUDA: {system.cuda_available ? 'Available' : 'Not available'}
          </div>
          {system.cuda_note && (
            <div style={{ marginTop: 4, fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>
              {system.cuda_note}
            </div>
          )}
        </div>

        {/* Memory */}
        <div style={cardStyle}>
          <h3 style={{ ...sectionHeading, margin: '0 0 14px' }}>RAM Usage</h3>
          <div style={{ fontSize: 36, fontWeight: 700, color: '#1e88e5' }}>
            {fmt(ramUsedGb, 1)} <span style={{ fontSize: 18, color: '#64748b' }}>GB</span>
          </div>
          <div style={{ marginTop: 10, background: '#e2e8f0', borderRadius: 6, height: 10 }}>
            <div style={{
              width: `${Math.min(ramPct, 100)}%`, height: '100%', borderRadius: 6,
              background: ramPct > 80 ? '#f44336' : ramPct > 60 ? '#ff9800' : '#1e88e5',
              transition: 'width 0.4s ease',
            }} />
          </div>
          <div style={{ marginTop: 6, fontSize: 12, color: '#94a3b8' }}>
            {fmt(ramUsedGb, 1)} / {fmt(ramTotalGb, 1)} GB ({fmt(ramPct, 1)}% used)
          </div>
        </div>
      </div>

      {/* Inference Activity */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={sectionHeading}>Inference Activity</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginBottom: 16 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: COLORS[0] }}>{inference.total_inferences || 0}</div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>Total Inferences</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: COLORS[2] }}>{inference.models_loaded || 0}</div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>Models Loaded</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: COLORS[3] }}>{fmt(inference.avg_throughput_per_hour, 1)}</div>
            <div style={{ fontSize: 12, color: '#94a3b8' }}>Avg/Hour</div>
          </div>
        </div>
        {inference.last_inference_at && (
          <div style={{ fontSize: 12, color: '#94a3b8', textAlign: 'center' }}>
            Last inference: {inference.last_inference_at}
          </div>
        )}
      </div>

      {/* Loaded Models */}
      {modelList.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Model Files ({modelList.length})</h3>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Name</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Size (MB)</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Last Modified</th>
                </tr>
              </thead>
              <tbody>
                {modelList.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>{m.name}</td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>{fmt(m.size_mb, 2)}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 12 }}>{m.modified || '--'}</td>
                  </tr>
                ))}
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
