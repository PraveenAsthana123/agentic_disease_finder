import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  AreaChart, Area
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

function fmt(v) { return v == null ? '—' : typeof v === 'number' ? v.toLocaleString() : String(v) }
function fmtPct(v) { return v == null ? '—' : `${v}%` }

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
  const map = {
    validated: { bg: '#dcfce7', fg: '#16a34a' },
    complete: { bg: '#dcfce7', fg: '#16a34a' },
    beta: { bg: '#dbeafe', fg: '#2563eb' },
    experimental: { bg: '#fef3c7', fg: '#d97706' },
    planned: { bg: '#f1f5f9', fg: '#64748b' },
    pending: { bg: '#f1f5f9', fg: '#64748b' },
    partial: { bg: '#fef3c7', fg: '#d97706' },
    ready: { bg: '#dbeafe', fg: '#2563eb' },
    baseline: { bg: '#f3e8ff', fg: '#7c3aed' },
  }
  const s = map[status] || { bg: '#f1f5f9', fg: '#64748b' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: s.bg, color: s.fg
    }}>{status}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'models', label: 'Model Details' },
  { id: 'devices', label: 'Device Targets' },
  { id: 'quantization', label: 'Quantization' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EdgeDeployDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/edge-deploy/overview`),
      axios.get(`${API_URL}/edge-deploy/breakdown`),
      axios.get(`${API_URL}/edge-deploy/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading edge deployment data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return null

  const renderOverview = () => {
    const pipeline = overview.export_pipeline || []
    const pipelineChart = pipeline.map(s => ({
      name: s.step,
      value: s.status === 'complete' ? 100 : s.status === 'partial' ? 50 : s.status === 'ready' ? 25 : 0
    }))

    const deviceStatusData = (overview.target_devices || []).map(d => ({
      name: d.name,
      latency: d.latency_ms || 0,
      ram: d.ram_mb,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <Card><KPI label="sklearn Models" value={fmt(overview.total_sklearn_models)} color="#3b82f6" /></Card>
        <Card><KPI label="ONNX Models" value={fmt(overview.total_onnx_models)} color="#10b981" /></Card>
        <Card><KPI label="ONNX Coverage" value={fmtPct(overview.onnx_coverage_pct)} color="#8b5cf6" /></Card>
        <Card><KPI label="Target Devices" value={fmt((overview.target_devices || []).length)} color="#f59e0b" /></Card>

        <Card title="Export Pipeline Status" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={pipelineChart} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => `${v}%`} />
              <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Device Inference Latency (ms)" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={deviceStatusData.filter(d => d.latency > 0)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="latency" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Export Pipeline Steps" span={4}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {pipeline.map((s, i) => (
              <div key={i} style={{
                flex: '1 1 160px', padding: 12, borderRadius: 8,
                background: s.status === 'complete' ? '#dcfce7' : s.status === 'partial' ? '#fef3c7' : '#f1f5f9',
                border: `1px solid ${s.status === 'complete' ? '#86efac' : s.status === 'partial' ? '#fde68a' : '#e2e8f0'}`
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{s.step}</div>
                <StatusBadge status={s.status} />
                <div style={{ fontSize: 10, color: '#64748b', marginTop: 4 }}>{s.output}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const renderModels = () => {
    const models = breakdown?.models || []
    const sizeChart = models.filter(m => m.sklearn_size_kb > 0).map(m => ({
      name: m.name.length > 20 ? m.name.slice(0, 20) + '...' : m.name,
      sklearn: m.sklearn_size_kb,
      onnx: m.onnx_size_kb || 0,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Model Size Comparison (KB)" span={2}>
          {sizeChart.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={sizeChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="sklearn" name="sklearn (KB)" fill="#3b82f6" />
                <Bar dataKey="onnx" name="ONNX (KB)" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 20, color: '#64748b', textAlign: 'center' }}>No model size data available</div>
          )}
        </Card>

        <Card title="Total Size by Format" span={1}>
          {(breakdown?.size_by_format || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
              <span style={{ fontSize: 13, color: '#334155' }}>{s.format}</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: COLORS[i] }}>{fmt(s.total_kb)} KB</span>
            </div>
          ))}
        </Card>

        <Card title="ONNX Export Status" span={1}>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={[
                  { name: 'Exported', value: models.filter(m => m.onnx_exported).length },
                  { name: 'Pending', value: models.filter(m => !m.onnx_exported).length },
                ]}
                cx="50%" cy="50%" innerRadius={40} outerRadius={70} dataKey="value"
              >
                <Cell fill="#10b981" />
                <Cell fill="#e2e8f0" />
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Model Details" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Model</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>sklearn (KB)</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>ONNX</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>ONNX (KB)</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Reduction</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Devices</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontWeight: 500 }}>{m.name}</td>
                    <td style={{ padding: 8, textAlign: 'right' }}>{fmt(m.sklearn_size_kb)}</td>
                    <td style={{ padding: 8, textAlign: 'center' }}>
                      {m.onnx_exported
                        ? <span style={{ color: '#16a34a', fontWeight: 700 }}>Yes</span>
                        : <span style={{ color: '#94a3b8' }}>No</span>}
                    </td>
                    <td style={{ padding: 8, textAlign: 'right' }}>{m.onnx_size_kb ? fmt(m.onnx_size_kb) : '—'}</td>
                    <td style={{ padding: 8, textAlign: 'right', color: m.size_reduction_pct > 0 ? '#16a34a' : '#64748b' }}>
                      {m.size_reduction_pct != null ? `${m.size_reduction_pct}%` : '—'}
                    </td>
                    <td style={{ padding: 8, fontSize: 11 }}>{(m.target_devices || []).join(', ') || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  const renderDevices = () => {
    const devices = overview?.target_devices || []
    const radarData = devices.filter(d => d.latency_ms).map(d => ({
      device: d.name,
      latency: Math.max(0, 100 - d.latency_ms),
      ram: Math.min(100, d.ram_mb / 164),
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {devices.map((d, i) => (
          <Card key={i} title={d.name}>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Architecture</span>
                <span style={{ fontSize: 12, fontWeight: 500 }}>{d.arch}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>RAM</span>
                <span style={{ fontSize: 12, fontWeight: 500 }}>{d.ram_mb >= 1024 ? `${(d.ram_mb / 1024).toFixed(0)} GB` : `${d.ram_mb} MB`}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Runtime</span>
                <span style={{ fontSize: 12, fontWeight: 500 }}>{d.runtime}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Latency</span>
                <span style={{ fontSize: 12, fontWeight: 500 }}>{d.latency_ms ? `${d.latency_ms} ms` : '—'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: '#64748b' }}>Status</span>
                <StatusBadge status={d.status} />
              </div>
            </div>
          </Card>
        ))}

        <Card title="Compatibility Matrix" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Model</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>ONNX</th>
                  {devices.map((d, i) => (
                    <th key={i} style={{ textAlign: 'center', padding: 8, fontSize: 10 }}>{d.name}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown?.compatibility_matrix || []).map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontWeight: 500 }}>{row.model}</td>
                    <td style={{ padding: 8, textAlign: 'center' }}>
                      {row.onnx ? <span style={{ color: '#16a34a' }}>Yes</span> : <span style={{ color: '#94a3b8' }}>No</span>}
                    </td>
                    {devices.map((d, j) => (
                      <td key={j} style={{ padding: 8, textAlign: 'center' }}>
                        {row[d.name]
                          ? <span style={{ color: '#16a34a', fontWeight: 700 }}>OK</span>
                          : <span style={{ color: '#e2e8f0' }}>—</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  const renderQuantization = () => {
    const modes = overview?.quantization_modes || []
    const chartData = modes.map(m => ({
      name: m.mode.split(' ')[0],
      latency_factor: m.latency_factor * 100,
      accuracy_delta: parseFloat(m.accuracy_delta) || 0,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Quantization Modes" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Mode</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Size Reduction</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Accuracy Delta</th>
                  <th style={{ textAlign: 'right', padding: 8 }}>Latency Factor</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {modes.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8, fontWeight: 500 }}>{m.mode}</td>
                    <td style={{ padding: 8, textAlign: 'right', color: '#16a34a', fontWeight: 600 }}>{m.size_reduction}</td>
                    <td style={{ padding: 8, textAlign: 'right', color: parseFloat(m.accuracy_delta) < -1 ? '#ef4444' : '#f59e0b' }}>{m.accuracy_delta}</td>
                    <td style={{ padding: 8, textAlign: 'right' }}>{m.latency_factor}x</td>
                    <td style={{ padding: 8, textAlign: 'center' }}><StatusBadge status={m.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Relative Latency by Quantization">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => `${v.toFixed(0)}%`} />
              <Bar dataKey="latency_factor" name="Latency %" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Accuracy Impact by Quantization">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => `${v.toFixed(1)}%`} />
              <Bar dataKey="accuracy_delta" name="Accuracy Delta %" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    const terms = definitions?.terms || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
        {terms.map((t, i) => (
          <Card key={i}>
            <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{t.term}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{t.definition}</div>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Edge Deployment Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          ONNX export, quantization, and device target management for edge inference
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'models' && renderModels()}
      {tab === 'devices' && renderDevices()}
      {tab === 'quantization' && renderQuantization()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
