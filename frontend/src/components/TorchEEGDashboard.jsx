import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, Legend, Cell
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']
const BAND_COLORS = { delta: '#1e88e5', theta: '#7c4dff', alpha: '#4caf50', beta: '#ff9800' }

export default function TorchEEGDashboard() {
  const [overview, setOverview] = useState(null)
  const [features, setFeatures] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [activeTransform, setActiveTransform] = useState('differential_entropy')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, ft, df] = await Promise.all([
          axios.get(`${API_URL}/api/torcheeg/overview`),
          axios.get(`${API_URL}/api/torcheeg/features`),
          axios.get(`${API_URL}/api/torcheeg/definitions`)
        ])
        setOverview(ov.data)
        setFeatures(ft.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load TorchEEG data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
      Running TorchEEG transforms on real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'TorchEEG not available. Ensure torcheeg and EDF data are present.'}
    </div>
  )

  const { classification, transform_summary, model_info, train_losses, channels, bands } = overview

  const kpiItems = [
    { label: 'Accuracy', value: classification.accuracy, color: '#1e88e5' },
    { label: 'Precision', value: classification.precision, color: '#7c4dff' },
    { label: 'Recall', value: classification.recall, color: '#4caf50' },
    { label: 'F1 Score', value: classification.f1, color: '#ff9800' },
  ]

  const kpiColor = (v) => {
    if (v >= 0.8) return '#16a34a'
    if (v >= 0.6) return '#ca8a04'
    return '#dc2626'
  }

  const radarData = Object.entries(transform_summary).map(([name, info]) => {
    const avgMag = Object.values(info.mean_per_band).reduce((a, b) => a + Math.abs(b), 0) / 4
    return { transform: name.replace(/_/g, ' '), magnitude: Math.min(avgMag, 20) }
  })

  const heatmapData = features?.heatmaps?.[activeTransform] || []

  const bandCompData = features?.band_comparison || []

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    border: '1px solid #e5e7eb',
  }

  const kpiStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    minWidth: 150,
    flex: 1,
  })

  const tabStyle = (active) => ({
    padding: '6px 14px',
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: active ? 600 : 400,
    background: active ? '#1e88e5' : '#f1f5f9',
    color: active ? '#fff' : '#475569',
    border: 'none',
  })

  const heatVal = (v) => {
    const abs = Math.abs(v)
    if (abs > 10) return { bg: '#1e88e5', color: '#fff' }
    if (abs > 5) return { bg: '#64b5f6', color: '#fff' }
    if (abs > 1) return { bg: '#bbdefb', color: '#1e293b' }
    return { bg: '#e3f2fd', color: '#1e293b' }
  }

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          🧠 TorchEEG Feature Analysis
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Real <b>torcheeg</b> v{overview.version} transforms on {overview.data_info?.file || 'EDF'}
          {' '}&middot; {channels?.length || 0} channels
          {' '}&middot; {overview.data_info?.n_epochs || 0} epochs
          {' '}&middot; {overview.transforms_applied?.length || 0} transforms
        </p>
      </div>

      {/* Classification KPIs */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: kpiColor(kpi.value) }}>
              {kpi.value != null ? kpi.value.toFixed(3) : '\u2014'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              {kpi.value >= 0.8 ? 'Good' : kpi.value >= 0.6 ? 'Fair' : 'Low'}
            </div>
          </div>
        ))}
        <div style={{ ...kpiStyle('#00bcd4') }}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Model</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>
            {model_info?.name || 'EEGNet-Mini'}
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>
            {model_info?.input_features || 0} features
          </div>
        </div>
      </div>

      {/* Feature Heatmap + Band Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <h3 style={{ margin: 0, fontSize: 15, color: '#334155' }}>Channel x Band Heatmap</h3>
            <div style={{ display: 'flex', gap: 6 }}>
              {Object.keys(transform_summary).map(t => (
                <button key={t} onClick={() => setActiveTransform(t)} style={tabStyle(activeTransform === t)}>
                  {t.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b', borderBottom: '1px solid #e5e7eb' }}>Channel</th>
                  {bands?.map(b => (
                    <th key={b} style={{ textAlign: 'center', padding: '6px 10px', color: BAND_COLORS[b] || '#64748b', borderBottom: '1px solid #e5e7eb' }}>
                      {b}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heatmapData.map((row, i) => (
                  <tr key={i}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b', borderBottom: '1px solid #f1f5f9' }}>
                      {row.channel}
                    </td>
                    {bands?.map(b => {
                      const v = row[b]
                      const style = heatVal(v)
                      return (
                        <td key={b} style={{
                          textAlign: 'center', padding: '6px 10px',
                          background: style.bg, color: style.color,
                          fontWeight: 500, borderRadius: 4, borderBottom: '1px solid #f1f5f9'
                        }}>
                          {v != null ? v.toFixed(3) : '\u2014'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Band Comparison</h3>
          {bandCompData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bandCompData} margin={{ left: 0, right: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="band" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="differential_entropy" name="DE" fill="#1e88e5" radius={[3, 3, 0, 0]} />
                <Bar dataKey="power_spectral_density" name="PSD" fill="#7c4dff" radius={[3, 3, 0, 0]} />
                <Bar dataKey="hjorth" name="Hjorth" fill="#4caf50" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No data</div>
          )}
        </div>
      </div>

      {/* Training Loss Curve + Transform Radar */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Training Loss Curve</h3>
          {train_losses?.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={train_losses} margin={{ left: 10, right: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }}
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }}
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Line type="monotone" dataKey="loss" stroke="#f44336" dot={{ r: 3 }} strokeWidth={2} name="CE Loss" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No training data</div>
          )}
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Transform Feature Magnitudes</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="transform" tick={{ fontSize: 10, fill: '#475569' }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} />
              <Radar name="Avg |magnitude|" dataKey="magnitude" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.25} strokeWidth={2} />
              <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(3) : v} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Transform Summary Cards */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>Transform Summary</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
          {Object.entries(transform_summary).map(([name, info]) => (
            <div key={name} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 8, fontSize: 13, textTransform: 'capitalize' }}>
                {name.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                Shape: {info.shape?.join(' x ')}
              </div>
              {bands?.map(b => (
                <div key={b} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '2px 0' }}>
                  <span style={{ color: BAND_COLORS[b] || '#64748b', fontWeight: 500 }}>{b}</span>
                  <span style={{ color: '#1e293b' }}>
                    {info.mean_per_band?.[b]?.toFixed(4)} &plusmn; {info.std_per_band?.[b]?.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} Transform Definitions & Clinical Relevance
        </button>
        {showDefs && defs?.transforms && (
          <div style={{ marginTop: 16, display: 'grid', gap: 14 }}>
            {defs.transforms.map((def, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{def.name}</div>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>{def.class}</div>
                <div style={{ color: '#475569', fontSize: 13, lineHeight: 1.5 }}>{def.description}</div>
                <div style={{ marginTop: 4, fontSize: 12, color: '#1e88e5' }}>Bands: {def.bands}</div>
              </div>
            ))}
            {defs.clinical_relevance && (
              <div style={{ background: '#eff6ff', borderRadius: 8, padding: 14, border: '1px solid #bfdbfe' }}>
                <div style={{ fontWeight: 600, color: '#1e40af', marginBottom: 6 }}>Clinical Relevance</div>
                <div style={{ color: '#1e3a5f', fontSize: 13, lineHeight: 1.6 }}>{defs.clinical_relevance}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
