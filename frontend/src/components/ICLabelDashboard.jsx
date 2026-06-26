import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = {
  brain: '#22c55e',
  muscle: '#ef4444',
  eye: '#f59e0b',
  heart: '#ec4899',
  line_noise: '#6366f1',
  channel_noise: '#64748b',
  other: '#94a3b8',
}
const PIE_COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#ec4899', '#6366f1', '#64748b', '#94a3b8']

export default function ICLabelDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(0)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/icalabel`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load ICLabel report')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🧠</div>
      Running ICLabel ICA component classification on real EEG data...
      <div style={{ fontSize: 12, marginTop: 8, color: '#94a3b8' }}>
        (ICA fitting + neural-net classification — may take ~45 seconds on first load)
      </div>
    </div>
  )
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  )
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>
      ICLabel analysis unavailable: {data?.error || 'No EEG data found'}
    </div>
  )

  const agg = data.aggregate_class_distribution || {}
  const files = data.per_file || []
  const current = files[selectedFile] || {}

  // Aggregate pie chart
  const pieData = Object.entries(agg)
    .map(([name, value]) => ({ name, value }))
    .filter(d => d.value > 0)

  // Per-file stacked bar
  const barData = files.map(f => ({
    name: f.subject || f.file,
    Brain: f.class_distribution?.brain || 0,
    Eye: f.class_distribution?.eye || 0,
    Muscle: f.class_distribution?.muscle || 0,
    Heart: f.class_distribution?.heart || 0,
    'Line Noise': f.class_distribution?.line_noise || 0,
    'Ch Noise': f.class_distribution?.channel_noise || 0,
    Other: f.class_distribution?.other || 0,
  }))

  // Radar: data quality dimensions
  const radarData = [
    { axis: 'Brain %', val: Math.round(data.brain_ratio * 100) },
    { axis: 'Files OK', val: data.files_analyzed > 0 ? 100 : 0 },
    { axis: 'Components', val: Math.min(100, Math.round(data.total_components / 45 * 100)) },
    { axis: 'Low Artifact', val: Math.round((1 - data.total_artifact / Math.max(data.total_components, 1)) * 100) },
    { axis: 'Confidence', val: current.components
      ? Math.round(current.components.reduce((s, c) => s + c.confidence, 0) / current.components.length * 100)
      : 0 },
  ]

  const card = { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }
  const kpiCard = { ...card, textAlign: 'center', flex: 1, minWidth: 130 }

  return (
    <div style={{ padding: 16, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: '#1e293b' }}>
        🧠 ICLabel ICA Component Classification
      </h2>
      <p style={{ margin: '0 0 16px', color: '#64748b', fontSize: 13 }}>
        Automatic classification of ICA components using <b>mne-icalabel</b> neural-net
        on real CHB-MIT EEG recordings. Classes: brain, muscle, eye, heart, line noise, channel noise, other.
      </p>

      {/* KPI tiles */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={kpiCard}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#22c55e' }}>{data.total_brain}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Brain Components</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{data.total_artifact}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Artifact Components</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#6366f1' }}>{data.total_components}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Total ICA Components</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#0ea5e9' }}>
            {Math.round(data.brain_ratio * 100)}%
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Brain Ratio</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#8b5cf6' }}>{data.files_analyzed}</div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Files Analyzed</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Pie: aggregate class distribution */}
        <div style={{ ...card, flex: 1, minWidth: 300 }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            Component Class Distribution (All Files)
          </h3>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={COLORS[pieData[i].name] || PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Radar: quality dimensions */}
        <div style={{ ...card, flex: 1, minWidth: 300 }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            ICLabel Quality Radar
          </h3>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar name="Score" dataKey="val" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Stacked bar: per-file breakdown */}
      {barData.length > 0 && (
        <div style={{ ...card }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            Per-Subject Component Breakdown
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Brain" stackId="a" fill={COLORS.brain} />
              <Bar dataKey="Eye" stackId="a" fill={COLORS.eye} />
              <Bar dataKey="Muscle" stackId="a" fill={COLORS.muscle} />
              <Bar dataKey="Heart" stackId="a" fill={COLORS.heart} />
              <Bar dataKey="Line Noise" stackId="a" fill={COLORS.line_noise} />
              <Bar dataKey="Ch Noise" stackId="a" fill={COLORS.channel_noise} />
              <Bar dataKey="Other" stackId="a" fill={COLORS.other} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-file detail selector */}
      {files.length > 1 && (
        <div style={{ marginBottom: 12 }}>
          <span style={{ fontSize: 13, color: '#64748b', marginRight: 8 }}>Select file:</span>
          {files.map((f, i) => (
            <button key={i} onClick={() => setSelectedFile(i)}
              style={{
                padding: '4px 12px', marginRight: 6, fontSize: 12,
                border: i === selectedFile ? '2px solid #6366f1' : '1px solid #e5e7eb',
                borderRadius: 6, background: i === selectedFile ? '#eef2ff' : '#fff',
                cursor: 'pointer', color: '#334155',
              }}>
              {f.subject}/{f.file}
            </button>
          ))}
        </div>
      )}

      {/* Component table for selected file */}
      {current.components && (
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 15, color: '#334155' }}>
            ICA Components — {current.subject}/{current.file}
          </h3>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
            {current.n_channels} channels | {current.n_components} components |
            sfreq {current.sfreq} Hz | {current.duration_sec}s duration
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>IC#</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Class</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Confidence</th>
                  <th style={{ textAlign: 'left', padding: '6px 8px' }}>Bar</th>
                </tr>
              </thead>
              <tbody>
                {current.components.map(c => (
                  <tr key={c.index} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace' }}>IC{c.index}</td>
                    <td style={{ padding: '5px 8px' }}>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 4,
                        background: COLORS[c.label] + '20',
                        color: COLORS[c.label] || '#64748b',
                        fontWeight: 600, fontSize: 12,
                      }}>
                        {c.label}
                      </span>
                    </td>
                    <td style={{ padding: '5px 8px', fontFamily: 'monospace' }}>
                      {(c.confidence * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '5px 8px', width: '40%' }}>
                      <div style={{ background: '#f1f5f9', borderRadius: 4, height: 14, overflow: 'hidden' }}>
                        <div style={{
                          width: `${c.confidence * 100}%`,
                          background: COLORS[c.label] || '#94a3b8',
                          height: '100%', borderRadius: 4,
                        }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Errors */}
      {data.errors?.length > 0 && (
        <div style={{ ...card, background: '#fef2f2', borderColor: '#fecaca' }}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#991b1b' }}>Processing Errors</h3>
          {data.errors.map((e, i) => (
            <div key={i} style={{ fontSize: 12, color: '#7f1d1d', marginBottom: 4 }}>
              <b>{e.file}</b>: {e.error}
            </div>
          ))}
        </div>
      )}

      {/* Footer */}
      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        {data.tool} | {data.elapsed_sec}s | {data.generated_at}
      </div>
    </div>
  )
}
