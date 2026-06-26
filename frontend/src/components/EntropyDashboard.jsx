import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

const METRIC_LABELS = {
  sample_entropy: 'SampEn',
  permutation_entropy: 'Perm. Entropy',
  spectral_entropy: 'Spectral Ent.',
  approximate_entropy: 'ApEn',
  higuchi_fd: 'Higuchi FD',
  dfa: 'DFA α'
}

export default function EntropyDashboard() {
  const [overview, setOverview] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeMetric, setActiveMetric] = useState('permutation_entropy')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, df] = await Promise.all([
          axios.get(`${API_URL}/entropy/overview?seconds=30`),
          axios.get(`${API_URL}/entropy/definitions`)
        ])
        setOverview(ov.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load entropy data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🧮</div>
      Computing entropy features from real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'No EDF data available for entropy analysis.'}
    </div>
  )

  const { per_channel, summary, file, sfreq, n_channels, seconds_analyzed } = overview
  const metricKeys = Object.keys(METRIC_LABELS)

  // Bar chart data: per-channel values for the selected metric
  const barData = per_channel.map(ch => ({
    channel: ch.channel.length > 8 ? ch.channel.slice(0, 8) : ch.channel,
    value: ch[activeMetric] ?? 0,
  }))

  // Radar data: summary means across all metrics (normalized 0-1 for display)
  const radarData = metricKeys.map(k => {
    const s = summary[k]
    return {
      metric: METRIC_LABELS[k],
      value: s ? s.mean : 0,
      fullMark: k === 'higuchi_fd' ? 2.0 : k === 'dfa' ? 1.5 : 2.5,
    }
  })

  // Channel comparison: line chart showing all channels for selected metric
  const channelLine = per_channel.map((ch, i) => ({
    idx: i + 1,
    channel: ch.channel,
    value: ch[activeMetric] ?? 0,
  }))

  // Summary cards data
  const summaryCards = metricKeys.map(k => ({
    key: k,
    label: METRIC_LABELS[k],
    ...summary[k],
  })).filter(c => c.mean !== undefined)

  const cardStyle = {
    padding: '12px 16px', borderRadius: 8, border: '1px solid #e5e7eb',
    background: '#f8fafc', minWidth: 140, flex: '1 1 140px',
  }

  return (
    <div style={{ padding: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16,
        padding: '12px 16px', background: 'linear-gradient(90deg,#ede9fe,#dbeafe)',
        borderRadius: 8, border: '1px solid #c7d2fe' }}>
        <span style={{ fontSize: 28 }}>🧮</span>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#1e1b4b' }}>
            Entropy & Complexity Dashboard
          </div>
          <div style={{ fontSize: 12, color: '#4338ca' }}>
            AntroPy (Vallat 2023) — {n_channels} channels · {seconds_analyzed}s · {file} · {sfreq} Hz
          </div>
        </div>
      </div>

      {/* Summary stat cards */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 20 }}>
        {summaryCards.map((c, i) => (
          <div key={c.key} style={{
            ...cardStyle,
            borderLeft: `3px solid ${COLORS[i % COLORS.length]}`,
            cursor: 'pointer',
            outline: activeMetric === c.key ? '2px solid #6366f1' : 'none',
          }} onClick={() => setActiveMetric(c.key)}>
            <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>{c.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{c.mean}</div>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>
              σ {c.std} · range [{c.min}, {c.max}]
            </div>
          </div>
        ))}
      </div>

      {/* Charts grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        {/* Bar chart — per-channel */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>
            Per-Channel: {METRIC_LABELS[activeMetric]}
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="channel" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar chart — metric profile */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>
            Entropy Profile (Mean Across Channels)
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis tick={{ fontSize: 9 }} />
              <Radar name="Mean" dataKey="value" stroke="#7c3aed" fill="#7c3aed" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Channel comparison line */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 8 }}>
            Channel Variation: {METRIC_LABELS[activeMetric]}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={channelLine} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="idx" tick={{ fontSize: 10 }} label={{ value: 'Channel #', position: 'bottom', fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip content={({ payload }) => payload?.[0] ? (
                <div style={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: 6, padding: 8, fontSize: 11 }}>
                  <strong>{payload[0].payload.channel}</strong><br/>
                  {METRIC_LABELS[activeMetric]}: {payload[0].value}
                </div>
              ) : null} />
              <Line type="monotone" dataKey="value" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 3, fill: '#0ea5e9' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Interpretation guide */}
        <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>
            Clinical Interpretation
          </div>
          {defs?.metrics?.filter(m => m.name.includes(METRIC_LABELS[activeMetric].split('.')[0].trim().slice(0,4)))
            .slice(0, 1).map((m, i) => (
            <div key={i} style={{ fontSize: 12, lineHeight: 1.6 }}>
              <div style={{ fontWeight: 600, color: '#4338ca', marginBottom: 4 }}>{m.name}</div>
              <div style={{ color: '#334155', marginBottom: 6 }}>{m.description}</div>
              <div style={{ color: '#059669', fontSize: 11 }}>Range: {m.range}</div>
              <div style={{ color: '#b45309', fontSize: 11, marginTop: 4 }}>Clinical: {m.clinical}</div>
              <div style={{ color: '#64748b', fontSize: 10, marginTop: 4, fontStyle: 'italic' }}>
                Ref: {m.reference}
              </div>
            </div>
          ))}
          {defs?.clinical_summary && (
            <div style={{ marginTop: 12, padding: 10, background: '#f0fdf4', borderRadius: 6,
              fontSize: 11, color: '#166534', lineHeight: 1.5 }}>
              {defs.clinical_summary}
            </div>
          )}
        </div>
      </div>

      {/* Per-channel table */}
      <div style={{ background: '#ffffff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 10 }}>
          Per-Channel Entropy Table
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '6px 10px', color: '#475569' }}>Channel</th>
                {metricKeys.map(k => (
                  <th key={k} style={{ textAlign: 'right', padding: '6px 10px', color: '#475569',
                    fontWeight: activeMetric === k ? 700 : 500,
                    background: activeMetric === k ? '#eef2ff' : 'transparent' }}>
                    {METRIC_LABELS[k]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {per_channel.map((ch, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9',
                  background: i % 2 === 0 ? '#fafafa' : '#ffffff' }}>
                  <td style={{ padding: '5px 10px', fontWeight: 600, color: '#1e293b' }}>{ch.channel}</td>
                  {metricKeys.map(k => (
                    <td key={k} style={{ textAlign: 'right', padding: '5px 10px',
                      color: ch[k] == null ? '#cbd5e1' : '#0f172a',
                      background: activeMetric === k ? '#eef2ff' : 'transparent' }}>
                      {ch[k] ?? '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Source attribution */}
      <div style={{ marginTop: 12, fontSize: 10, color: '#94a3b8', textAlign: 'center' }}>
        {overview.source} · Library: AntroPy v0.2.2 (BSD-3) · {defs?.library?.url}
      </div>
    </div>
  )
}
