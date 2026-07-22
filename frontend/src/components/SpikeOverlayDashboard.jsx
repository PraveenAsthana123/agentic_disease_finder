import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, ReferenceLine
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const COLORS = {
  blue: '#1e88e5',
  red: '#ef4444',
  green: '#22c55e',
  amber: '#f59e0b',
  purple: '#7c4dff',
}

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

const sectionHeadingStyle = {
  fontSize: 16,
  fontWeight: 600,
  color: '#1e293b',
  marginBottom: 12,
  marginTop: 24,
}

function KPITile({ label, value, color }) {
  return (
    <div style={{ ...cardStyle, borderLeft: `4px solid ${color}`, minWidth: 150, flex: 1 }}>
      <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value != null ? value.toLocaleString() : '--'}</div>
    </div>
  )
}

export default function SpikeOverlayDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/api/spike-overlay`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load spike overlay data')
      }
      setLoading(false)
    }
    load()
  }, [])

  const channelData = useMemo(() => {
    if (!data?.per_channel_summary) return []
    return [...data.per_channel_summary]
      .map(c => ({ ...c, total: (c.spikes || 0) + (c.sharp_waves || 0) }))
      .sort((a, b) => b.total - a.total)
      .slice(0, 12)
  }, [data])

  const subjectData = useMemo(() => {
    if (!data?.per_subject_summary) return []
    return data.per_subject_summary
  }, [data])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Detecting spikes and sharp waves in CHB-MIT EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {data?.note || 'Spike overlay data not available. Ensure CHB-MIT EEG data is loaded.'}
    </div>
  )

  const totalEvents = (data.total_spikes || 0) + (data.total_sharp_waves || 0)
  const events = (data.events || []).slice(0, 100)
  const waveforms = data.waveform_samples || []

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Spike &amp; Sharp-Wave Overlay Dashboard &mdash; CHB-MIT
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          {data.total_spikes?.toLocaleString()} spikes &middot; {data.total_sharp_waves?.toLocaleString()} sharp waves &middot; {data.subjects_analyzed} subjects analyzed
        </p>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        <KPITile label="Total Spikes" value={data.total_spikes} color={COLORS.red} />
        <KPITile label="Total Sharp Waves" value={data.total_sharp_waves} color={COLORS.amber} />
        <KPITile label="Subjects Analyzed" value={data.subjects_analyzed} color={COLORS.blue} />
        <KPITile label="Events Detected" value={totalEvents} color={COLORS.green} />
      </div>

      {/* Per-Channel Detection */}
      <h3 style={sectionHeadingStyle}>Per-Channel Detection (Top 12)</h3>
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={channelData} layout="vertical" margin={{ top: 10, right: 20, left: 80, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} />
            <YAxis type="category" dataKey="channel" tick={{ fontSize: 12, fill: '#475569' }} width={75} />
            <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
            <Legend />
            <Bar dataKey="spikes" stackId="a" fill={COLORS.red} name="Spikes" radius={[0, 0, 0, 0]} />
            <Bar dataKey="sharp_waves" stackId="a" fill={COLORS.amber} name="Sharp Waves" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-Subject Summary */}
      <h3 style={sectionHeadingStyle}>Per-Subject Summary</h3>
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={subjectData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="subject" tick={{ fontSize: 12, fill: '#475569' }} />
            <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
            <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
            <Legend />
            <Bar dataKey="spikes" stackId="a" fill={COLORS.red} name="Spikes" radius={[0, 0, 0, 0]} />
            <Bar dataKey="sharp_waves" stackId="a" fill={COLORS.amber} name="Sharp Waves" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Waveform Samples */}
      {waveforms.length > 0 && (
        <>
          <h3 style={sectionHeadingStyle}>Waveform Samples</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
            {waveforms.map((w, i) => {
              const chartData = (w.times || []).map((t, j) => ({ time: t, amplitude: (w.values || [])[j] }))
              const color = w.type === 'spike' ? COLORS.red : COLORS.amber
              return (
                <div key={i} style={{ ...cardStyle, padding: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 4 }}>
                    {w.type} &mdash; {w.channel} ({w.subject})
                  </div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                    Peak: {w.peak_uv?.toFixed(1)} &micro;V
                  </div>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                      <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: 'ms', position: 'insideBottomRight', fontSize: 10, fill: '#94a3b8' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#94a3b8' }} label={{ value: '\u00B5V', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#94a3b8' }} />
                      <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }} formatter={(v) => [v?.toFixed(1) + ' \u00B5V', 'Amplitude']} />
                      <Line type="monotone" dataKey="amplitude" stroke={color} strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Event Table */}
      <h3 style={sectionHeadingStyle}>Event Table</h3>
      <div style={{ ...cardStyle, marginBottom: 16, maxHeight: 300, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#ffffff' }}>
              {['Subject', 'File', 'Type', 'Channel', 'Time (s)', 'Amplitude (\u00B5V)', 'Duration (ms)'].map(h => (
                <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => {
              const typeColor = e.type === 'spike' ? COLORS.red : COLORS.amber
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', color: '#1e293b' }}>{e.subject}</td>
                  <td style={{ padding: '6px 10px', color: '#475569', fontFamily: 'monospace', fontSize: 11 }}>{e.file}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={{ background: typeColor, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>
                      {e.type === 'spike' ? 'Spike' : 'Sharp Wave'}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px', color: '#334155' }}>{e.channel}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: '#334155' }}>{e.time_sec?.toFixed(1)}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: '#334155' }}>{e.amplitude_uv?.toFixed(1)}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: '#334155' }}>{e.duration_ms?.toFixed(1)}</td>
                </tr>
              )
            })}
            {events.length === 0 && (
              <tr><td colSpan={7} style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No events detected</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 24, paddingBottom: 20 }}>
        Data source: CHB-MIT Scalp EEG Database &middot; Real spike/sharp-wave detection via MAD threshold
      </div>
    </div>
  )
}
