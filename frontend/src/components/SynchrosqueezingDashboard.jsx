import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']
const BAND_COLORS = { delta: '#1e88e5', theta: '#7c4dff', alpha: '#4caf50', beta: '#ff9800', gamma: '#f44336' }

export default function SynchrosqueezingDashboard() {
  const [overview, setOverview] = useState(null)
  const [spectrum, setSpectrum] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, sp, df] = await Promise.all([
          axios.get(`${API_URL}/synchrosqueezing/overview?seconds=10`),
          axios.get(`${API_URL}/synchrosqueezing/spectrum?seconds=10`),
          axios.get(`${API_URL}/synchrosqueezing/definitions`)
        ])
        setOverview(ov.data)
        setSpectrum(sp.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load synchrosqueezing data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🌊</div>
      Computing synchrosqueezing transforms on real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'No EDF data available for synchrosqueezing analysis.'}
    </div>
  )

  const { per_channel, summary, frequency_bands, file, sfreq, n_channels, seconds_analyzed } = overview

  // Bar chart: per-channel peak frequency
  const peakFreqData = per_channel.map(ch => ({
    channel: ch.channel.length > 10 ? ch.channel.slice(0, 10) : ch.channel,
    peak_freq: ch.peak_frequency_hz,
    bandwidth: ch.bandwidth_hz,
  }))

  // Bar chart: per-channel energy (log10)
  const energyData = per_channel.map(ch => ({
    channel: ch.channel.length > 10 ? ch.channel.slice(0, 10) : ch.channel,
    energy: ch.mean_energy_log10,
  }))

  // Pie chart: frequency band proportions
  const bandData = Object.entries(frequency_bands)
    .filter(([, v]) => v.proportion > 0)
    .map(([band, v]) => ({
      name: band.charAt(0).toUpperCase() + band.slice(1),
      value: Math.round(v.proportion * 100),
      fill: BAND_COLORS[band] || '#999',
    }))

  // Radar: channel bandwidth comparison
  const radarData = per_channel.map(ch => ({
    channel: ch.channel.length > 8 ? ch.channel.slice(0, 8) : ch.channel,
    bandwidth: ch.bandwidth_hz,
    peak: ch.peak_frequency_hz,
  }))

  // Spectrum line chart data
  const spectrumData = spectrum?.available && spectrum.per_channel?.length > 0
    ? spectrum.per_channel[0].frequencies.map((f, i) => {
        const row = { freq: f }
        spectrum.per_channel.forEach((ch, ci) => {
          const label = ch.channel.length > 8 ? ch.channel.slice(0, 8) : ch.channel
          row[label] = ch.power[i] || 0
        })
        return row
      })
    : []

  const specChannels = spectrum?.available
    ? spectrum.per_channel.map(ch => ch.channel.length > 8 ? ch.channel.slice(0, 8) : ch.channel)
    : []

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

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          🌊 Synchrosqueezing Time-Frequency Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          SSQ-CWT via <b>ssqueezepy</b> (Muradeli, 2020) on real EEG — {file} · {sfreq} Hz · {n_channels} channels · {seconds_analyzed}s analyzed
        </p>
      </div>

      {/* KPI tiles */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={kpiStyle('#1e88e5')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Peak Freq (mean)</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{summary.peak_frequency_hz.mean} Hz</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>range: {summary.peak_frequency_hz.min}–{summary.peak_frequency_hz.max} Hz</div>
        </div>
        <div style={kpiStyle('#7c4dff')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Energy (log₁₀ mean)</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{summary.mean_energy_log10.mean} dB</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>range: {summary.mean_energy_log10.min} to {summary.mean_energy_log10.max}</div>
        </div>
        <div style={kpiStyle('#4caf50')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Channels</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{n_channels}</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>SSQ-CWT computed</div>
        </div>
        <div style={kpiStyle('#ff9800')}>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Dominant Band</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>
            {bandData.length > 0 ? bandData.sort((a, b) => b.value - a.value)[0].name : '—'}
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>
            {bandData.length > 0 ? `${bandData[0].value}% energy` : ''}
          </div>
        </div>
      </div>

      {/* Charts row 1: Peak Frequency + Band Pie */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Peak Frequency per Channel (Hz)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={peakFreqData} margin={{ left: 10, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="channel" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: 'Hz', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="peak_freq" name="Peak Freq (Hz)" fill="#1e88e5" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Frequency Band Energy</h3>
          {bandData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={bandData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={90} innerRadius={40} label={({ name, value }) => `${name} ${value}%`}>
                  {bandData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Pie>
                <Tooltip formatter={(v) => `${v}%`} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No band data</div>
          )}
        </div>
      </div>

      {/* Charts row 2: Energy + Bandwidth */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Mean Energy per Channel (log₁₀)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={energyData} margin={{ left: 10, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="channel" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: 'log₁₀', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="energy" name="Energy (log₁₀)" fill="#7c4dff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Frequency Bandwidth per Channel (Hz)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={peakFreqData} margin={{ left: 10, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="channel" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: 'Hz', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="bandwidth" name="Bandwidth (Hz)" fill="#4caf50" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Spectrum line chart */}
      {spectrumData.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            SSQ Marginal Frequency Spectrum (power vs frequency)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={spectrumData} margin={{ left: 10, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="freq" tick={{ fontSize: 10 }} label={{ value: 'Frequency (Hz)', position: 'insideBottom', offset: -2, fontSize: 12 }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: 'Power', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip />
              <Legend />
              {specChannels.map((ch, i) => (
                <Line key={ch} type="monotone" dataKey={ch} stroke={COLORS[i % COLORS.length]}
                  dot={false} strokeWidth={1.5} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-channel details table */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Per-Channel SSQ-CWT Details</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Channel', 'Peak Freq (Hz)', 'Energy (log₁₀)', 'Bandwidth (Hz)', 'Dominant Band', 'TF Matrix Shape'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {per_channel.map((ch, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#ffffff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 500 }}>{ch.channel}</td>
                  <td style={{ padding: '8px 10px' }}>{ch.peak_frequency_hz}</td>
                  <td style={{ padding: '8px 10px' }}>{ch.mean_energy_log10}</td>
                  <td style={{ padding: '8px 10px' }}>{ch.bandwidth_hz}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                      background: BAND_COLORS[ch.dominant_band] ? `${BAND_COLORS[ch.dominant_band]}18` : '#e2e8f0',
                      color: BAND_COLORS[ch.dominant_band] || '#475569',
                    }}>
                      {ch.dominant_band}
                    </span>
                  </td>
                  <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>
                    {ch.time_frequency_matrix_shape.join(' × ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '▾ Hide' : '▸ Show'} Method Definitions & Clinical Relevance
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, display: 'grid', gap: 14 }}>
            {Object.values(defs).filter(d => typeof d === 'object' && d.name).map((def, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 6 }}>{def.name}</div>
                <div style={{ color: '#475569', fontSize: 13, lineHeight: 1.5 }}>{def.description}</div>
                {def.reference && (
                  <div style={{ marginTop: 6, fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>{def.reference}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
