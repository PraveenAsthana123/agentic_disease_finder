import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function FlatnessBadge({ value }) {
  const v = parseFloat(value)
  let color = '#10b981', label = 'Tonal'
  if (v > 0.5) { color = '#f59e0b'; label = 'Mixed' }
  if (v > 0.8) { color = '#ef4444'; label = 'Noisy' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{label} ({v.toFixed(4)})</span>
  )
}

export default function LibrosaDashboard() {
  const [overview, setOverview] = useState(null)
  const [heatmap, setHeatmap] = useState(null)
  const [melData, setMelData] = useState(null)
  const [mfccData, setMfccData] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, hm, mel, mfcc, df] = await Promise.all([
          axios.get(`${API_URL}/librosa/overview`),
          axios.get(`${API_URL}/librosa/heatmap`),
          axios.get(`${API_URL}/librosa/mel-spectrogram`),
          axios.get(`${API_URL}/librosa/mfcc`),
          axios.get(`${API_URL}/librosa/definitions`)
        ])
        setOverview(ov.data)
        setHeatmap(hm.data)
        setMelData(mel.data)
        setMfccData(mfcc.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load librosa spectral data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading librosa spectral features...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (overview?.error) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview.error}</div>

  const tabs = ['overview', 'heatmap', 'mel', 'mfcc', 'definitions']
  const tabLabels = { overview: 'Overview', heatmap: 'Heatmap', mel: 'Mel Spectrogram', mfcc: 'MFCC Profile', definitions: 'Definitions' }

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Librosa Spectral Features</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Real EEG spectral analysis — centroid, bandwidth, rolloff, flatness, ZCR, MFCC, mel spectrogram
        {overview?.tool && <span> &middot; librosa {overview.version}</span>}
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{tabLabels[t]}</button>
        ))}
      </div>

      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Recording Info" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
              <KPI label="Channels" value={overview.n_channels} />
              <KPI label="Sample Rate" value={overview.sfreq} sub="Hz" />
              <KPI label="Duration" value={overview.duration_sec} sub="sec" />
              <KPI label="Mean Centroid" value={overview.summary?.mean_centroid_hz} sub="Hz" color="#3b82f6" />
              <KPI label="Mean Flatness" value={overview.summary?.mean_flatness} color="#f59e0b" />
              <KPI label="Most Tonal Ch" value={overview.summary?.min_flatness_ch} color="#10b981" />
            </div>
          </Card>

          <Card title="Spectral Centroid by Channel (Hz)" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(overview.channels || []).map(c => ({ channel: c.channel, centroid: c.spectral_centroid_hz }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="centroid" fill="#3b82f6" name="Centroid (Hz)" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Spectral Flatness by Channel">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(overview.channels || []).map(c => ({ channel: c.channel, flatness: c.spectral_flatness }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
                <Tooltip />
                <Bar dataKey="flatness" fill="#f59e0b" name="Flatness (0=tonal, 1=noisy)" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Zero-Crossing Rate by Channel">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(overview.channels || []).map(c => ({ channel: c.channel, zcr: c.zero_crossing_rate }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="channel" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="zcr" fill="#10b981" name="ZCR" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Channel Spectral Summary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Channel', 'Centroid (Hz)', 'Bandwidth (Hz)', 'Rolloff (Hz)', 'Flatness', 'ZCR'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.channels || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.channel}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(c.spectral_centroid_hz)}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(c.spectral_bandwidth_hz)}</td>
                      <td style={{ padding: '6px 10px' }}>{fmt(c.spectral_rolloff_hz)}</td>
                      <td style={{ padding: '6px 10px' }}><FlatnessBadge value={c.spectral_flatness} /></td>
                      <td style={{ padding: '6px 10px' }}>{c.zero_crossing_rate?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'heatmap' && heatmap && !heatmap.error && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="Channels x Spectral Metrics Heatmap" span={2}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>{heatmap.note}</p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>Channel</th>
                    {(heatmap.metrics || []).map(m => (
                      <th key={m} style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, textTransform: 'capitalize' }}>{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(heatmap.channels || []).map((ch, ri) => {
                    const row = heatmap.matrix?.[ri] || []
                    return (
                      <tr key={ri} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{ch}</td>
                        {row.map((val, ci) => {
                          const allVals = (heatmap.matrix || []).map(r => r[ci] || 0)
                          const min = Math.min(...allVals)
                          const max = Math.max(...allVals)
                          const norm = max > min ? (val - min) / (max - min) : 0.5
                          const r = Math.round(255 * (1 - norm))
                          const g = Math.round(100 + 155 * norm)
                          const b = Math.round(255 * norm)
                          return (
                            <td key={ci} style={{
                              padding: '6px 10px', textAlign: 'center',
                              background: `rgba(${r}, ${g}, ${b}, 0.15)`,
                              fontWeight: norm > 0.7 ? 700 : 400
                            }}>{typeof val === 'number' ? val.toFixed(4) : val}</td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'mel' && melData && !melData.error && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Average Mel Spectrogram (dB)" span={2}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 8px' }}>
              Mean dB power per mel bin across {melData.n_channels} channels
            </p>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={(melData.average_mel_power_dB || []).map((v, i) => ({
                bin: i,
                freq: melData.mel_freqs_hz?.[i]?.toFixed(1) || i,
                power: v
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="freq" label={{ value: 'Mel Frequency (Hz)', position: 'insideBottom', offset: -5, fontSize: 11 }} tick={{ fontSize: 9 }} />
                <YAxis label={{ value: 'Power (dB)', angle: -90, position: 'insideLeft', fontSize: 11 }} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => [v?.toFixed(2) + ' dB', 'Power']} labelFormatter={l => `${l} Hz`} />
                <Bar dataKey="power" fill="#8b5cf6" name="Mel Power (dB)" radius={[2,2,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Channel Mel Summary">
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Channel', 'Mel Bins', 'Time Frames'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(melData.channels || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.channel}</td>
                      <td style={{ padding: '6px 10px' }}>{c.mel_bins?.length || 0}</td>
                      <td style={{ padding: '6px 10px' }}>{c.n_time_frames || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'mfcc' && mfccData && !mfccData.error && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Average MFCC Profile (13 Coefficients)" span={2}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 8px' }}>{mfccData.note}</p>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(mfccData.average_mfcc || []).map((v, i) => ({ coeff: `MFCC-${i}`, value: v }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="coeff" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => [v?.toFixed(4), 'Value']} />
                <Bar dataKey="value" fill="#06b6d4" name="MFCC Value" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Channel MFCC Table" span={2}>
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, position: 'sticky', left: 0, background: '#f8fafc' }}>Channel</th>
                    {Array.from({ length: 13 }, (_, i) => (
                      <th key={i} style={{ padding: '8px 6px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>M{i}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(mfccData.channels || []).map((c, ri) => (
                    <tr key={ri} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600, position: 'sticky', left: 0, background: '#fff' }}>{c.channel}</td>
                      {(c.mfcc_means || []).map((v, ci) => (
                        <td key={ci} style={{ padding: '6px 6px', textAlign: 'center' }}>{v?.toFixed(2)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card title="Spectral Feature Definitions" span={2}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>
              librosa {defs.version} &mdash; spectral features, clinical interpretation, and references
            </p>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Feature', 'Unit', 'Description', 'Clinical Relevance', 'Reference'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(defs.features || []).map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.name}</td>
                    <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{f.unit}</td>
                    <td style={{ padding: '8px 10px', maxWidth: 260 }}>{f.description}</td>
                    <td style={{ padding: '8px 10px', maxWidth: 260, color: '#475569' }}>{f.clinical}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8' }}>{f.reference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
