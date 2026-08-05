import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, Legend,
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined,
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const CONTACT_COLOR = { good: '#10b981', marginal: '#f59e0b', poor: '#ef4444' }
const BAND_COLORS = {
  delta: '#6366f1', theta: '#3b82f6', alpha: '#10b981', beta: '#f59e0b', gamma: '#ef4444',
}
const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'channels', label: 'Channel Analysis' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EmotivFlexDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [channels, setChannels] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/emotiv-flex/overview`),
      axios.get(`${API_URL}/api/emotiv-flex/channels`),
      axios.get(`${API_URL}/api/emotiv-flex/definitions`),
    ]).then(([o, c, d]) => {
      setOverview(o.data)
      setChannels(c.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Emotiv EPOC Flex data…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Emotiv EPOC Flex Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          32-channel research-grade EEG cap · 2048 Hz · online + offline modes ·
          impedance monitoring · band-power analysis · seizure detection
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t.id ? '#6366f1' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontWeight: tab === t.id ? 600 : 400,
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 12, marginBottom: 20 }}>
            <Card><KPI label="Sessions" value={k.total_sessions} /></Card>
            <Card><KPI label="Uploaded" value={k.uploaded} color="#10b981" /></Card>
            <Card><KPI label="Avg Signal Quality" value={k.avg_signal_quality_pct != null ? `${k.avg_signal_quality_pct}%` : '--'} color="#6366f1" /></Card>
            <Card><KPI label="Seizure Events" value={k.total_seizure_events} color="#ef4444" /></Card>
            <Card><KPI label="Avg Alpha Peak" value={k.avg_alpha_peak_hz != null ? `${k.avg_alpha_peak_hz} Hz` : '--'} color="#10b981" /></Card>
            <Card><KPI label="Channels" value={k.channels} sub="10-20 layout" /></Card>
            <Card><KPI label="Sample Rate" value={k.sampling_rate_hz != null ? `${k.sampling_rate_hz} Hz` : '--'} /></Card>
          </div>

          {/* Band power trend + impedance bars */}
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="7-Day Band Power Trend (relative %)">
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={overview?.band_trend_7d || []} stackOffset="expand">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                  <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n]} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                  {['delta', 'theta', 'alpha', 'beta', 'gamma'].map(b => (
                    <Area key={b} type="monotone" dataKey={b} stackId="1"
                      stroke={BAND_COLORS[b]} fill={BAND_COLORS[b]} fillOpacity={0.7} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </Card>

            {/* Impedance contact quality summary */}
            <Card title="Impedance Contact Quality">
              {overview?.impedance_map && (() => {
                const imp = overview.impedance_map
                const good = imp.filter(c => c.contact === 'good').length
                const marginal = imp.filter(c => c.contact === 'marginal').length
                const poor = imp.filter(c => c.contact === 'poor').length
                return (
                  <div>
                    {[
                      { label: 'Good (<10 kΩ)', count: good, color: '#10b981' },
                      { label: 'Marginal (10–25 kΩ)', count: marginal, color: '#f59e0b' },
                      { label: 'Poor (>25 kΩ)', count: poor, color: '#ef4444' },
                    ].map(row => (
                      <div key={row.label} style={{ marginBottom: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                          <span style={{ color: '#64748b' }}>{row.label}</span>
                          <span style={{ fontWeight: 600, color: row.color }}>{row.count} ch</span>
                        </div>
                        <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8 }}>
                          <div style={{ width: `${(row.count / 32) * 100}%`, background: row.color, height: 8, borderRadius: 4 }} />
                        </div>
                      </div>
                    ))}
                    <div style={{ marginTop: 16, fontSize: 12, color: '#94a3b8', textAlign: 'center' }}>
                      32 channels total
                    </div>
                  </div>
                )
              })()}
            </Card>
          </div>

          {/* Sessions table */}
          <Card title="Recording Sessions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Session', 'Patient', 'Date', 'Duration (min)', 'Good Ch', 'Signal Q%', 'Motion Art%', 'Seizure Events', 'Alpha Peak', 'ICA Removed', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview?.sessions || []).map((s, i) => (
                    <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '7px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.session_id}</td>
                      <td style={{ padding: '7px 10px' }}>{s.patient_id}</td>
                      <td style={{ padding: '7px 10px' }}>{s.date}</td>
                      <td style={{ padding: '7px 10px' }}>{s.duration_min}</td>
                      <td style={{ padding: '7px 10px' }}>{s.channels_good}/32</td>
                      <td style={{ padding: '7px 10px', color: s.signal_quality_pct < 80 ? '#f59e0b' : '#10b981' }}>{s.signal_quality_pct}%</td>
                      <td style={{ padding: '7px 10px', color: s.motion_artifact_pct > 8 ? '#ef4444' : '#475569' }}>{s.motion_artifact_pct}%</td>
                      <td style={{ padding: '7px 10px', color: s.seizure_events > 0 ? '#ef4444' : '#10b981', fontWeight: s.seizure_events > 0 ? 600 : 400 }}>{s.seizure_events}</td>
                      <td style={{ padding: '7px 10px' }}>{s.alpha_peak_hz} Hz</td>
                      <td style={{ padding: '7px 10px' }}>{s.ica_components_removed}</td>
                      <td style={{ padding: '7px 10px' }}>
                        <span style={{
                          padding: '2px 7px', borderRadius: 10, fontSize: 10,
                          background: s.upload_status === 'uploaded' ? '#10b98122' : '#f59e0b22',
                          color: s.upload_status === 'uploaded' ? '#10b981' : '#f59e0b',
                        }}>{s.upload_status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── CHANNELS TAB ── */}
      {tab === 'channels' && channels && (
        <>
          {/* Summary KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
            <Card><KPI label="Good Channels" value={channels.summary?.good} color="#10b981" sub={`${channels.summary?.good_pct}%`} /></Card>
            <Card><KPI label="Marginal Channels" value={channels.summary?.marginal} color="#f59e0b" /></Card>
            <Card><KPI label="Poor Channels" value={channels.summary?.poor} color="#ef4444" /></Card>
            <Card><KPI label="Total Channels" value={32} sub="10-20 standard layout" /></Card>
          </div>

          {/* Impedance bar chart */}
          <Card title="Per-Channel Impedance (kΩ)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={channels?.channels || []} margin={{ left: 0, right: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="channel" tick={{ fontSize: 9 }} interval={0} angle={-45} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} unit=" kΩ" />
                <Tooltip formatter={(v, n) => [`${v} kΩ`, 'Impedance']} />
                <Bar dataKey="impedance_kohm" radius={[3, 3, 0, 0]}>
                  {(channels?.channels || []).map((c, i) => (
                    <rect key={i} fill={CONTACT_COLOR[c.contact] || '#94a3b8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* SNR bar chart */}
          <div style={{ marginTop: 16 }}>
            <Card title="Per-Channel SNR (dB)">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={channels?.channels || []} margin={{ left: 0, right: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="channel" tick={{ fontSize: 9 }} interval={0} angle={-45} textAnchor="end" height={50} />
                  <YAxis domain={[0, 40]} tick={{ fontSize: 11 }} unit=" dB" />
                  <Tooltip formatter={(v) => [`${v} dB`, 'SNR']} />
                  <Bar dataKey="snr_db" fill="#6366f1" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Frequency spectrum */}
          <div style={{ marginTop: 16 }}>
            <Card title="Mean Power Spectrum (0–50 Hz, averaged across channels)">
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={(channels?.spectrum || []).filter(s => s.freq_hz <= 50)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="freq_hz" tick={{ fontSize: 11 }} unit=" Hz" />
                  <YAxis tick={{ fontSize: 11 }} unit=" μV²" />
                  <Tooltip formatter={v => [`${v} μV²`, 'Power']} />
                  <Line type="monotone" dataKey="power_uv2" stroke="#6366f1" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Channel table */}
          <div style={{ marginTop: 16 }}>
            <Card title="Channel Detail">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      {['Channel', 'Impedance (kΩ)', 'Contact', 'SNR (dB)', 'Artifact %', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'Alpha Peak'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 11, whiteSpace: 'nowrap' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(channels?.channels || []).map((c, i) => (
                      <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '7px 10px', fontFamily: 'monospace', fontWeight: 600 }}>{c.channel}</td>
                        <td style={{ padding: '7px 10px', color: CONTACT_COLOR[c.contact] }}>{c.impedance_kohm}</td>
                        <td style={{ padding: '7px 10px' }}>
                          <span style={{
                            padding: '2px 7px', borderRadius: 10, fontSize: 10,
                            background: CONTACT_COLOR[c.contact] + '22',
                            color: CONTACT_COLOR[c.contact],
                          }}>{c.contact}</span>
                        </td>
                        <td style={{ padding: '7px 10px', color: c.snr_db < 15 ? '#ef4444' : c.snr_db < 20 ? '#f59e0b' : '#10b981' }}>{c.snr_db}</td>
                        <td style={{ padding: '7px 10px', color: c.artifact_pct > 8 ? '#ef4444' : '#475569' }}>{c.artifact_pct}%</td>
                        {['delta', 'theta', 'alpha', 'beta', 'gamma'].map(b => (
                          <td key={b} style={{ padding: '7px 10px', color: BAND_COLORS[b] }}>
                            {((c.band_power?.[b] || 0) * 100).toFixed(1)}%
                          </td>
                        ))}
                        <td style={{ padding: '7px 10px' }}>{c.alpha_peak_hz} Hz</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && definitions && (
        <>
          {/* Device specs */}
          <Card title="Device Specification">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {Object.entries(definitions.device || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#f8fafc', borderRadius: 8, padding: '10px 14px' }}>
                  <div style={{ fontSize: 11, color: '#94a3b8', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 13, color: '#1e293b', marginTop: 3, fontWeight: 500 }}>
                    {Array.isArray(v) ? v.join(', ') : String(v)}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Epilepsy context */}
          <div style={{ marginTop: 16 }}>
            <Card title="Epilepsy Research Context">
              <p style={{ margin: 0, fontSize: 13, color: '#334155', lineHeight: 1.7 }}>
                {definitions.epilepsy_context}
              </p>
            </Card>
          </div>

          {/* Metric glossary */}
          <div style={{ marginTop: 16 }}>
            <Card title="EEG Metric Glossary">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Metric', 'Normal Range', 'Clinical Note'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#64748b', fontWeight: 600, fontSize: 12 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(definitions.metrics || []).map((m, i) => (
                    <tr key={i} style={{ borderTop: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b', whiteSpace: 'nowrap' }}>{m.term}</td>
                      <td style={{ padding: '8px 12px', color: '#10b981', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{m.normal_range}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{m.clinical_note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>

          {/* References */}
          <div style={{ marginTop: 16 }}>
            <Card title="References">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(definitions.references || []).map((r, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{r}</li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
