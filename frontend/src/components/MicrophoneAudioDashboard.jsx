import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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

function Badge({ value, colorMap }) {
  const color = colorMap?.[value] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{value}</span>
  )
}

const valueColors = { high: '#22c55e', medium: '#eab308', low: '#94a3b8', pipeline: '#3b82f6', primary: '#3b82f6', secondary: '#8b5cf6', ambulatory: '#f97316', config: '#64748b' }
const qualColors = { good: '#22c55e', fair: '#eab308', poor: '#ef4444' }

export default function MicrophoneAudioDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/api/microphone-audio/overview`),
          axios.get(`${API_URL}/api/microphone-audio/breakdown`),
          axios.get(`${API_URL}/api/microphone-audio/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Microphone Audio data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: '🎙️ Overview' },
    { id: 'breakdown', label: '📊 Breakdown' },
    { id: 'definitions', label: '📚 Definitions' },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>🎙️ Microphone Audio Capture</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 13 }}>
          Vocalization detection during seizures — ictal cry, automatisms, postictal speech, respiratory patterns
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && (
        <>
          {/* KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients" value={fmt(overview.summary.n_patients)} /></Card>
            <Card><KPI label="Recordings" value={fmt(overview.summary.n_recordings)} /></Card>
            <Card><KPI label="Audio-Capable %" value={fmt(overview.summary.audio_capable_pct) + '%'} color="#3b82f6" /></Card>
            <Card><KPI label="Vocalizations Detected" value={fmt(overview.summary.vocalizations_detected)} color="#8b5cf6" /></Card>
            <Card><KPI label="Ictal Events Confirmed" value={fmt(overview.summary.ictal_events_confirmed)} color="#22c55e" /></Card>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="VAD Precision" value={fmtPct(overview.summary.avg_vad_precision)} color="#f97316" /></Card>
            <Card><KPI label="VAD Recall" value={fmtPct(overview.summary.avg_vad_recall)} color="#06b6d4" /></Card>
            <Card><KPI label="Avg SNR" value={fmt(overview.summary.avg_snr_db) + ' dB'} color="#22c55e" /></Card>
            <Card><KPI label="Pipeline Latency" value={fmt(overview.summary.pipeline_latency_ms) + ' ms'} color="#eab308" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Vocalization Type Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.vocalization_distribution} margin={{ top: 4, right: 8, left: 0, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Count">
                    {overview.vocalization_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Capture Source — Sessions">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.capture_sources} margin={{ top: 4, right: 8, left: 0, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="source" tick={{ fontSize: 10 }} angle={-25} textAnchor="end" interval={0} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="sessions" name="Sessions" fill="#3b82f6" />
                  <Bar dataKey="snr_db" name="SNR (dB)" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* SNR trend */}
          <div style={{ marginBottom: 20 }}>
            <Card title="SNR Trend Over Time">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={overview.snr_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="snr_db" name="SNR (dB)" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="sessions" name="Sessions" stroke="#22c55e" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Pipeline stages */}
          <Card title="Audio Processing Pipeline">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
              {overview.pipeline_stages.map((s, i) => (
                <div key={i} style={{ background: '#f1f5f9', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>{i + 1}. {s.stage}</div>
                  <Badge value={s.status} colorMap={{ active: '#22c55e' }} />
                  {s.sample_rate_hz && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.sample_rate_hz.toLocaleString()} Hz / {s.bit_depth}-bit</div>}
                  {s.model && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.model}</div>}
                  {s.n_coefficients && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{s.n_coefficients} MFCCs</div>}
                  {s.lag_ms && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>Lag: {s.lag_ms} ms</div>}
                </div>
              ))}
            </div>
          </Card>
        </>
      )}

      {tab === 'breakdown' && breakdown && (
        <>
          {/* MFCC scatter */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="MFCC C1 vs C2 — Vocalization Scatter">
              <ResponsiveContainer width="100%" height={240}>
                <ScatterChart margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="mfcc1" name="MFCC C1" tick={{ fontSize: 11 }} />
                  <YAxis dataKey="mfcc2" name="MFCC C2" tick={{ fontSize: 11 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
                    if (!payload?.length) return null
                    const d = payload[0].payload
                    return <div style={{ background: '#fff', border: '1px solid #e2e8f0', padding: 8, borderRadius: 6, fontSize: 11 }}>
                      <div><b>{d.label}</b></div>
                      <div>C1: {d.mfcc1} | C2: {d.mfcc2}</div>
                    </div>
                  }} />
                  <Scatter data={breakdown.mfcc_scatter} fill="#3b82f6" opacity={0.7} />
                </ScatterChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Feature Importance">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={breakdown.feature_importance} layout="vertical" margin={{ top: 4, right: 20, left: 120, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={120} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Bar dataKey="importance" name="Importance">
                    {breakdown.feature_importance.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Event timeline */}
          <Card title="Vocalization Event Timeline" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Event', 'Onset (s)', 'Duration (s)', 'Vocalization Type', 'Clinical Value', 'Seizure?', 'Confidence'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#334155', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.event_timeline.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                      <td style={{ padding: '6px 10px' }}>{ev.event_id}</td>
                      <td style={{ padding: '6px 10px' }}>{ev.onset_sec}</td>
                      <td style={{ padding: '6px 10px' }}>{ev.duration_sec}</td>
                      <td style={{ padding: '6px 10px' }}>{ev.vocalization_type}</td>
                      <td style={{ padding: '6px 10px' }}><Badge value={ev.clinical_value} colorMap={valueColors} /></td>
                      <td style={{ padding: '6px 10px' }}>
                        <span style={{ color: ev.coincides_with_seizure ? '#22c55e' : '#94a3b8', fontWeight: 600 }}>
                          {ev.coincides_with_seizure ? '✓ Yes' : '—'}
                        </span>
                      </td>
                      <td style={{ padding: '6px 10px' }}>{(ev.confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Patient profiles */}
          <div style={{ marginTop: 16 }}>
            <Card title="Patient Audio Profiles">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      {['Patient', 'Age', 'Diagnosis', 'Vocalizations', 'Ictal Cry', 'Dominant Type', 'MFCC C1', 'ZCR', 'Quality'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#334155', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.patient_profiles.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                        <td style={{ padding: '6px 10px' }}>{p.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{p.age || '--'}</td>
                        <td style={{ padding: '6px 10px' }}>{p.diagnosis || '--'}</td>
                        <td style={{ padding: '6px 10px' }}>{p.vocalizations}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <span style={{ color: p.ictal_cry_present ? '#22c55e' : '#94a3b8', fontWeight: 600 }}>
                            {p.ictal_cry_present ? '✓' : '—'}
                          </span>
                        </td>
                        <td style={{ padding: '6px 10px', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.dominant_vocal_type}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(p.mfcc_c1)}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(p.zero_crossing_rate)}</td>
                        <td style={{ padding: '6px 10px' }}><Badge value={p.audio_quality} colorMap={qualColors} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          <h3 style={{ margin: 0, color: '#1e293b' }}>{definitions.title}</h3>
          {definitions.sections.map((sec, si) => (
            <Card key={si} title={sec.section}>
              <div style={{ display: 'grid', gap: 10 }}>
                {sec.terms.map((t, ti) => (
                  <div key={ti} style={{ background: '#f8fafc', borderRadius: 8, padding: '10px 14px', borderLeft: '3px solid ' + (valueColors[t.clinical_relevance] || '#94a3b8') }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{t.term}</span>
                      <Badge value={t.clinical_relevance} colorMap={valueColors} />
                    </div>
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>{t.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
          <Card title="References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {definitions.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{r}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
