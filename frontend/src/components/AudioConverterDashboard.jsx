import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function StatusBadge({ status }) {
  const colorMap = { ready: '#22c55e', complete: '#22c55e', computed: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8', partial: '#eab308', failed: '#ef4444', high: '#22c55e', medium: '#eab308', low: '#ef4444' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function AudioConverterDashboard() {
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
          axios.get(`${API_URL}/audio-converter/overview`),
          axios.get(`${API_URL}/audio-converter/breakdown`),
          axios.get(`${API_URL}/audio-converter/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Audio Converter data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'extraction', label: 'Extraction' },
    { id: 'vocalizations', label: 'Vocalizations' },
    { id: 'features', label: 'Features' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const recordingTypeBreakdown = overview?.recording_type_breakdown || []
  const pipelineStatus = overview?.pipeline_status || {}
  const audioChannelAnalysis = overview?.audio_channel_analysis || {}

  /* --- Extraction data --- */
  const extractionReadiness = overview?.extraction_readiness || []
  const perRecordingSegments = breakdown?.per_recording_segments || []

  /* --- Vocalizations data --- */
  const vocalizationEvents = breakdown?.vocalization_events || []

  /* --- Features data --- */
  const audioFeatureExtraction = breakdown?.audio_feature_extraction || []
  const speechVsNonspeechRatios = breakdown?.speech_vs_nonspeech_ratios || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Audio Converter Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Audio conversion and extraction analytics for EEG recordings with vocalization detection and speech marker analysis
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Recordings" value={fmt(overview.total_recordings)} />
              <KPI label="Audio-Capable" value={fmt(overview.audio_capable_recordings)} color="#22c55e" />
              <KPI label="Extraction Ready" value={fmt(overview.audio_extraction_ready)} color="#3b82f6" />
              <KPI label="Total Audio Hours" value={fmt(audioChannelAnalysis.total_audio_hours)} color="#8b5cf6" />
            </div>
          </Card>

          {/* Recording Type Breakdown Pie Chart */}
          <Card title="Recording Type Breakdown" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={recordingTypeBreakdown} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={90} label={({ type, count }) => `${type}: ${count}`}>
                  {recordingTypeBreakdown.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Pipeline Status Cards */}
          <Card title="Pipeline Status" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(pipelineStatus.stages || pipelineStatus).map(([stage, status], i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{stage.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                  <StatusBadge status={typeof status === 'string' ? status : status?.status || 'pending'} />
                </div>
              ))}
            </div>
          </Card>

          {/* Audio Channel Analysis Summary */}
          <Card title="Audio Channel Analysis" span={3}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(audioChannelAnalysis).map(([key, value], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155', width: 280 }}>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                    <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{fmt(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ======================== EXTRACTION TAB ======================== */}
      {tab === 'extraction' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Extraction Readiness Table */}
          <Card title="Per-Recording Extraction Readiness" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Recording Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Duration (min)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sampling Rate</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Audio Extractable</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Quality Estimate</th>
                  </tr>
                </thead>
                <tbody>
                  {extractionReadiness.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.recording_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.duration_min)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.sampling_rate)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={r.audio_extractable ? 'ready' : 'pending'} />
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.audio_quality_estimate || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Audio Quality Distribution Bar Chart */}
          <Card title="Audio Quality Distribution" span={2}>
            {(() => {
              const qualityCounts = {}
              extractionReadiness.forEach(r => {
                const q = r.audio_quality_estimate || 'unknown'
                qualityCounts[q] = (qualityCounts[q] || 0) + 1
              })
              const qualityData = Object.entries(qualityCounts).map(([quality, count]) => ({ quality, count }))
              return (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={qualityData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                      {qualityData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>
        </div>
      )}

      {/* ======================== VOCALIZATIONS TAB ======================== */}
      {tab === 'vocalizations' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Vocalization Events Table */}
          <Card title="Vocalization Events" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Event Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizure Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Likelihood</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Duration (s)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Clinical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {vocalizationEvents.map((v, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{v.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{v.event_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{v.seizure_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(v.vocalization_likelihood)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(v.duration_estimate_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={v.clinical_significance || 'pending'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Vocalization Types Bar Chart */}
          <Card title="Vocalization Types Distribution" span={2}>
            {(() => {
              const typeCounts = {}
              vocalizationEvents.forEach(v => {
                const t = v.event_type || 'unknown'
                typeCounts[t] = (typeCounts[t] || 0) + 1
              })
              const typeData = Object.entries(typeCounts).map(([event_type, count]) => ({ event_type, count }))
              return (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={typeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="event_type" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" name="Events" radius={[4, 4, 0, 0]}>
                      {typeData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>
        </div>
      )}

      {/* ======================== FEATURES TAB ======================== */}
      {tab === 'features' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Audio Feature Extraction Stats Table */}
          <Card title="Audio Feature Extraction Stats" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Feature Type</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Values Computed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Value</th>
                  </tr>
                </thead>
                <tbody>
                  {audioFeatureExtraction.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{f.feature_type}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{f.description}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={f.computation_status || 'pending'} />
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(f.values_computed)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(f.avg_value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Speech vs Non-Speech Ratio Chart */}
          <Card title="Speech vs Non-Speech Ratios" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={speechVsNonspeechRatios}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="speech_ratio" fill={COLORS[0]} name="Speech" radius={[4, 4, 0, 0]} />
                <Bar dataKey="nonspeech_ratio" fill={COLORS[2]} name="Non-Speech" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <Card title={definitions.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 200 }}>Term</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Definition</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 200 }}>Clinical Relevance</th>
              </tr>
            </thead>
            <tbody>
              {(definitions.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 200 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                  <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{d.clinical_relevance || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
