import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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
  const colorMap = { ready: '#22c55e', complete: '#22c55e', computed: '#22c55e', detected: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8', partial: '#eab308', failed: '#ef4444', high: '#22c55e', medium: '#eab308', low: '#ef4444', left: '#3b82f6', right: '#f97316', bilateral: '#8b5cf6', contralateral: '#06b6d4', ipsilateral: '#eab308', excellent: '#22c55e', good: '#3b82f6', fair: '#eab308', poor: '#ef4444', mp4: '#3b82f6', avi: '#f97316', mkv: '#8b5cf6', converted: '#22c55e', queued: '#eab308' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function VideoConverterDashboard() {
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
          axios.get(`${API_URL}/api/video-converter/overview`),
          axios.get(`${API_URL}/api/video-converter/breakdown`),
          axios.get(`${API_URL}/api/video-converter/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Video Converter data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'conversions', label: 'Conversions' },
    { id: 'quality', label: 'Quality' },
    { id: 'patients', label: 'Patients' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const formatDistribution = overview?.format_distribution || []
  const codecDistribution = overview?.codec_distribution || []
  const resolutionDistribution = overview?.resolution_distribution || []
  const pipelineStatus = overview?.pipeline_status || {}

  /* --- Breakdown data --- */
  const conversionDetails = breakdown?.conversion_details || []
  const qualityMetrics = breakdown?.conversion_quality_metrics || {}
  const perPatientSummary = breakdown?.per_patient_summary || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Video Converter Pipeline Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Video format/codec normalization and frame export for video-EEG recordings
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Recordings" value={fmt(overview.total_recordings)} />
              <KPI label="Video Capable" value={fmt(overview.video_capable)} color="#22c55e" />
              <KPI label="Total Frames Exported" value={fmt(overview.total_frames_exported)} color="#f97316" />
              <KPI label="Avg FPS" value={fmt(overview.avg_fps)} color="#3b82f6" />
              <KPI label="Total Storage (GB)" value={fmt(overview.total_storage_gb)} color="#8b5cf6" />
            </div>
          </Card>

          {/* Format Distribution */}
          <Card title="Format Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={formatDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="format" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {formatDistribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Codec Distribution */}
          <Card title="Codec Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={codecDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="codec" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {codecDistribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Resolution Distribution */}
          <Card title="Resolution Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={resolutionDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="resolution" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {resolutionDistribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pipeline Status */}
          <Card title="Pipeline Status" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {Object.entries(pipelineStatus).map(([stage, info], i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{(info.stage || stage).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    <StatusBadge status={info.status || 'pending'} />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Processed: {fmt(info.processed)} | Pending: {fmt(info.pending)}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ======================== CONVERSIONS TAB ======================== */}
      {tab === 'conversions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Recording Conversion Details Table */}
          <Card title="Per-Recording Conversion Details" span={2}>
            <div style={{ maxHeight: 480, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Recording ID</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Input Format</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Output Format</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Input Res</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Output Res</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Compression</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Quality (SSIM)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Level</th>
                  </tr>
                </thead>
                <tbody>
                  {conversionDetails.slice(0, 20).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.recording_id}</td>
                      <td style={{ padding: '6px 8px' }}>{r.patient || r.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.input_format || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.output_format || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.input_resolution || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.output_resolution || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{fmt(r.compression_ratio)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{fmt(r.quality_ssim)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={r.quality_level || 'good'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== QUALITY TAB ======================== */}
      {tab === 'quality' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* PSNR Distribution */}
          <Card title="PSNR Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={qualityMetrics.psnr_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {(qualityMetrics.psnr_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* SSIM Distribution */}
          <Card title="SSIM Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={qualityMetrics.ssim_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {(qualityMetrics.ssim_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Bitrate Distribution */}
          <Card title="Bitrate Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={qualityMetrics.bitrate_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {(qualityMetrics.bitrate_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Compression Ratio Distribution */}
          <Card title="Compression Ratio Distribution" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={qualityMetrics.compression_ratio_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {(qualityMetrics.compression_ratio_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== PATIENTS TAB ======================== */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient Video Summary" span={2}>
            <div style={{ maxHeight: 480, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Video Recordings</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total Frames</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total Storage (GB)</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Formats Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatientSummary.slice(0, 15).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.video_recordings)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.total_frames)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{fmt(p.total_storage_gb)}</td>
                      <td style={{ padding: '6px 8px' }}>{Array.isArray(p.formats_seen) ? p.formats_seen.join(', ') : (p.formats_seen || '--')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <Card title={definitions.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 180 }}>Term</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Definition</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 120 }}>Category</th>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 200 }}>Clinical Relevance</th>
              </tr>
            </thead>
            <tbody>
              {(definitions.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 180 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                  <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{d.category || '--'}</td>
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
