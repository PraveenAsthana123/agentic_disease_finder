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
  const colorMap = { ready: '#22c55e', complete: '#22c55e', computed: '#22c55e', detected: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8', partial: '#eab308', failed: '#ef4444', high: '#22c55e', medium: '#eab308', low: '#ef4444', left: '#3b82f6', right: '#f97316', bilateral: '#8b5cf6', contralateral: '#06b6d4', ipsilateral: '#eab308' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function BodyMovementDashboard() {
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
          axios.get(`${API_URL}/api/body-movement/overview`),
          axios.get(`${API_URL}/api/body-movement/breakdown`),
          axios.get(`${API_URL}/api/body-movement/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Body Movement data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'movements', label: 'Movements' },
    { id: 'lateralization', label: 'Lateralization' },
    { id: 'pose', label: 'Pose Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const movementTypes = overview?.movement_types_detected || []
  const bodyRegions = overview?.body_region_involvement || []
  const pipelineStatus = overview?.pipeline_status || {}
  const keypointCoverage = overview?.keypoint_coverage || {}

  /* --- Breakdown data --- */
  const perPatient = breakdown?.per_patient_profiles || []
  const perRecording = breakdown?.per_recording_movements || []
  const temporalPatterns = breakdown?.movement_temporal_patterns || []
  const regionHeatmap = breakdown?.body_region_heatmap || []
  const lateralizationBySz = breakdown?.lateralization_by_seizure_type || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Body Movement Classifier Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Seizure-related body movement classification from video-EEG using pose estimation and motor semiology analysis
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
              <KPI label="Video-Capable" value={fmt(overview.video_capable_recordings)} color="#22c55e" />
              <KPI label="Movement Events" value={fmt(overview.total_movement_events)} color="#f97316" />
              <KPI label="Patients Analysed" value={fmt(overview.total_patients)} color="#3b82f6" />
              <KPI label="Avg Keypoint Conf." value={fmtPct(keypointCoverage.mean_confidence)} color="#8b5cf6" />
            </div>
          </Card>

          {/* Movement Types Distribution */}
          <Card title="Movement Types Detected" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={movementTypes} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="movement_type" type="category" tick={{ fontSize: 10 }} width={130} />
                <Tooltip />
                <Bar dataKey="count" name="Events" radius={[0, 4, 4, 0]}>
                  {movementTypes.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Body Region Involvement Radar */}
          <Card title="Body Region Involvement" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={bodyRegions} cx="50%" cy="50%" outerRadius={90}>
                <PolarGrid />
                <PolarAngleAxis dataKey="region" tick={{ fontSize: 9 }} />
                <PolarRadiusAxis tick={{ fontSize: 9 }} />
                <Radar name="Events" dataKey="event_count" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pipeline Status */}
          <Card title="Pipeline Status" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(pipelineStatus).map(([stage, info], i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{(info.stage || stage).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    <StatusBadge status={info.status || 'pending'} />
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>{info.tool || ''}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Keypoint Coverage Summary */}
          <Card title="Keypoint Coverage" span={1}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(keypointCoverage).map(([key, value], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155' }}>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                    <td style={{ padding: '6px 8px', color: '#475569', fontFamily: 'monospace', fontSize: 11, textAlign: 'right' }}>{typeof value === 'number' ? (value < 1 ? fmtPct(value) : fmt(value)) : fmt(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ======================== MOVEMENTS TAB ======================== */}
      {tab === 'movements' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Patient Profiles Table */}
          <Card title="Per-Patient Movement Profiles" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Video Recs</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Movement Events</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Dominant Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Lateralization</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Analyses</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatient.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.video_capable_recordings)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.movement_events)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.dominant_movement_type || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={p.lateralization || 'bilateral'} />
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.analyses_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Movement Temporal Patterns Chart */}
          <Card title="Movement Rate: Ictal vs Interictal (per min)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={temporalPatterns}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="ictal_movement_rate_per_min" fill={COLORS[4]} name="Ictal Rate" radius={[4, 4, 0, 0]} />
                <Bar dataKey="interictal_movement_rate_per_min" fill={COLORS[1]} name="Interictal Rate" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Recording Movement Table */}
          <Card title="Per-Recording Movements" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Duration</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Movements</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean Velocity</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean Angle</th>
                  </tr>
                </thead>
                <tbody>
                  {perRecording.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.recording_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.duration_min)} min</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.total_movement_events)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.velocity_stats?.mean_deg_per_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.joint_angle_stats?.mean_deg)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== LATERALIZATION TAB ======================== */}
      {tab === 'lateralization' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Lateralization Summary Pie */}
          <Card title="Lateralization Distribution" span={1}>
            {(() => {
              const latCounts = {}
              perPatient.forEach(p => {
                const l = p.lateralization || 'unknown'
                latCounts[l] = (latCounts[l] || 0) + 1
              })
              const latData = Object.entries(latCounts).map(([name, value]) => ({ name, value }))
              return (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={latData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                      {latData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>

          {/* Body Region Heatmap Bar */}
          <Card title="Body Region Movement Frequency" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={regionHeatmap} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="body_region" type="category" tick={{ fontSize: 10 }} width={120} />
                <Tooltip />
                <Bar dataKey="total_movement_events" name="Events" radius={[0, 4, 4, 0]}>
                  {regionHeatmap.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Lateralization by Seizure Type Table */}
          <Card title="Lateralization by Seizure Type" span={2}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Movement Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Left</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Right</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Bilateral</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Lateralizing Sign</th>
                  </tr>
                </thead>
                <tbody>
                  {lateralizationBySz.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.movement_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.left)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.right)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.bilateral)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 600 }}>{fmt(r.total)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={r.is_lateralising_sign ? 'high' : 'low'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== POSE ANALYSIS TAB ======================== */}
      {tab === 'pose' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Recording Pose Stats */}
          <Card title="Per-Recording Pose Estimation" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean Angle</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Max Angle</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean Velocity</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Max Velocity</th>
                  </tr>
                </thead>
                <tbody>
                  {perRecording.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{r.recording_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.joint_angle_stats?.mean_deg)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.joint_angle_stats?.max_deg)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.velocity_stats?.mean_deg_per_sec)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.velocity_stats?.max_deg_per_sec)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Joint Angle Distribution Chart */}
          <Card title="Amplitude & Velocity by Body Region" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={regionHeatmap}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="body_region" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="mean_amplitude_degrees" fill={COLORS[0]} name="Mean Amplitude (deg)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="mean_velocity_deg_per_sec" fill={COLORS[2]} name="Mean Velocity (deg/s)" radius={[4, 4, 0, 0]} />
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
