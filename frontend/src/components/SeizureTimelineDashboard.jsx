import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, ReferenceLine
} from 'recharts'

const API_URL = '/api'

const COLORS = {
  blue: '#1e88e5',
  red: '#ef4444',
  green: '#22c55e',
  amber: '#f59e0b',
  purple: '#7c4dff',
}

const PIE_COLORS = [
  '#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff',
  '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#64748b',
  '#8b5cf6', '#06b6d4', '#84cc16', '#e11d48', '#0ea5e9',
]

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

export default function SeizureTimelineDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedSeizure, setSelectedSeizure] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/seizure-timeline`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load seizure timeline')
      }
      setLoading(false)
    }
    load()
  }, [])

  // Derived data
  const kpis = useMemo(() => {
    if (!data?.timeline) return null
    const totalSeizures = data.total_seizures || data.timeline.length
    const totalSubjects = data.subjects?.length || 0
    const totalDuration = data.timeline.reduce((s, e) => s + (e.duration_sec || 0), 0)
    const avgDuration = totalSeizures > 0 ? (totalDuration / totalSeizures).toFixed(1) : 0
    const totalSpikes = data.timeline.reduce((s, e) => s + (e.spike_count || 0), 0)
    return { totalSeizures, totalSubjects, avgDuration, totalSpikes }
  }, [data])

  const subjectBarData = useMemo(() => {
    if (!data?.per_subject_summary) return []
    return data.per_subject_summary.map(s => ({
      subject: s.subject,
      seizures: s.total_seizures,
      meanDuration: Number((s.mean_duration_sec || 0).toFixed(1)),
    }))
  }, [data])

  const spikeChannelData = useMemo(() => {
    if (!data?.timeline) return []
    const counts = {}
    data.timeline.forEach(ev => {
      if (ev.channels_with_spikes) {
        ev.channels_with_spikes.forEach(ch => {
          counts[ch] = (counts[ch] || 0) + (ev.spike_count || 1)
        })
      }
    })
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 15)
  }, [data])

  const sortedTimeline = useMemo(() => {
    if (!data?.timeline) return []
    return [...data.timeline].sort((a, b) => {
      if (a.subject !== b.subject) return a.subject.localeCompare(b.subject)
      return (a.start_sec || 0) - (b.start_sec || 0)
    })
  }, [data])

  // Loading state
  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9889;</div>
      Loading seizure timeline from real CHB-MIT data...
      <div style={{ fontSize: 12, marginTop: 8, color: '#94a3b8' }}>
        (Analyzing seizure annotations and peri-onset EEG signals)
      </div>
    </div>
  )

  // Error state
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  // Unavailable state
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>
      Seizure timeline unavailable: {data?.error || 'No seizure data found'}
    </div>
  )

  const selectedEvent = selectedSeizure !== null ? sortedTimeline[selectedSeizure] : null
  const periChannels = selectedEvent?.peri_onset_eeg?.channels
    ? Object.keys(selectedEvent.peri_onset_eeg.channels).slice(0, 6)
    : []

  return (
    <div style={{ padding: 16, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ ...cardStyle, marginBottom: 16, display: 'flex', alignItems: 'center',
        justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
            &#9889; Seizure Timeline Dashboard &mdash; CHB-MIT
          </h2>
          <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
            Real seizure events from CHB-MIT scalp EEG corpus &middot;{' '}
            {kpis?.totalSeizures || 0} seizures across {kpis?.totalSubjects || 0} subjects
          </div>
        </div>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: 12, marginBottom: 16 }}>
        <KPITile label="Total Seizures" value={kpis?.totalSeizures} icon="&#9889;"
          color={COLORS.red} />
        <KPITile label="Total Subjects" value={kpis?.totalSubjects} icon="&#128101;"
          color={COLORS.blue} />
        <KPITile label="Avg Duration (sec)" value={kpis?.avgDuration} icon="&#9202;"
          color={COLORS.amber} />
        <KPITile label="Total Spikes Detected" value={kpis?.totalSpikes} icon="&#128200;"
          color={COLORS.green} />
      </div>

      {/* Per-Subject Bar Chart */}
      <div style={sectionHeadingStyle}>Per-Subject Seizure Summary</div>
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        {subjectBarData.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={subjectBarData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="subject" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11, fill: '#64748b' }}
                label={{ value: 'Count', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: '#64748b' }}
                label={{ value: 'Mean Duration (s)', angle: 90, position: 'insideRight', fontSize: 11, fill: '#94a3b8' }} />
              <Tooltip contentStyle={{ borderRadius: 8, fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar yAxisId="left" dataKey="seizures" name="Seizure Count"
                fill={COLORS.red} radius={[4, 4, 0, 0]} />
              <Bar yAxisId="right" dataKey="meanDuration" name="Mean Duration (s)"
                fill={COLORS.blue} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>
            No per-subject data available
          </div>
        )}
      </div>

      {/* Seizure Timeline Table */}
      <div style={sectionHeadingStyle}>Seizure Timeline (click a row to view peri-onset EEG)</div>
      <div style={{ ...cardStyle, marginBottom: 16, maxHeight: 360, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569' }}>
              <th style={thStyle}>Subject</th>
              <th style={thStyle}>File</th>
              <th style={thStyle}>Onset Clock</th>
              <th style={thStyle}>Duration (s)</th>
              <th style={thStyle}>Spikes</th>
              <th style={thStyle}>Peak Channel</th>
              <th style={{ ...thStyle, textAlign: 'right' }}>Peak Amp (&micro;V)</th>
            </tr>
          </thead>
          <tbody>
            {sortedTimeline.map((ev, idx) => (
              <tr key={idx}
                onClick={() => setSelectedSeizure(idx === selectedSeizure ? null : idx)}
                style={{
                  cursor: 'pointer',
                  background: idx === selectedSeizure ? '#eff6ff' : idx % 2 === 0 ? '#ffffff' : '#f8fafc',
                  borderBottom: '1px solid #f1f5f9',
                  transition: 'background 0.15s',
                }}>
                <td style={tdStyle}>{ev.subject}</td>
                <td style={tdStyle}>{ev.file}</td>
                <td style={tdStyle}>{ev.onset_clock || '--'}</td>
                <td style={tdStyle}>{ev.duration_sec ?? '--'}</td>
                <td style={tdStyle}>{ev.spike_count ?? '--'}</td>
                <td style={tdStyle}>
                  <span style={{ background: '#fef3c7', borderRadius: 4, padding: '1px 6px',
                    fontSize: 11, color: '#92400e' }}>
                    {ev.peak_channel || '--'}
                  </span>
                </td>
                <td style={{ ...tdStyle, textAlign: 'right' }}>
                  {ev.peak_amplitude_uv != null ? ev.peak_amplitude_uv.toFixed(1) : '--'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {sortedTimeline.length === 0 && (
          <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>
            No seizure events found
          </div>
        )}
      </div>

      {/* Peri-Onset EEG Viewer */}
      {selectedEvent && periChannels.length > 0 && (
        <>
          <div style={sectionHeadingStyle}>
            Peri-Onset EEG &mdash; {selectedEvent.subject} / {selectedEvent.file}
            {selectedEvent.onset_clock && (
              <span style={{ fontWeight: 400, color: '#64748b', marginLeft: 8, fontSize: 13 }}>
                onset at {selectedEvent.onset_clock}
              </span>
            )}
          </div>
          <div style={{ ...cardStyle, marginBottom: 16 }}>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
              Red dashed line = seizure onset (t=0). Showing top {periChannels.length} channel(s).
            </div>
            {periChannels.map((ch, i) => {
              const times = selectedEvent.peri_onset_eeg.times || []
              const values = selectedEvent.peri_onset_eeg.channels[ch] || []
              const chartData = times.map((t, j) => ({
                time: Number(t.toFixed(2)),
                value: values[j] ?? 0,
              }))
              return (
                <div key={ch} style={{ marginBottom: i < periChannels.length - 1 ? 12 : 0 }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', marginBottom: 2 }}>
                    {ch}
                  </div>
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={chartData} margin={{ top: 4, right: 12, left: 40, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#94a3b8' }}
                        label={{ value: 's', position: 'insideBottomRight', offset: -2,
                          fontSize: 9, fill: '#94a3b8' }}
                        type="number" domain={['dataMin', 'dataMax']}
                        tickCount={7} />
                      <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }}
                        label={{ value: '\u00B5V', angle: -90, position: 'insideLeft',
                          fontSize: 9, fill: '#94a3b8' }}
                        width={36} />
                      <Tooltip contentStyle={{ borderRadius: 6, fontSize: 11 }}
                        formatter={(v) => [`${v.toFixed(1)} \u00B5V`, ch]}
                        labelFormatter={(v) => `t = ${v} s`} />
                      <ReferenceLine x={0} stroke={COLORS.red} strokeDasharray="6 3"
                        strokeWidth={2} label={{ value: 'onset', position: 'top',
                          fill: COLORS.red, fontSize: 9 }} />
                      <Line type="monotone" dataKey="value" stroke={PIE_COLORS[i % PIE_COLORS.length]}
                        strokeWidth={1.2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )
            })}
          </div>
        </>
      )}

      {selectedSeizure !== null && periChannels.length === 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, color: '#94a3b8', textAlign: 'center',
          padding: 20 }}>
          No peri-onset EEG data available for this seizure event.
        </div>
      )}

      {/* Spike Distribution Pie Chart */}
      <div style={sectionHeadingStyle}>Spike Distribution by Channel</div>
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        {spikeChannelData.length > 0 ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 24 }}>
            <div style={{ flex: '1 1 320px', minWidth: 280 }}>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={spikeChannelData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" outerRadius={110} innerRadius={50}
                    paddingAngle={2} label={({ name, percent }) =>
                      `${name} (${(percent * 100).toFixed(0)}%)`}
                    labelLine={{ stroke: '#94a3b8', strokeWidth: 0.5 }}
                    style={{ fontSize: 10 }}>
                    {spikeChannelData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 8, fontSize: 12 }}
                    formatter={(v) => [`${v} spikes`]} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: '1 1 220px', minWidth: 200 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#475569' }}>
                    <th style={{ ...thStyle, padding: '4px 8px' }}>Channel</th>
                    <th style={{ ...thStyle, padding: '4px 8px', textAlign: 'right' }}>Spikes</th>
                  </tr>
                </thead>
                <tbody>
                  {spikeChannelData.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '3px 8px' }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8,
                          borderRadius: '50%', background: PIE_COLORS[i % PIE_COLORS.length],
                          marginRight: 6 }} />
                        {d.name}
                      </td>
                      <td style={{ padding: '3px 8px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                        {d.value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>
            No spike channel data available
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ textAlign: 'center', fontSize: 11, color: '#94a3b8', marginTop: 8, marginBottom: 16 }}>
        Data source: CHB-MIT Scalp EEG Database &middot; Real seizure annotations
      </div>
    </div>
  )
}

/* --- Helper Components --- */

function KPITile({ label, value, icon, color }) {
  return (
    <div style={{
      ...cardStyle,
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      borderLeft: `4px solid ${color}`,
    }}>
      <div style={{ fontSize: 24 }}>{icon}</div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b',
          fontVariantNumeric: 'tabular-nums' }}>
          {value ?? '--'}
        </div>
        <div style={{ fontSize: 11, color: '#64748b' }}>{label}</div>
      </div>
    </div>
  )
}

/* --- Table cell styles --- */

const thStyle = {
  textAlign: 'left',
  padding: '6px 10px',
  fontSize: 11,
  fontWeight: 600,
  whiteSpace: 'nowrap',
}

const tdStyle = {
  padding: '5px 10px',
  whiteSpace: 'nowrap',
}
