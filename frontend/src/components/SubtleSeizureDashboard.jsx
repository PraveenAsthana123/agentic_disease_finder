import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
]

export default function SubtleSeizureDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filterEventType, setFilterEventType] = useState('all')
  const [filterConfidence, setFilterConfidence] = useState('all')

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/subtle-seizure/overview',
      events: '/api/subtle-seizure/breakdown',
      channelmap: '/api/subtle-seizure/breakdown',
      hourly: '/api/subtle-seizure/overview',
      definitions: '/api/subtle-seizure/definitions',
    }
    const url = endpoints[tab]
    if (!url) return
    axios.get(`${API}${url}`)
      .then(r => {
        if (tab === 'overview' || tab === 'hourly') setOverview(r.data)
        else if (tab === 'events' || tab === 'channelmap') setBreakdown(r.data)
        else setDefinitions(r.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'events', label: 'Events' },
    { id: 'channelmap', label: 'Channel Map' },
    { id: 'hourly', label: 'Hourly Pattern' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const sty = {
    card: { background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 },
    kpi: { background: '#334155', borderRadius: 8, padding: 16, textAlign: 'center', flex: 1 },
    kpiVal: { fontSize: 28, fontWeight: 700, color: '#818cf8' },
    kpiLabel: { fontSize: 12, color: '#94a3b8', marginTop: 4 },
    grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 },
    tab: (active) => ({
      padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
      background: active ? '#6366f1' : '#334155', color: active ? '#fff' : '#94a3b8', fontSize: 13
    }),
    badge: (level) => ({
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
      background: level === 'high' ? '#10b98133' : level === 'medium' ? '#f59e0b33' : '#ef444433',
      color: level === 'high' ? '#10b981' : level === 'medium' ? '#f59e0b' : '#ef4444'
    }),
    boolBadge: (val) => ({
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
      background: val ? '#10b98133' : '#ef444433',
      color: val ? '#10b981' : '#ef4444'
    }),
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', borderBottom: '1px solid #334155', color: '#94a3b8', fontWeight: 600 },
    td: { padding: '8px 10px', borderBottom: '1px solid #1e293b' },
    select: {
      background: '#334155', color: '#e2e8f0', border: '1px solid #475569',
      borderRadius: 6, padding: '6px 12px', fontSize: 13, cursor: 'pointer'
    },
  }

  const confidenceLevel = (val) => {
    if (val == null) return 'low'
    if (val >= 0.75) return 'high'
    if (val >= 0.5) return 'medium'
    return 'low'
  }

  // ── Tab: Overview ────────────────────────────────────────────────
  const renderOverview = () => {
    if (!overview) return null
    const d = overview
    const etDist = d.event_type_distribution || []
    const confDist = d.confidence_distribution || []
    const reviewDist = d.review_status_distribution || []
    const hourly = d.hourly_detection_pattern || []

    return (<>
      <div style={sty.grid}>
        {[
          [d.total_recordings_scanned, 'Total Recordings Scanned'],
          [d.total_subtle_events_detected, 'Subtle Events Detected'],
          [d.sensitivity_rate != null ? `${(d.sensitivity_rate * 100).toFixed(1)}%` : '—', 'Sensitivity Rate'],
          [d.specificity != null ? `${(d.specificity * 100).toFixed(1)}%` : '—', 'False Positive Rate'],
          [d.avg_event_duration_sec != null ? `${d.avg_event_duration_sec}s` : '—', 'Avg Event Duration'],
          [d.fatigue_adjusted_detection_gain != null ? `${(d.fatigue_adjusted_detection_gain * 100).toFixed(1)}%` : '—', 'Fatigue-Adjusted Gain'],
        ].map(([val, label], i) => (
          <div key={i} style={sty.kpi}>
            <div style={sty.kpiVal}>{val ?? '—'}</div>
            <div style={sty.kpiLabel}>{label}</div>
          </div>
        ))}
      </div>

      {etDist.length > 0 && (
        <div style={sty.card}>
          <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Event Type Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={etDist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="event_type" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="count" fill="#6366f1" name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {confDist.length > 0 && (
          <div style={sty.card}>
            <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Confidence Distribution</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={confDist}
                  cx="50%" cy="50%" outerRadius={75} dataKey="count" nameKey="bucket" label
                >
                  {confDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {reviewDist.length > 0 && (
          <div style={sty.card}>
            <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Review Status Distribution</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={reviewDist}
                  cx="50%" cy="50%" outerRadius={75} dataKey="count" nameKey="status" label
                >
                  {reviewDist.map((_, i) => <Cell key={i} fill={COLORS[(i + 3) % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {hourly.length > 0 && (
        <div style={sty.card}>
          <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Hourly Detection Pattern (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={hourly}>
              <defs>
                <linearGradient id="hourlyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Hour (0–23)', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -2 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Area type="monotone" dataKey="detections" stroke="#6366f1" fill="url(#hourlyGrad)" name="Detections" />
            </AreaChart>
          </ResponsiveContainer>
          <p style={{ color: '#64748b', fontSize: 11, marginTop: 6 }}>
            Late-night hours (2–6 AM) highlighted — peak fatigue period for human reviewers.
          </p>
        </div>
      )}
    </>)
  }

  // ── Tab: Events ──────────────────────────────────────────────────
  const renderEvents = () => {
    if (!breakdown) return null
    const events = breakdown.events || []

    const eventTypes = ['all', ...Array.from(new Set(events.map(e => e.event_type).filter(Boolean)))]
    const confidenceLevels = ['all', 'high', 'medium', 'low']

    const filtered = events.filter(e => {
      const typeMatch = filterEventType === 'all' || e.event_type === filterEventType
      const confMatch = filterConfidence === 'all' || confidenceLevel(e.confidence) === filterConfidence
      return typeMatch && confMatch
    })

    return (<>
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div>
          <label style={{ color: '#94a3b8', fontSize: 12, marginRight: 6 }}>Event Type:</label>
          <select style={sty.select} value={filterEventType} onChange={e => setFilterEventType(e.target.value)}>
            {eventTypes.map(t => <option key={t} value={t}>{t === 'all' ? 'All Types' : t}</option>)}
          </select>
        </div>
        <div>
          <label style={{ color: '#94a3b8', fontSize: 12, marginRight: 6 }}>Confidence:</label>
          <select style={sty.select} value={filterConfidence} onChange={e => setFilterConfidence(e.target.value)}>
            {confidenceLevels.map(c => <option key={c} value={c}>{c === 'all' ? 'All Levels' : c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
          </select>
        </div>
        <span style={{ color: '#64748b', fontSize: 12 }}>{filtered.length} of {events.length} events</span>
      </div>

      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Subtle Seizure Events</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={sty.table}>
            <thead>
              <tr>
                <th style={sty.th}>Patient</th>
                <th style={sty.th}>Event Type</th>
                <th style={sty.th}>Onset Time</th>
                <th style={sty.th}>Duration (s)</th>
                <th style={sty.th}>Amplitude (µV)</th>
                <th style={sty.th}>Confidence</th>
                <th style={sty.th}>Channels</th>
                <th style={sty.th}>AI Flagged</th>
                <th style={sty.th}>Neurologist</th>
                <th style={sty.th}>Lateralization</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, i) => {
                const level = confidenceLevel(e.confidence)
                return (
                  <tr key={i}>
                    <td style={sty.td}>{e.patient_id != null ? `P${e.patient_id}` : '—'}</td>
                    <td style={sty.td}>{e.event_type || '—'}</td>
                    <td style={sty.td}>{e.onset_time || '—'}</td>
                    <td style={sty.td}>{e.duration_sec ?? '—'}</td>
                    <td style={sty.td}>{e.amplitude_uv != null ? e.amplitude_uv.toFixed(1) : '—'}</td>
                    <td style={sty.td}>
                      <span style={sty.badge(level)}>
                        {e.confidence != null ? `${(e.confidence * 100).toFixed(0)}%` : '—'}
                      </span>
                    </td>
                    <td style={sty.td}>{Array.isArray(e.channels_involved) ? e.channels_involved.join(', ') : (e.channels_involved || '—')}</td>
                    <td style={sty.td}>
                      <span style={sty.boolBadge(e.ai_flagged)}>{e.ai_flagged ? 'Yes' : 'No'}</span>
                    </td>
                    <td style={sty.td}>{e.neurologist_verdict || '—'}</td>
                    <td style={sty.td}>{e.lateralization || '—'}</td>
                  </tr>
                )
              })}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={10} style={{ ...sty.td, textAlign: 'center', color: '#64748b' }}>No events match the selected filters.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>)
  }

  // ── Tab: Channel Map ─────────────────────────────────────────────
  const renderChannelMap = () => {
    if (!breakdown) return null
    const channelInvolvement = breakdown.channel_involvement || []

    const chartData = channelInvolvement.map(c => ({
      channel: c.channel,
      count: c.count ?? c.involvement_count ?? 0
    })).sort((a, b) => b.count - a.count)

    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>Channel Involvement in Subtle Events</h3>
        <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>
          Channels most frequently involved in subtle seizure events across all recordings.
        </p>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis type="category" dataKey="channel" tick={{ fill: '#94a3b8', fontSize: 11 }} width={80} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Bar dataKey="count" name="Event Count">
                {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: '#64748b', fontSize: 13 }}>No channel involvement data available.</p>
        )}
      </div>

      <div style={sty.grid}>
        {chartData.slice(0, 6).map((c, i) => (
          <div key={i} style={sty.kpi}>
            <div style={{ ...sty.kpiVal, color: COLORS[i % COLORS.length] }}>{c.count}</div>
            <div style={sty.kpiLabel}>{c.channel}</div>
          </div>
        ))}
      </div>
    </>)
  }

  // ── Tab: Hourly Pattern ──────────────────────────────────────────
  const renderHourly = () => {
    if (!overview) return null
    const hourly = overview.hourly_detection_pattern || []

    const lateNightHours = new Set([2, 3, 4, 5, 6])
    const enriched = hourly.map(h => ({
      ...h,
      lateNight: lateNightHours.has(h.hour) ? h.detections : 0,
      regular: lateNightHours.has(h.hour) ? 0 : h.detections,
    }))

    const lateNightTotal = hourly.filter(h => lateNightHours.has(h.hour)).reduce((s, h) => s + (h.detections || 0), 0)
    const totalDetections = hourly.reduce((s, h) => s + (h.detections || 0), 0)
    const lateNightPct = totalDetections > 0 ? ((lateNightTotal / totalDetections) * 100).toFixed(1) : '—'

    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0', marginBottom: 6 }}>24-Hour Detection Timeline</h3>
        <p style={{ color: '#f59e0b', fontSize: 13, marginBottom: 12 }}>
          Detection gain is highest during hours 2–6 AM when reviewer fatigue peaks.
        </p>
        {hourly.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={enriched}>
              <defs>
                <linearGradient id="regularGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.05} />
                </linearGradient>
                <linearGradient id="lateNightGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Hour of Day', fill: '#64748b', fontSize: 11, position: 'insideBottom', offset: -2 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
              <Area type="monotone" dataKey="regular" stroke="#6366f1" fill="url(#regularGrad)" name="Standard Hours" stackId="1" />
              <Area type="monotone" dataKey="lateNight" stroke="#f59e0b" fill="url(#lateNightGrad)" name="Fatigue Hours (2–6 AM)" stackId="1" />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: '#64748b', fontSize: 13 }}>No hourly data available.</p>
        )}
      </div>

      <div style={sty.grid}>
        {[
          [totalDetections, 'Total Detections'],
          [lateNightTotal, 'Late-Night (2–6 AM)'],
          [`${lateNightPct}%`, 'Late-Night Share'],
          [overview.fatigue_adjusted_detection_gain != null ? `${(overview.fatigue_adjusted_detection_gain * 100).toFixed(1)}%` : '—', 'Fatigue-Adjusted Gain'],
        ].map(([val, label], i) => (
          <div key={i} style={sty.kpi}>
            <div style={{ ...sty.kpiVal, color: i === 1 || i === 2 ? '#f59e0b' : '#818cf8' }}>{val ?? '—'}</div>
            <div style={sty.kpiLabel}>{label}</div>
          </div>
        ))}
      </div>

      <div style={sty.card}>
        <h4 style={{ color: '#818cf8', marginBottom: 8 }}>About Fatigue-Adjusted Detection</h4>
        <p style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.6 }}>
          Human EEG reviewers experience significant cognitive fatigue during overnight sessions,
          particularly between 2–6 AM. AI-assisted subtle seizure detection compensates by maintaining
          consistent sensitivity across all hours, yielding a higher effective detection gain during
          peak fatigue windows. The fatigue-adjusted detection gain metric quantifies this improvement
          over baseline human-only review.
        </p>
      </div>
    </>)
  }

  // ── Tab: Definitions ────────────────────────────────────────────
  const renderDefinitions = () => {
    if (!definitions) return null

    return (<>
      <div style={sty.card}>
        <h3 style={{ color: '#e2e8f0' }}>{definitions.title || 'Subtle Seizure Detection'}</h3>
        {definitions.pipeline_step && (
          <p style={{ color: '#94a3b8', fontSize: 13 }}>Pipeline Step {definitions.pipeline_step} — {definitions.purpose || ''}</p>
        )}
        {definitions.description && (
          <p style={{ color: '#cbd5e1', fontSize: 13, marginTop: 8 }}>{definitions.description}</p>
        )}
      </div>

      {definitions.event_types && (
        <div style={sty.card}>
          <h4 style={{ color: '#818cf8', marginBottom: 8 }}>Event Types</h4>
          {Object.entries(definitions.event_types).map(([key, val]) => (
            <div key={key} style={{ padding: '6px 0', borderBottom: '1px solid #334155' }}>
              <span style={{ color: '#10b981', fontWeight: 600, fontSize: 13 }}>{key.replace(/_/g, ' ')}: </span>
              <span style={{ color: '#cbd5e1', fontSize: 13 }}>{typeof val === 'string' ? val : JSON.stringify(val)}</span>
            </div>
          ))}
        </div>
      )}

      {definitions.detection_criteria && (
        <div style={sty.card}>
          <h4 style={{ color: '#818cf8', marginBottom: 8 }}>Detection Criteria</h4>
          {(Array.isArray(definitions.detection_criteria)
            ? definitions.detection_criteria
            : Object.entries(definitions.detection_criteria).map(([k, v]) => `${k}: ${v}`)
          ).map((c, i) => (
            <div key={i} style={{ padding: '4px 0', color: '#cbd5e1', fontSize: 13 }}>{c}</div>
          ))}
        </div>
      )}

      {definitions.clinical_significance && (
        <div style={sty.card}>
          <h4 style={{ color: '#818cf8', marginBottom: 8 }}>Clinical Significance</h4>
          <p style={{ color: '#cbd5e1', fontSize: 13 }}>{definitions.clinical_significance}</p>
        </div>
      )}

      {definitions.references && (
        <div style={sty.card}>
          <h4 style={{ color: '#818cf8', marginBottom: 6 }}>References</h4>
          {(definitions.references || []).map((r, i) => (
            <div key={i} style={{ padding: '4px 0', color: '#94a3b8', fontSize: 12 }}>{r}</div>
          ))}
        </div>
      )}

      {!definitions.event_types && !definitions.detection_criteria && !definitions.clinical_significance && (
        <div style={sty.card}>
          <pre style={{ color: '#94a3b8', fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {JSON.stringify(definitions, null, 2)}
          </pre>
        </div>
      )}
    </>)
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ color: '#e2e8f0', marginBottom: 4 }}>Subtle Seizure Detection</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
        AI-assisted detection of low-amplitude, brief, and visually ambiguous seizure patterns missed by human reviewers
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} style={sty.tab(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading...</div>}
      {error && <div style={{ color: '#ef4444', padding: 20 }}>Error: {error}</div>}
      {!loading && !error && tab === 'overview' && renderOverview()}
      {!loading && !error && tab === 'events' && renderEvents()}
      {!loading && !error && tab === 'channelmap' && renderChannelMap()}
      {!loading && !error && tab === 'hourly' && renderHourly()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
