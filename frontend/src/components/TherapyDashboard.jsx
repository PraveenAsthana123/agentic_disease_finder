import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '—')

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

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

const statusColor = (s) => {
  const v = (s || '').toLowerCase()
  if (v === 'active') return '#22c55e'
  if (v === 'completed') return '#1e88e5'
  if (v === 'planned') return '#f59e0b'
  return '#94a3b8'
}

const typeColor = (t) => {
  const v = (t || '').toLowerCase()
  if (v === 'physio' || v === 'physiotherapy') return '#1e88e5'
  if (v === 'cognitive') return '#7c4dff'
  if (v === 'sleep') return '#6366f1'
  if (v === 'safety') return '#ef4444'
  if (v === 'mindfulness' || v === 'meditation') return '#14b8a6'
  if (v === 'breathing') return '#22c55e'
  if (v === 'yoga_nidra') return '#ec4899'
  if (v === 'progressive_relaxation') return '#f59e0b'
  return '#94a3b8'
}

export default function TherapyDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [defs, setDefs] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/therapy`),
      axios.get(`${API_URL}/therapy/definitions`),
    ])
      .then(([overviewRes, defsRes]) => {
        setData(overviewRes.data)
        setDefs(defsRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Therapy Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}

  // Rehab programs — flatten per patient
  const rehabData = data.rehab_programs || {}
  const rehabPrograms = (rehabData.patients || []).flatMap(p =>
    (p.programs || []).map(prog => ({ ...prog, patient_id: p.patient_id }))
  )

  // Exercise plans — flatten
  const exerciseData = data.exercise_plans || {}
  const exercises = (exerciseData.patients || []).flatMap(p =>
    (p.exercises || []).map(ex => ({ ...ex, patient_id: p.patient_id }))
  )

  // Meditation programs — flatten
  const meditationData = data.meditation_programs || {}
  const meditations = (meditationData.patients || []).flatMap(p =>
    (p.programs || []).map(prog => ({ ...prog, patient_id: p.patient_id }))
  )

  // Physio protocols — flatten
  const physioData = data.physio_protocols || {}
  const physioProtos = (physioData.patients || []).flatMap(p =>
    (p.protocols || []).map(proto => ({ ...proto, patient_id: p.patient_id }))
  )

  // --- Charts ---

  // Program type distribution (pie)
  const typeCounts = {}
  rehabPrograms.forEach(p => {
    const t = p.type || 'other'
    typeCounts[t] = (typeCounts[t] || 0) + 1
  })
  const typeDistrib = Object.entries(typeCounts)
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))

  // Status distribution (pie)
  const statusCounts = { active: 0, completed: 0, planned: 0 }
  rehabPrograms.forEach(p => {
    const s = (p.status || '').toLowerCase()
    if (s in statusCounts) statusCounts[s]++
  })
  const statusDistrib = Object.entries(statusCounts)
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
    .filter(d => d.value > 0)

  // Exercise categories bar chart
  const exCatCounts = {}
  exercises.forEach(e => {
    const c = e.category || 'other'
    exCatCounts[c] = (exCatCounts[c] || 0) + 1
  })
  const exerciseChart = Object.entries(exCatCounts)
    .map(([name, count]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), count }))
    .sort((a, b) => b.count - a.count)

  // Meditation type bar chart
  const medTypeCounts = {}
  meditations.forEach(m => {
    const t = (m.type || 'other').replace(/_/g, ' ')
    medTypeCounts[t] = (medTypeCounts[t] || 0) + 1
  })
  const meditationChart = Object.entries(medTypeCounts)
    .map(([name, count]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), count }))

  // KPI tiles
  const kpis = [
    { label: 'Patients in Therapy', value: summary.total_patients_with_therapy ?? summary.total_patients, color: COLORS[0] },
    { label: 'Rehab Programs', value: summary.total_rehab_programs ?? summary.active_programs, color: COLORS[2] },
    { label: 'Meditation Programs', value: summary.total_meditation_programs ?? summary.total_meditations, color: COLORS[7] },
    { label: 'Physio Protocols', value: summary.total_physio_protocols ?? summary.total_physio, color: COLORS[3] },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
            Meditation / Physio / Therapy
          </h1>
          <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
            Patient Portal &middot; Rehab programs, exercises, meditation &amp; physiotherapy
          </p>
        </div>
        {defs && (
          <button
            onClick={() => setShowDefs(!showDefs)}
            style={{ padding: '6px 14px', borderRadius: 8, border: '1px solid #e2e8f0', background: showDefs ? '#1e88e5' : '#fff', color: showDefs ? '#fff' : '#64748b', cursor: 'pointer', fontSize: 13 }}
          >
            {showDefs ? 'Hide' : 'Show'} Definitions
          </button>
        )}
      </div>

      {/* Definitions panel */}
      {showDefs && defs && (
        <div style={{ ...cardStyle, marginBottom: 20, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Metric Definitions</div>
          {Object.entries(defs).map(([k, v]) => (
            <div key={k} style={{ marginBottom: 6, fontSize: 13 }}>
              <strong>{k}:</strong> {v}
            </div>
          ))}
        </div>
      )}

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            ...cardStyle,
            flex: '1 1 160px',
            borderLeft: `4px solid ${kpi.color}`,
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Charts row: type distribution + status */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {typeDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Program Types</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={typeDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {typeDistrib.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
        {statusDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Program Status</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={statusDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {statusDistrib.map((d, i) => <Cell key={i} fill={statusColor(d.name)} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Exercise categories + meditation types bar charts */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {exerciseChart.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Exercise Categories</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={exerciseChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {meditationChart.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Meditation Types</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={meditationChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={12} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill={COLORS[7]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Rehabilitation Programs Table */}
      <h2 style={sectionHeadingStyle}>Rehabilitation Programs</h2>
      {rehabPrograms.length > 0 ? (
        <div style={{ ...cardStyle, overflowX: 'auto', marginBottom: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Patient</th>
                <th style={{ padding: '6px 10px' }}>Program</th>
                <th style={{ padding: '6px 10px' }}>Type</th>
                <th style={{ padding: '6px 10px' }}>Frequency</th>
                <th style={{ padding: '6px 10px' }}>Duration</th>
                <th style={{ padding: '6px 10px' }}>Status</th>
                <th style={{ padding: '6px 10px' }}>Progress</th>
              </tr>
            </thead>
            <tbody>
              {rehabPrograms.map((prog, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{prog.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{prog.program_name || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(typeColor(prog.type))}>{prog.type || '—'}</span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>{prog.frequency || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{prog.duration_weeks ? `${prog.duration_weeks} wk` : '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(statusColor(prog.status))}>{prog.status || '—'}</span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    {prog.progress_pct != null ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ flex: 1, height: 6, background: '#e2e8f0', borderRadius: 3, maxWidth: 80 }}>
                          <div style={{ height: '100%', width: `${prog.progress_pct}%`, background: statusColor(prog.status), borderRadius: 3 }} />
                        </div>
                        <span style={{ fontSize: 11, color: '#64748b' }}>{prog.progress_pct}%</span>
                      </div>
                    ) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div style={cardStyle}>No rehabilitation programs found.</div>}

      {/* Exercise Plans Table */}
      <h2 style={sectionHeadingStyle}>Exercise Plans</h2>
      {exercises.length > 0 ? (
        <div style={{ ...cardStyle, overflowX: 'auto', marginBottom: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Patient</th>
                <th style={{ padding: '6px 10px' }}>Exercise</th>
                <th style={{ padding: '6px 10px' }}>Category</th>
                <th style={{ padding: '6px 10px' }}>Duration</th>
                <th style={{ padding: '6px 10px' }}>Frequency</th>
                <th style={{ padding: '6px 10px' }}>Precautions</th>
              </tr>
            </thead>
            <tbody>
              {exercises.map((ex, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{ex.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{ex.exercise_name || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(COLORS[['aerobic','strength','flexibility','balance'].indexOf(ex.category) % COLORS.length] || COLORS[0])}>
                      {ex.category || '—'}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>{ex.duration_min ? `${ex.duration_min} min` : '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{ex.frequency || '—'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>
                    {ex.contraindicated ? (
                      <span style={badgeStyle('#ef4444')}>Contraindicated</span>
                    ) : (
                      (ex.precautions || []).slice(0, 2).join('; ') || 'None'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div style={cardStyle}>No exercise plans found.</div>}

      {/* Meditation Programs Table */}
      <h2 style={sectionHeadingStyle}>Meditation &amp; Mindfulness Programs</h2>
      {meditations.length > 0 ? (
        <div style={{ ...cardStyle, overflowX: 'auto', marginBottom: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Patient</th>
                <th style={{ padding: '6px 10px' }}>Program</th>
                <th style={{ padding: '6px 10px' }}>Type</th>
                <th style={{ padding: '6px 10px' }}>Session</th>
                <th style={{ padding: '6px 10px' }}>Frequency</th>
                <th style={{ padding: '6px 10px' }}>Target</th>
                <th style={{ padding: '6px 10px' }}>Evidence</th>
              </tr>
            </thead>
            <tbody>
              {meditations.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{m.program_name || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(typeColor(m.type))}>{(m.type || '—').replace(/_/g, ' ')}</span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>{m.session_duration_min ? `${m.session_duration_min} min` : '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{m.frequency || '—'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{m.target_condition || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(m.evidence_level === 'strong' ? '#22c55e' : m.evidence_level === 'moderate' ? '#f59e0b' : '#94a3b8')}>
                      {m.evidence_level || '—'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div style={cardStyle}>No meditation programs found.</div>}

      {/* Physiotherapy Protocols Table */}
      <h2 style={sectionHeadingStyle}>Physiotherapy Protocols</h2>
      {physioProtos.length > 0 ? (
        <div style={{ ...cardStyle, overflowX: 'auto', marginBottom: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Patient</th>
                <th style={{ padding: '6px 10px' }}>Protocol</th>
                <th style={{ padding: '6px 10px' }}>Target Area</th>
                <th style={{ padding: '6px 10px' }}>Sessions/wk</th>
                <th style={{ padding: '6px 10px' }}>Duration</th>
                <th style={{ padding: '6px 10px' }}>Exercises</th>
                <th style={{ padding: '6px 10px' }}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {physioProtos.map((proto, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{proto.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{proto.protocol_name || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <span style={badgeStyle(COLORS[['upper_limb','lower_limb','balance','swallowing','speech'].indexOf(proto.target_area) % COLORS.length] || COLORS[0])}>
                      {(proto.target_area || '—').replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td style={{ padding: '6px 10px' }}>{proto.sessions_per_week || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{proto.duration_weeks ? `${proto.duration_weeks} wk` : '—'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>
                    {(proto.exercises || []).slice(0, 3).join(', ') || '—'}
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b', maxWidth: 200 }}>
                    {proto.therapist_notes || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div style={cardStyle}>No physiotherapy protocols found.</div>}

      {/* Footer */}
      <div style={{ marginTop: 24, textAlign: 'center', color: '#94a3b8', fontSize: 12 }}>
        Therapy programs derived from real patient assessments in clinical.db &middot; Evidence-based epilepsy rehabilitation
      </div>
    </div>
  )
}
