import React, { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const QUALITY_COLORS = { Good: '#22c55e', Fair: '#f59e0b', Poor: '#ef4444' }
const RESPONSE_COLORS = {
  'Well controlled': '#22c55e',
  'Partially controlled': '#f59e0b',
  Refractory: '#ef4444',
}
const PREDICTION_COLORS = {
  normal: '#22c55e', epilepsy: '#ef4444', seizure: '#ef4444',
  abnormal: '#f59e0b', artifact: '#94a3b8', focal: '#a855f7',
  generalized: '#3b82f6',
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 12, padding: 20,
      border: '1px solid #334155',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#e2e8f0' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '16px 12px',
      border: '1px solid #334155', borderLeft: `4px solid ${color || '#3b82f6'}`,
      textAlign: 'center', flex: 1, minWidth: 130
    }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#e2e8f0' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function SeverityBadge({ severity }) {
  const color = QUALITY_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color
    }}>{severity || 'Unknown'}</span>
  )
}

function ResponseBadge({ grade }) {
  const color = RESPONSE_COLORS[grade] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color
    }}>{grade || 'Unknown'}</span>
  )
}

function HBar({ items, maxVal }) {
  const mx = maxVal || Math.max(...items.map(i => i.value), 1)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {items.map((item, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 120, fontSize: 12, color: '#94a3b8', textAlign: 'right', flexShrink: 0 }}>{item.label}</div>
          <div style={{ flex: 1, background: '#0f172a', borderRadius: 4, height: 20, position: 'relative' }}>
            <div style={{
              width: `${Math.max((item.value / mx) * 100, 2)}%`,
              height: '100%', borderRadius: 4,
              background: item.color || '#3b82f6',
              transition: 'width 0.3s ease'
            }} />
          </div>
          <div style={{ width: 40, fontSize: 12, color: '#e2e8f0', textAlign: 'right' }}>{fmt(item.value)}</div>
        </div>
      ))}
    </div>
  )
}

export default function NeurologistDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedPatient, setExpandedPatient] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/neurologist/overview`),
          axios.get(`${API_URL}/api/neurologist/breakdown`),
          axios.get(`${API_URL}/api/neurologist/definitions`),
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 60, textAlign: 'center', color: '#94a3b8', background: '#0f172a', minHeight: '100vh' }}>
      <div style={{ fontSize: 24, marginBottom: 12 }}>Loading...</div>
      <div style={{
        width: 40, height: 40, border: '4px solid #334155', borderTop: '4px solid #3b82f6',
        borderRadius: '50%', margin: '0 auto',
        animation: 'spin 1s linear infinite'
      }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
  if (error) return (
    <div style={{ padding: 40, color: '#ef4444', background: '#0f172a', minHeight: '100vh' }}>
      <h2 style={{ color: '#ef4444' }}>Error loading Neurologist Dashboard</h2>
      <p>{error}</p>
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'eeg_reads', label: 'EEG Reads' },
    { id: 'hitl_review', label: 'HITL Review' },
    { id: 'medications', label: 'Medications & Seizures' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const predDist = overview?.prediction_distribution || []
  const sevDist = overview?.severity_distribution || []
  const confHist = overview?.confidence_histogram || []
  const turnaroundHist = overview?.turnaround_histogram || []
  const weeklyVol = overview?.weekly_volume || []
  const overrideRate = overview?.override_rate

  const patients = breakdown?.patients || []
  const seizureClass = breakdown?.seizure_classification || []
  const medSummary = breakdown?.medication_summary || []

  const thStyle = {
    textAlign: 'left', padding: '8px 10px', fontSize: 12,
    color: '#94a3b8', borderBottom: '2px solid #334155', fontWeight: 600
  }
  const tdStyle = {
    padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #1e293b', color: '#e2e8f0'
  }

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', padding: '24px 32px', color: '#e2e8f0' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Neurologist Dashboard</h1>
      <p style={{ color: '#94a3b8', marginTop: 0, marginBottom: 20, fontSize: 14 }}>
        EEG interpretation, seizure classification, and treatment monitoring
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #334155', marginBottom: 24 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => { setTab(t.id); setExpandedPatient(null) }}
            style={{
              padding: '10px 20px', fontSize: 13, fontWeight: 600, cursor: 'pointer',
              background: 'transparent', border: 'none',
              color: tab === t.id ? '#3b82f6' : '#94a3b8',
              borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
              transition: 'all 0.2s'
            }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI Row */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
            <KPI label="Total Patients" value={fmt(kpis.total_patients)} color="#3b82f6" />
            <KPI label="Total Analyses" value={fmt(kpis.total_analyses)} color="#a855f7" />
            <KPI label="Seizure Positive" value={fmt(kpis.seizure_positive)} sub={kpis.seizure_rate != null ? `${fmt(kpis.seizure_rate)}%` : undefined} color="#ef4444" />
            <KPI label="Avg Confidence" value={kpis.avg_confidence != null ? `${fmt(kpis.avg_confidence)}%` : '--'} color="#22c55e" />
            <KPI label="Pending Reviews" value={fmt(kpis.pending_reviews)} color="#f59e0b" />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            {/* Prediction Distribution */}
            <Card title="Prediction Distribution">
              <HBar items={predDist.map(d => ({
                label: d.prediction || d.label || d.name,
                value: d.count || d.value || 0,
                color: PREDICTION_COLORS[(d.prediction || d.label || d.name || '').toLowerCase()] || '#3b82f6'
              }))} />
            </Card>

            {/* Severity Distribution */}
            <Card title="Signal Quality Distribution">
              <HBar items={sevDist.map(d => ({
                label: d.quality || d.severity || d.label || d.name,
                value: d.count || d.value || 0,
                color: QUALITY_COLORS[d.quality || d.severity || d.label] || '#3b82f6'
              }))} />
            </Card>

            {/* Confidence Histogram */}
            <Card title="Confidence Histogram">
              {confHist.length > 0 ? (
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 120 }}>
                  {confHist.map((bin, i) => {
                    const mx = Math.max(...confHist.map(b => b.count || b.value || 0), 1)
                    const val = bin.count || bin.value || 0
                    return (
                      <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>{val}</div>
                        <div style={{
                          width: '100%', maxWidth: 36,
                          height: `${Math.max((val / mx) * 100, 4)}%`,
                          background: '#3b82f6', borderRadius: '4px 4px 0 0',
                          minHeight: 4
                        }} />
                        <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>{bin.bin || bin.label || ''}</div>
                      </div>
                    )
                  })}
                </div>
              ) : <div style={{ color: '#64748b', fontSize: 13 }}>No data</div>}
            </Card>

            {/* Turnaround Time Histogram */}
            <Card title="Turnaround Time (hrs)">
              {turnaroundHist.length > 0 ? (
                <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 120 }}>
                  {turnaroundHist.map((bin, i) => {
                    const mx = Math.max(...turnaroundHist.map(b => b.count || b.value || 0), 1)
                    const val = bin.count || bin.value || 0
                    return (
                      <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>{val}</div>
                        <div style={{
                          width: '100%', maxWidth: 36,
                          height: `${Math.max((val / mx) * 100, 4)}%`,
                          background: '#a855f7', borderRadius: '4px 4px 0 0',
                          minHeight: 4
                        }} />
                        <div style={{ fontSize: 9, color: '#64748b', marginTop: 2 }}>{bin.bin || bin.label || ''}</div>
                      </div>
                    )
                  })}
                </div>
              ) : <div style={{ color: '#64748b', fontSize: 13 }}>No data</div>}
            </Card>
          </div>

          {/* Weekly Volume */}
          <Card title="Weekly Volume">
            {weeklyVol.length > 0 ? (
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 140 }}>
                {weeklyVol.map((w, i) => {
                  const mx = Math.max(...weeklyVol.map(x => x.count || x.value || 0), 1)
                  const val = w.count || w.value || 0
                  return (
                    <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 2 }}>{val}</div>
                      <div style={{
                        width: '100%', maxWidth: 48,
                        height: `${Math.max((val / mx) * 100, 4)}%`,
                        background: '#22c55e', borderRadius: '4px 4px 0 0',
                        minHeight: 4
                      }} />
                      <div style={{ fontSize: 9, color: '#64748b', marginTop: 4 }}>{w.week || w.label || ''}</div>
                    </div>
                  )
                })}
              </div>
            ) : <div style={{ color: '#64748b', fontSize: 13 }}>No data</div>}
          </Card>
        </div>
      )}

      {/* ── EEG Reads Tab ── */}
      {tab === 'eeg_reads' && (
        <Card title="EEG Read Queue">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient ID</th>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>Analyses</th>
                  <th style={thStyle}>Pending</th>
                  <th style={thStyle}>Seizure+</th>
                  <th style={thStyle}>Avg Confidence</th>
                  <th style={thStyle}>Signal Quality</th>
                  <th style={{ ...thStyle, width: 30 }}></th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => {
                  const isExp = expandedPatient === p.patient_id
                  return (
                    <React.Fragment key={p.patient_id}>
                      <tr onClick={() => setExpandedPatient(isExp ? null : p.patient_id)}
                        style={{ cursor: 'pointer', background: isExp ? '#334155' : 'transparent' }}>
                        <td style={tdStyle}>{p.patient_id}</td>
                        <td style={tdStyle}>{p.name || '--'}</td>
                        <td style={tdStyle}>{fmt(p.total_analyses)}</td>
                        <td style={tdStyle}>{fmt(p.pending_count)}</td>
                        <td style={tdStyle}>
                          {p.seizure_positive ? (
                            <span style={{ color: '#ef4444', fontWeight: 600 }}>Yes</span>
                          ) : (
                            <span style={{ color: '#22c55e' }}>No</span>
                          )}
                        </td>
                        <td style={tdStyle}>{p.avg_confidence != null ? `${fmt(p.avg_confidence)}%` : '--'}</td>
                        <td style={tdStyle}><SeverityBadge severity={p.signal_quality} /></td>
                        <td style={tdStyle}>
                          <span style={{
                            display: 'inline-block', transition: 'transform 0.2s',
                            transform: isExp ? 'rotate(90deg)' : 'rotate(0deg)',
                            color: '#94a3b8', fontSize: 14
                          }}>&#9654;</span>
                        </td>
                      </tr>
                      {isExp && (
                        <tr>
                          <td colSpan={8} style={{ padding: 0, borderBottom: '1px solid #334155' }}>
                            <div style={{ background: '#0f172a', padding: 16 }}>
                              <h4 style={{ margin: '0 0 10px', fontSize: 14, color: '#e2e8f0' }}>Analysis Results</h4>
                              {(p.analyses || []).length > 0 ? (
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                  <thead>
                                    <tr>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Analysis ID</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Prediction</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Confidence</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Status</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Created</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {(p.analyses || []).map((a, i) => (
                                      <tr key={i}>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{a.analysis_id || a.id || i + 1}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>
                                          <span style={{
                                            color: PREDICTION_COLORS[(a.prediction || '').toLowerCase()] || '#e2e8f0',
                                            fontWeight: 600
                                          }}>{a.prediction || '--'}</span>
                                        </td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{a.confidence != null ? `${fmt(a.confidence)}%` : '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{a.status || '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{a.created_at || a.date || '--'}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              ) : <div style={{ color: '#64748b', fontSize: 13 }}>No analyses found</div>}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
                {patients.length === 0 && (
                  <tr><td colSpan={8} style={{ ...tdStyle, textAlign: 'center', color: '#64748b' }}>No patients</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── HITL Review Tab ── */}
      {tab === 'hitl_review' && (
        <div>
          {/* Override Rate KPI */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
            <KPI label="Override Rate" value={overrideRate != null ? `${fmt(overrideRate)}%` : '--'} color="#f59e0b" />
            <KPI label="Total Overrides" value={fmt(kpis.total_overrides)} color="#ef4444" />
            <KPI label="Total Reviews" value={fmt(kpis.total_reviews)} color="#3b82f6" />
          </div>

          <Card title="Human-in-the-Loop Override Log">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Patient ID</th>
                    <th style={thStyle}>AI Prediction</th>
                    <th style={thStyle}>Human Decision</th>
                    <th style={thStyle}>Reason Code</th>
                    <th style={thStyle}>Reviewed At</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.filter(p => p.overrides && p.overrides.length > 0).flatMap(p =>
                    (p.overrides || []).map((ov, i) => (
                      <tr key={`${p.patient_id}-${i}`}>
                        <td style={tdStyle}>{p.patient_id}</td>
                        <td style={tdStyle}>
                          <span style={{
                            color: PREDICTION_COLORS[(ov.ai_prediction || '').toLowerCase()] || '#e2e8f0',
                            fontWeight: 600
                          }}>{ov.ai_prediction || '--'}</span>
                        </td>
                        <td style={tdStyle}>
                          <span style={{
                            color: PREDICTION_COLORS[(ov.human_decision || '').toLowerCase()] || '#e2e8f0',
                            fontWeight: 600
                          }}>{ov.human_decision || '--'}</span>
                        </td>
                        <td style={tdStyle}>
                          <span style={{
                            display: 'inline-block', padding: '2px 8px', borderRadius: 6,
                            fontSize: 11, fontWeight: 600, background: '#3b82f622', color: '#3b82f6'
                          }}>{ov.reason_code || ov.reason || '--'}</span>
                        </td>
                        <td style={tdStyle}>{ov.reviewed_at || ov.date || '--'}</td>
                      </tr>
                    ))
                  )}
                  {patients.filter(p => p.overrides && p.overrides.length > 0).length === 0 && (
                    <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#64748b' }}>No overrides recorded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Override Reason Summary */}
          {(() => {
            const reasonCounts = {}
            patients.forEach(p => {
              (p.overrides || []).forEach(ov => {
                const r = ov.reason_code || ov.reason || 'Unknown'
                reasonCounts[r] = (reasonCounts[r] || 0) + 1
              })
            })
            const reasons = Object.entries(reasonCounts).map(([reason, count]) => ({ label: reason, value: count, color: '#f59e0b' }))
            if (reasons.length === 0) return null
            return (
              <div style={{ marginTop: 20 }}>
                <Card title="Override Reason Summary">
                  <HBar items={reasons} />
                </Card>
              </div>
            )
          })()}
        </div>
      )}

      {/* ── Medications & Seizures Tab ── */}
      {tab === 'medications' && (
        <Card title="Medication & Seizure Tracking">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Medications</th>
                  <th style={thStyle}>Seizure Freq/mo</th>
                  <th style={thStyle}>Response Grade</th>
                  <th style={{ ...thStyle, width: 30 }}></th>
                </tr>
              </thead>
              <tbody>
                {patients.map(p => {
                  const isExp = expandedPatient === p.patient_id
                  const meds = p.medications || medSummary.find(m => m.patient_id === p.patient_id)?.medications || []
                  const freq = p.seizure_freq ?? p.seizure_frequency ?? '--'
                  const grade = p.response_grade || p.treatment_response || '--'
                  return (
                    <React.Fragment key={p.patient_id}>
                      <tr onClick={() => setExpandedPatient(isExp ? null : p.patient_id)}
                        style={{ cursor: 'pointer', background: isExp ? '#334155' : 'transparent' }}>
                        <td style={tdStyle}>{p.patient_id} {p.name ? `- ${p.name}` : ''}</td>
                        <td style={tdStyle}>
                          {Array.isArray(meds) ? meds.map((m, i) => (
                            <span key={i} style={{
                              display: 'inline-block', padding: '1px 6px', borderRadius: 4,
                              fontSize: 11, background: '#3b82f622', color: '#3b82f6',
                              marginRight: 4, marginBottom: 2
                            }}>{typeof m === 'string' ? m : m.name || m.medication || '--'}</span>
                          )) : <span style={{ color: '#64748b' }}>--</span>}
                        </td>
                        <td style={tdStyle}>{fmt(freq)}</td>
                        <td style={tdStyle}><ResponseBadge grade={grade} /></td>
                        <td style={tdStyle}>
                          <span style={{
                            display: 'inline-block', transition: 'transform 0.2s',
                            transform: isExp ? 'rotate(90deg)' : 'rotate(0deg)',
                            color: '#94a3b8', fontSize: 14
                          }}>&#9654;</span>
                        </td>
                      </tr>
                      {isExp && (
                        <tr>
                          <td colSpan={5} style={{ padding: 0, borderBottom: '1px solid #334155' }}>
                            <div style={{ background: '#0f172a', padding: 16 }}>
                              <h4 style={{ margin: '0 0 10px', fontSize: 14, color: '#e2e8f0' }}>Seizure Diary</h4>
                              {(p.seizure_diary || p.seizure_entries || []).length > 0 ? (
                                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                  <thead>
                                    <tr>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Date</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Type</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Duration</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Severity</th>
                                      <th style={{ ...thStyle, fontSize: 11 }}>Notes</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {(p.seizure_diary || p.seizure_entries || []).map((entry, i) => (
                                      <tr key={i}>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{entry.date || '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{entry.type || entry.seizure_type || '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{entry.duration || '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{entry.severity || '--'}</td>
                                        <td style={{ ...tdStyle, fontSize: 12 }}>{entry.notes || '--'}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              ) : <div style={{ color: '#64748b', fontSize: 13 }}>No seizure diary entries</div>}

                              {/* Medication details */}
                              {Array.isArray(meds) && meds.length > 0 && meds[0] && typeof meds[0] === 'object' && (
                                <div style={{ marginTop: 16 }}>
                                  <h4 style={{ margin: '0 0 10px', fontSize: 14, color: '#e2e8f0' }}>Medication Details</h4>
                                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                    <thead>
                                      <tr>
                                        <th style={{ ...thStyle, fontSize: 11 }}>Medication</th>
                                        <th style={{ ...thStyle, fontSize: 11 }}>Dose</th>
                                        <th style={{ ...thStyle, fontSize: 11 }}>Started</th>
                                        <th style={{ ...thStyle, fontSize: 11 }}>Status</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {meds.map((m, i) => (
                                        <tr key={i}>
                                          <td style={{ ...tdStyle, fontSize: 12 }}>{m.name || m.medication || '--'}</td>
                                          <td style={{ ...tdStyle, fontSize: 12 }}>{m.dose || m.dosage || '--'}</td>
                                          <td style={{ ...tdStyle, fontSize: 12 }}>{m.started || m.start_date || '--'}</td>
                                          <td style={{ ...tdStyle, fontSize: 12 }}>{m.status || 'Active'}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
                {patients.length === 0 && (
                  <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#64748b' }}>No patients</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {(defs.sections || []).map((sec, i) => (
            <Card key={i} title={sec.title}>
              <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {sec.content}
              </div>
            </Card>
          ))}
          {(!defs.sections || defs.sections.length === 0) && (
            <Card><div style={{ color: '#64748b' }}>No definitions available</div></Card>
          )}
        </div>
      )}
    </div>
  )
}
