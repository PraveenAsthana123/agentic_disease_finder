import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#22c55e', '#f59e0b', '#ef4444', '#7c4dff', '#ec4899', '#14b8a6', '#94a3b8']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '—')

const cardStyle = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
}

const statusColor = (s) => {
  const v = (s || '').toLowerCase()
  if (v === 'complete') return '#22c55e'
  if (v === 'partial') return '#f59e0b'
  return '#94a3b8'
}

const statusIcon = (s) => {
  const v = (s || '').toLowerCase()
  if (v === 'complete') return '✅'
  if (v === 'partial') return '🔶'
  return '⏳'
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

export default function ReportsDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)
  const [defs, setDefs] = useState(null)
  const [summary, setSummary] = useState(null)
  const [filter, setFilter] = useState('all')
  const [expandedPatient, setExpandedPatient] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/patient-reports`),
      axios.get(`${API_URL}/api/patient-reports/summary`),
      axios.get(`${API_URL}/api/patient-reports/definitions`),
    ])
      .then(([rData, rSummary, rDefs]) => {
        setData(rData.data)
        setSummary(rSummary.data)
        setDefs(rDefs.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>Loading reports...</div>
  if (error) return <div style={{ textAlign: 'center', padding: 60, color: '#dc2626' }}>Error: {error}</div>
  if (!data?.available) return <div style={{ textAlign: 'center', padding: 60, color: '#64748b' }}>Reports not available</div>

  const reports = data.reports || []
  const filtered = filter === 'all' ? reports : reports.filter(r => r.status === filter)

  const statusCounts = [
    { name: 'Complete', value: reports.filter(r => r.status === 'complete').length, color: '#22c55e' },
    { name: 'Partial', value: reports.filter(r => r.status === 'partial').length, color: '#f59e0b' },
    { name: 'Pending', value: reports.filter(r => r.status === 'pending').length, color: '#94a3b8' },
  ].filter(s => s.value > 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0, fontSize: 20, color: '#1e293b' }}>
          📄 My Reports — Patient EEG & Clinical Summaries
        </h2>
        <button onClick={() => setShowDefs(!showDefs)}
          style={{ background: showDefs ? '#1e88e5' : '#e2e8f0', color: showDefs ? '#fff' : '#475569',
            border: 'none', borderRadius: 8, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}>
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {/* Definitions */}
      {showDefs && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 15, color: '#0369a1' }}>Metric Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {Object.entries(defs.definitions).map(([k, v]) => (
              <div key={k} style={{ fontSize: 13 }}>
                <strong style={{ color: '#0c4a6e' }}>{k}:</strong>{' '}
                <span style={{ color: '#334155' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary KPI cards */}
      {summary && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
          {[
            { label: 'Total Patients', value: summary.total_patients, icon: '👤' },
            { label: 'EEG Analyses', value: summary.total_analyses, icon: '🧠' },
            { label: 'Assessments', value: summary.total_assessments, icon: '📋' },
            { label: 'Seizure Events', value: summary.total_seizure_events, icon: '⚡' },
            { label: 'Expert Reviews', value: summary.total_expert_reviews, icon: '👨‍⚕️' },
            { label: 'Clinical Decisions', value: summary.total_clinical_decisions, icon: '✍️' },
            { label: 'HITL Reviews', value: summary.total_hitl_reviews, icon: '🔍' },
          ].map(kpi => (
            <div key={kpi.label} style={{ ...cardStyle, textAlign: 'center' }}>
              <div style={{ fontSize: 24 }}>{kpi.icon}</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{fmt(kpi.value)}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{kpi.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
        {/* Report status pie */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 8px', fontSize: 15, color: '#1e293b' }}>Report Status</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={statusCounts} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}`}>
                {statusCounts.map((s, i) => <Cell key={i} fill={s.color} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Disease breakdown bar */}
        {summary?.disease_breakdown?.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 8px', fontSize: 15, color: '#1e293b' }}>By Disease</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={summary.disease_breakdown}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Instrument breakdown bar */}
        {summary?.instrument_breakdown?.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ margin: '0 0 8px', fontSize: 15, color: '#1e293b' }}>By Assessment Instrument</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={summary.instrument_breakdown.slice(0, 8)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="instrument" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Filter tabs */}
      <div style={{ display: 'flex', gap: 8 }}>
        {['all', 'complete', 'partial', 'pending'].map(f => (
          <button key={f} onClick={() => setFilter(f)}
            style={{
              background: filter === f ? '#1e88e5' : '#f1f5f9',
              color: filter === f ? '#fff' : '#475569',
              border: 'none', borderRadius: 8, padding: '6px 16px', cursor: 'pointer',
              fontSize: 13, fontWeight: filter === f ? 600 : 400,
            }}>
            {f === 'all' ? `All (${reports.length})` :
              `${f.charAt(0).toUpperCase() + f.slice(1)} (${reports.filter(r => r.status === f).length})`}
          </button>
        ))}
      </div>

      {/* Patient report cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {filtered.map(r => (
          <div key={r.patient_id} style={{ ...cardStyle, cursor: 'pointer', border: expandedPatient === r.patient_id ? '2px solid #1e88e5' : '1px solid #e2e8f0' }}
            onClick={() => setExpandedPatient(expandedPatient === r.patient_id ? null : r.patient_id)}>
            {/* Header row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 20 }}>{statusIcon(r.status)}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 15, color: '#1e293b' }}>
                    {r.name || r.patient_id}
                    <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{r.patient_id}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>
                    {r.age ? `${r.age}y` : ''} {r.gender} — {r.disease || 'N/A'} — {r.department || 'N/A'}
                  </div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={badgeStyle(statusColor(r.status))}>{r.status}</span>
                <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ width: `${r.completeness}%`, height: '100%', background: statusColor(r.status), borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 12, color: '#64748b' }}>{r.completeness}%</span>
              </div>
            </div>

            {/* Quick stats row */}
            <div style={{ display: 'flex', gap: 16, marginTop: 10, flexWrap: 'wrap' }}>
              {[
                { icon: '🧠', label: 'EEG', value: r.latest_analysis ? `${r.latest_analysis.predicted_label} (${r.latest_analysis.confidence})` : 'None' },
                { icon: '📋', label: 'Assessments', value: r.assessment_count },
                { icon: '⚡', label: 'Seizure Log', value: r.seizure_diary_entries },
                { icon: '👨‍⚕️', label: 'Expert Reviews', value: r.expert_reviews?.length || 0 },
                { icon: '💊', label: 'Medications', value: r.medication_count },
                { icon: '🔍', label: 'HITL', value: r.hitl_review_count },
              ].map(s => (
                <div key={s.label} style={{ fontSize: 12, color: '#475569' }}>
                  {s.icon} <strong>{s.label}:</strong> {fmt(s.value)}
                </div>
              ))}
            </div>

            {/* Expanded detail */}
            {expandedPatient === r.patient_id && (
              <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #e2e8f0' }}>
                {/* Latest analysis detail */}
                {r.latest_analysis && (
                  <div style={{ marginBottom: 12 }}>
                    <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>🧠 Latest EEG Analysis</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, fontSize: 13 }}>
                      <div><strong>Prediction:</strong> {r.latest_analysis.predicted_label}</div>
                      <div><strong>Confidence:</strong> {fmt(r.latest_analysis.confidence)}</div>
                      <div><strong>Signal Quality:</strong> {r.latest_analysis.signal_quality || '—'}</div>
                      <div><strong>Date:</strong> {r.latest_analysis.date || '—'}</div>
                    </div>
                  </div>
                )}

                {/* Latest assessment */}
                {r.latest_assessment && (
                  <div style={{ marginBottom: 12 }}>
                    <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>📋 Latest Assessment</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, fontSize: 13 }}>
                      <div><strong>Instrument:</strong> {r.latest_assessment.instrument}</div>
                      <div><strong>Score:</strong> {r.latest_assessment.score}/{r.latest_assessment.max_score}</div>
                      <div><strong>Level:</strong> {r.latest_assessment.level || '—'}</div>
                      <div><strong>Date:</strong> {r.latest_assessment.created_at || '—'}</div>
                    </div>
                  </div>
                )}

                {/* Expert reviews */}
                {r.expert_reviews?.length > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>👨‍⚕️ Expert Reviews</h4>
                    <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Role</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Expert</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Finding</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Agree w/ AI</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.expert_reviews.map((rev, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 8px' }}>{rev.role}</td>
                            <td style={{ padding: '6px 8px' }}>{rev.expert}</td>
                            <td style={{ padding: '6px 8px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{rev.finding}</td>
                            <td style={{ padding: '6px 8px' }}>{rev.agree_with_ai || '—'}</td>
                            <td style={{ padding: '6px 8px' }}>{rev.created_at || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Clinical decisions */}
                {r.clinical_decisions?.length > 0 && (
                  <div>
                    <h4 style={{ margin: '0 0 6px', fontSize: 14, color: '#1e293b' }}>✍️ Clinical Decisions</h4>
                    <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ background: '#f8fafc', textAlign: 'left' }}>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>AI Prediction</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Final Decision</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Reviewer</th>
                          <th style={{ padding: '6px 8px', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {r.clinical_decisions.map((dec, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                            <td style={{ padding: '6px 8px' }}>{dec.ai_prediction}</td>
                            <td style={{ padding: '6px 8px' }}>{fmt(dec.ai_confidence)}</td>
                            <td style={{ padding: '6px 8px' }}>{dec.final_decision}</td>
                            <td style={{ padding: '6px 8px' }}>{dec.reviewer}</td>
                            <td style={{ padding: '6px 8px' }}>{dec.created_at || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div style={{ textAlign: 'center', padding: 40, color: '#94a3b8' }}>
          No reports match the selected filter.
        </div>
      )}
    </div>
  )
}
