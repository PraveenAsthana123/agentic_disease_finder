import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high') return '#ef4444'
  if (s === 'moderate') return '#f59e0b'
  if (s === 'low') return '#22c55e'
  return '#94a3b8'
}

const severityColor = (sev) => {
  const s = (sev || '').toLowerCase()
  if (s === 'severe') return '#ef4444'
  if (s === 'moderate') return '#f59e0b'
  if (s === 'mild') return '#22c55e'
  return '#94a3b8'
}

export default function EpilepsyNurseDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/epilepsy-nurse`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Epilepsy Nurse Coordinator...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}
  const diary = data.seizure_diary || {}
  const adherence = data.adherence || {}
  const sudep = data.sudep_risk || {}
  const plans = data.action_plans || {}
  const education = data.education || {}

  // Severity pie chart from aggregate_severity
  const severityData = (diary.aggregate_severity || []).filter(s => s.count > 0)

  // Triggers bar chart from aggregate_triggers
  const triggersData = (diary.aggregate_triggers || []).filter(t => t.count > 0)

  // Injury summary from aggregate_injuries
  const injuryData = (diary.aggregate_injuries || []).filter(inj => inj.count > 0)

  // KPI tiles
  const kpis = [
    { label: 'Total Patients', value: summary.total_patients, color: '#1e88e5' },
    { label: 'Seizure Diary Entries', value: summary.total_seizure_diary_entries, color: '#7c4dff' },
    { label: 'High SUDEP Risk', value: summary.high_sudep_risk, color: '#ef4444', urgent: (summary.high_sudep_risk || 0) > 0 },
    { label: 'ER Visits', value: summary.total_er_visits, color: '#f59e0b' },
  ]

  // SUDEP risk distribution for pie chart
  const sudepDistrib = [
    { name: 'High', value: sudep.high_risk_count || 0 },
    { name: 'Moderate', value: sudep.moderate_risk_count || 0 },
    { name: 'Low', value: sudep.low_risk_count || 0 },
  ].filter(d => d.value > 0)

  // Adherence risk distribution
  const adherenceDistrib = [
    { name: 'High Risk', value: adherence.high_risk_count || 0 },
    { name: 'Moderate Risk', value: adherence.moderate_risk_count || 0 },
    { name: 'Low Risk', value: adherence.low_risk_count || 0 },
  ].filter(d => d.value > 0)

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          {data.title || 'Epilepsy Nurse Coordinator Dashboard'}
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.subtitle || `${fmt(summary.total_patients)} patients · ${fmt(summary.total_seizure_diary_entries)} diary entries · ${fmt(summary.high_sudep_risk)} high SUDEP risk`}
        </p>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            ...cardStyle,
            flex: '1 1 180px',
            borderLeft: `4px solid ${kpi.color}`,
            ...(kpi.urgent ? { border: '2px solid #ef4444', borderLeft: '4px solid #ef4444' } : {}),
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Seizure Diary Analysis */}
      <h2 style={sectionHeadingStyle}>Seizure Diary Analysis</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {severityData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Severity Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={severityData} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label>
                  {severityData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {triggersData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Seizure Triggers</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={triggersData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="trigger" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Diary stats row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={{ ...cardStyle, flex: '1 1 200px' }}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Total Entries</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{fmt(diary.total_entries)}</div>
        </div>
        <div style={{ ...cardStyle, flex: '1 1 200px' }}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Patients with Diary</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{fmt(diary.patients_with_diary)}</div>
        </div>
        <div style={{ ...cardStyle, flex: '1 1 200px' }}>
          <div style={{ fontSize: 13, color: '#64748b' }}>ER Visits</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#ef4444' }}>{fmt(diary.total_er_visits)}</div>
        </div>
        <div style={{ ...cardStyle, flex: '1 1 200px' }}>
          <div style={{ fontSize: 13, color: '#64748b' }}>Nocturnal Events</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{fmt(diary.total_nocturnal_events)}</div>
        </div>
      </div>

      {/* Per-patient diary details */}
      {(diary.patients || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Patient Seizure Summaries</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Events</th>
                <th style={{ padding: '8px 12px' }}>Freq/30d</th>
                <th style={{ padding: '8px 12px' }}>ER Visits</th>
                <th style={{ padding: '8px 12px' }}>Injuries</th>
                <th style={{ padding: '8px 12px' }}>Nocturnal</th>
              </tr>
            </thead>
            <tbody>
              {diary.patients.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.total_events)}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.frequency_per_30d)}</td>
                  <td style={{ padding: '8px 12px', color: (pt.er_visits || 0) > 0 ? '#ef4444' : undefined }}>{fmt(pt.er_visits)}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.injuries)}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.nocturnal_events)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* SUDEP Risk Assessment */}
      <h2 style={sectionHeadingStyle}>SUDEP Risk Assessment</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {sudepDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Risk Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sudepDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  <Cell fill="#ef4444" />
                  <Cell fill="#f59e0b" />
                  <Cell fill="#22c55e" />
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {(sudep.patients || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Per-Patient SUDEP Scores</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Score</th>
                  <th style={{ padding: '6px 10px' }}>Level</th>
                  <th style={{ padding: '6px 10px' }}>Top Factors</th>
                </tr>
              </thead>
              <tbody>
                {sudep.patients.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.sudep_score)}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(riskColor(pt.risk_level))}>{pt.risk_level || '—'}</span>
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12, color: '#64748b' }}>
                      {(pt.top_factors || pt.risk_factors || []).slice(0, 3).join(', ') || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* AED Adherence Monitoring */}
      <h2 style={sectionHeadingStyle}>AED Adherence Monitoring</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {adherenceDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Adherence Risk Distribution</div>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={adherenceDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                  <Cell fill="#ef4444" />
                  <Cell fill="#f59e0b" />
                  <Cell fill="#22c55e" />
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {(adherence.patients || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Patient Medication Profiles</div>
            {adherence.patients.map((pt, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13 }}>
                  {pt.patient_id}
                  <span style={{ ...badgeStyle(riskColor(pt.adherence_risk)), marginLeft: 8 }}>
                    {pt.adherence_risk || '—'}
                  </span>
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
                  Complexity: {fmt(pt.complexity_score)} · Drugs: {fmt(pt.drug_count)} · Burden: {fmt(pt.dosing_burden)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Seizure Action Plans */}
      <h2 style={sectionHeadingStyle}>Seizure Action Plans</h2>
      {(plans.patients || []).length > 0 && (
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
          {plans.patients.map((pt, i) => (
            <div key={i} style={{
              ...cardStyle,
              flex: '1 1 300px',
              borderLeft: `4px solid ${(pt.has_status_risk || pt.has_injury_history) ? '#ef4444' : '#22c55e'}`,
            }}>
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6 }}>{pt.patient_id}</div>
              {pt.first_aid_steps && (
                <div style={{ fontSize: 12, marginBottom: 6 }}>
                  <strong>First Aid:</strong>
                  <ul style={{ margin: '4px 0', paddingLeft: 16 }}>
                    {(pt.first_aid_steps || []).slice(0, 4).map((step, j) => (
                      <li key={j}>{step}</li>
                    ))}
                  </ul>
                </div>
              )}
              {pt.emergency_criteria && (
                <div style={{ fontSize: 12, color: '#ef4444' }}>
                  <strong>Emergency:</strong> {(pt.emergency_criteria || []).join('; ')}
                </div>
              )}
              {pt.rescue_medication && (
                <div style={{ fontSize: 12, marginTop: 4 }}>
                  <strong>Rescue Med:</strong> {pt.rescue_medication}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Patient Education */}
      <h2 style={sectionHeadingStyle}>Patient Education Assessment</h2>
      {(education.patients || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Domains Covered</th>
                <th style={{ padding: '8px 12px' }}>Priority Topics</th>
              </tr>
            </thead>
            <tbody>
              {education.patients.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.domains_covered)}/{fmt(pt.total_domains)}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>
                    {(pt.priority_topics || []).slice(0, 3).map((t, j) => (
                      <span key={j} style={{ ...badgeStyle('#6366f1'), marginRight: 4 }}>
                        {typeof t === 'string' ? t : (t.domain || t.topic || '—')}
                      </span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
