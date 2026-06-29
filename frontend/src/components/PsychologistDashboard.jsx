import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, ScatterChart, Scatter, ZAxis
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

const severityColor = (sev) => {
  const s = (sev || '').toLowerCase()
  if (s === 'severe' || s === 'urgent') return '#ef4444'
  if (s === 'moderate' || s === 'high') return '#f59e0b'
  if (s === 'mild' || s === 'routine') return '#22c55e'
  if (s === 'minimal' || s === 'low') return '#1e88e5'
  return '#94a3b8'
}

const priorityColor = (p) => {
  const s = (p || '').toLowerCase()
  if (s === 'urgent') return '#ef4444'
  if (s === 'high') return '#f59e0b'
  if (s === 'routine') return '#22c55e'
  return '#94a3b8'
}

const cellTh = { padding: '8px 12px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: '#475569', borderBottom: '2px solid #e2e8f0' }
const cellTd = { padding: '8px 12px', fontSize: 12, borderBottom: '1px solid #f1f5f9' }

export default function PsychologistDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/psychologist`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Psychologist...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data?.modules) return <div style={{ padding: 40 }}>No data available</div>

  const da = data.modules.depression_anxiety || {}
  const coping = data.modules.coping_resilience || {}
  const seizEmo = data.modules.seizure_emotion || {}
  const therapy = data.modules.therapy_planning || {}
  const summary = da.summary || {}
  const alerts = da.alerts || []
  const instruments = da.instruments || {}

  // PHQ-9 severity distribution for pie chart
  const phq9Dist = Object.entries(instruments.PHQ9?.severity_distribution || {}).map(([name, value]) => ({ name, value }))
  const gad7Dist = Object.entries(instruments.GAD7?.severity_distribution || {}).map(([name, value]) => ({ name, value }))

  // Coping resilience distribution for pie chart
  const resilDist = Object.entries(coping.resilience_distribution || {}).map(([name, value]) => ({ name, value }))

  // Therapy priority distribution for bar chart
  const priorityDist = Object.entries(therapy.priority_distribution || {}).map(([name, value]) => ({ name, value }))

  // Seizure-emotion scatter data
  const scatterData = (seizEmo.pairs || []).map(p => ({
    seizure_count: p.seizure_count,
    phq9: p.phq9,
    gad7: p.gad7,
    patient_id: p.patient_id,
  }))

  // KPI tiles
  const kpis = [
    { label: 'Patients Assessed', value: summary.patients_assessed, color: '#1e88e5' },
    { label: 'Moderate+ Depression', value: summary.moderate_or_worse_depression, color: '#f59e0b', urgent: (summary.moderate_or_worse_depression || 0) > 5 },
    { label: 'Moderate+ Anxiety', value: summary.moderate_or_worse_anxiety, color: '#7c4dff' },
    { label: 'Suicide Risk Flags', value: summary.suicide_risk_flags, color: '#ef4444', urgent: (summary.suicide_risk_flags || 0) > 0 },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          Clinical Psychologist
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.description || 'Depression/anxiety assessment, coping & resilience, seizure-emotion correlation, and therapy planning'}
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

      {/* Alerts */}
      {alerts.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, borderLeft: '4px solid #ef4444' }}>
          <div style={{ fontWeight: 600, color: '#ef4444', marginBottom: 8, fontSize: 14 }}>Clinical Alerts ({alerts.length})</div>
          {alerts.map((a, i) => (
            <div key={i} style={{ padding: '6px 0', borderBottom: i < alerts.length - 1 ? '1px solid #f1f5f9' : 'none', fontSize: 13 }}>
              <span style={{ fontWeight: 600, color: '#0f172a' }}>{a.patient_id}</span>
              <span style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, color: '#fff', background: severityColor(a.severity), marginLeft: 8 }}>{a.severity}</span>
              <span style={{ color: '#475569', marginLeft: 8 }}>{a.type}: {a.detail}</span>
            </div>
          ))}
        </div>
      )}

      {/* Depression / Anxiety */}
      <h2 style={sectionHeadingStyle}>Depression & Anxiety Assessment</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* PHQ-9 Distribution */}
        {phq9Dist.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>PHQ-9 Depression (n={instruments.PHQ9?.n})</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={phq9Dist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {phq9Dist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
        {/* GAD-7 Distribution */}
        {gad7Dist.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>GAD-7 Anxiety (n={instruments.GAD7?.n})</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={gad7Dist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {gad7Dist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Per-patient PHQ-9 scores bar chart */}
      {(instruments.PHQ9?.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>PHQ-9 Scores by Patient</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={(instruments.PHQ9.results || []).slice(0, 20)} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={60} />
              <YAxis domain={[0, 27]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [v, 'PHQ-9']} />
              <Bar dataKey="score" fill="#7c4dff" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Coping & Resilience */}
      <h2 style={sectionHeadingStyle}>Coping & Resilience</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Resilience distribution */}
        {resilDist.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Resilience Distribution (n={coping.n})</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={resilDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {resilDist.map((e, i) => <Cell key={i} fill={e.name === 'High' ? '#22c55e' : e.name === 'Moderate' ? '#f59e0b' : '#ef4444'} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
        {/* Coping profiles table */}
        {(coping.profiles || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '2 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>QOLIE-31 Profiles</div>
            <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={cellTh}>Patient</th>
                  <th style={cellTh}>Score</th>
                  <th style={cellTh}>QoL Level</th>
                  <th style={cellTh}>Resilience</th>
                  <th style={cellTh}>Therapy Target</th>
                </tr>
              </thead>
              <tbody>
                {coping.profiles.slice(0, 15).map((p, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...cellTd, fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={cellTd}>{fmt(p.qolie31_score)}/{fmt(p.max)}</td>
                    <td style={cellTd}>
                      <span style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, color: '#fff', background: severityColor(p.qol_level) }}>{p.qol_level}</span>
                    </td>
                    <td style={cellTd}>
                      <span style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, color: '#fff', background: p.resilience_proxy === 'High' ? '#22c55e' : p.resilience_proxy === 'Moderate' ? '#f59e0b' : '#ef4444' }}>{p.resilience_proxy}</span>
                    </td>
                    <td style={cellTd}>{p.therapy_target || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Seizure-Emotion Correlation */}
      <h2 style={sectionHeadingStyle}>Seizure–Emotion Correlation</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {scatterData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 450px', minHeight: 300 }}>
            <div style={{ fontWeight: 600, marginBottom: 4, fontSize: 14 }}>Seizure Frequency vs PHQ-9</div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
              Pearson r = {fmt(seizEmo.seizure_vs_depression_corr)} · {seizEmo.interpretation || ''}
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="seizure_count" name="Seizures" tick={{ fontSize: 11 }} label={{ value: 'Seizure count', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis dataKey="phq9" name="PHQ-9" tick={{ fontSize: 11 }} label={{ value: 'PHQ-9', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <ZAxis dataKey="gad7" range={[40, 200]} name="GAD-7" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => {
                  if (!payload?.length) return null
                  const d = payload[0].payload
                  return (
                    <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, padding: 8, fontSize: 12 }}>
                      <div style={{ fontWeight: 600 }}>{d.patient_id}</div>
                      <div>Seizures: {d.seizure_count}</div>
                      <div>PHQ-9: {d.phq9} · GAD-7: {d.gad7}</div>
                    </div>
                  )
                }} />
                <Scatter data={scatterData} fill="#7c4dff" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
        <div style={{ ...cardStyle, flex: '1 1 250px' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Correlation Summary</div>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
            <div><strong>Patients:</strong> {fmt(seizEmo.n_patients)}</div>
            <div><strong>Seizure vs Depression (r):</strong> {fmt(seizEmo.seizure_vs_depression_corr)}</div>
            <div style={{ marginTop: 8 }}>{seizEmo.interpretation || ''}</div>
            <div style={{ marginTop: 8, fontSize: 12, color: '#94a3b8', fontStyle: 'italic' }}>{seizEmo.note || ''}</div>
          </div>
        </div>
      </div>

      {/* Therapy Planning */}
      <h2 style={sectionHeadingStyle}>Therapy Planning</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Priority distribution bar chart */}
        {priorityDist.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 260 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Priority Distribution (n={therapy.n})</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={priorityDist} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {priorityDist.map((e, i) => <Cell key={i} fill={priorityColor(e.name)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {/* Therapy plans table */}
        {(therapy.plans || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '2 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Individual Therapy Plans</div>
            <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  <th style={cellTh}>Patient</th>
                  <th style={cellTh}>Depression</th>
                  <th style={cellTh}>Anxiety</th>
                  <th style={cellTh}>Priority</th>
                  <th style={cellTh}>Modalities</th>
                  <th style={cellTh}>Sessions</th>
                </tr>
              </thead>
              <tbody>
                {therapy.plans.slice(0, 20).map((p, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...cellTd, fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={cellTd}>{p.depression || '—'}</td>
                    <td style={cellTd}>{p.anxiety || '—'}</td>
                    <td style={cellTd}>
                      <span style={{ display: 'inline-block', padding: '1px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, color: '#fff', background: priorityColor(p.priority) }}>{p.priority}</span>
                    </td>
                    <td style={cellTd}>{(p.recommended_modalities || []).join(', ') || '—'}</td>
                    <td style={cellTd}>{fmt(p.sessions_suggested)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
