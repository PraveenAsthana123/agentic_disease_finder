import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high' || s === 'contraindicated' || s === 'major') return '#ef4444'
  if (s === 'caution' || s === 'moderate' || s === 'medium') return '#f59e0b'
  if (s === 'low' || s === 'safe' || s === 'minor') return '#22c55e'
  return '#94a3b8'
}

export default function MedicationDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/medication`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Medication Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}

  // Flatten patient-nested medications into flat list with patient_id
  const myMeds = data.my_medications || {}
  const medications = (myMeds.patients || []).flatMap(p =>
    (p.medications || []).map(m => ({ ...m, patient_id: p.patient_id }))
  )

  // Flatten schedule: each patient has daily_schedule with morning/noon/evening/bedtime arrays
  const scheduleData = data.medication_schedule || {}
  const schedule = (scheduleData.patients || []).map(p => ({
    patient_id: p.patient_id,
    morning: (p.daily_schedule?.morning || []).map(m => ({ drug: m.drug_name, dose: m.dose_mg })),
    noon: (p.daily_schedule?.noon || []).map(m => ({ drug: m.drug_name, dose: m.dose_mg })),
    evening: (p.daily_schedule?.evening || []).map(m => ({ drug: m.drug_name, dose: m.dose_mg })),
    bedtime: (p.daily_schedule?.bedtime || []).map(m => ({ drug: m.drug_name, dose: m.dose_mg })),
  }))

  // Flatten adherence
  const adherenceData = data.adherence_summary || {}
  const adherenceResults = (adherenceData.patients || []).map(p => ({
    patient_id: p.patient_id,
    score: p.adherence_score_pct,
    level: p.adherence_level === 'needs_attention' ? 'low' : p.adherence_level === 'engaged' ? 'high' : (p.adherence_level || 'medium'),
    seizure_count: p.seizure_count,
    concern: p.concern_flag ? (p.concern_notes || []).join('; ') : null,
  }))

  // Flatten recommendations
  const recData = data.medication_recommendations || {}
  const recommendations = (recData.patients || []).flatMap(p =>
    (p.warnings || []).map(w => ({ ...w, patient_id: p.patient_id }))
  )

  // Flatten side effects — aggregate across patients
  const seData = data.side_effect_profile || {}
  const allEffects = {}
  const overlappingRisks = []
  ;(seData.patients || []).forEach(p => {
    (p.per_drug_profile || []).forEach(dp => {
      (dp.side_effects || []).forEach(eff => {
        if (!allEffects[eff]) allEffects[eff] = new Set()
        allEffects[eff].add(dp.drug)
      })
    })
    ;(p.overlapping_serious || p.high_concern_overlapping || []).forEach(o => {
      overlappingRisks.push({ effect: o.side_effect || o.effect, drugs: o.drugs || [] })
    })
  })
  const sideEffectRanked = Object.entries(allEffects)
    .map(([effect, drugs]) => ({ effect, count: drugs.size, drugs: [...drugs] }))
    .sort((a, b) => b.count - a.count)
  const sideEffects = {
    total_unique_effects: sideEffectRanked.length,
    ranked: sideEffectRanked,
    overlapping_risks: overlappingRisks,
  }

  // Drug distribution for pie chart
  const drugCounts = {}
  medications.forEach(m => {
    const name = m.drug_name || 'Unknown'
    drugCounts[name] = (drugCounts[name] || 0) + 1
  })
  const drugDistrib = Object.entries(drugCounts).map(([name, value]) => ({ name, value }))

  // Adherence distribution
  const adherenceLevels = { high: 0, medium: 0, low: 0 }
  adherenceResults.forEach(r => {
    const lvl = (r.level || '').toLowerCase()
    if (lvl in adherenceLevels) adherenceLevels[lvl]++
  })
  const adherenceDistrib = Object.entries(adherenceLevels)
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
    .filter(d => d.value > 0)

  // Side effect frequency bar chart
  const sideEffectChart = sideEffectRanked.slice(0, 10).map(s => ({
    name: s.effect,
    count: s.count,
  }))

  // KPI tiles
  const kpis = [
    { label: 'Patients on Meds', value: summary.total_patients_on_meds, color: COLORS[0] },
    { label: 'Total Prescriptions', value: summary.total_prescriptions, color: COLORS[4] },
    { label: 'Unique Drugs', value: summary.unique_drugs, color: COLORS[2] },
    { label: 'Most Common', value: summary.most_common_drug, color: COLORS[3] },
    {
      label: 'Polypharmacy (3+)',
      value: summary.polypharmacy_count,
      color: COLORS[5],
      warn: (summary.polypharmacy_count || 0) > 0,
    },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          My Medications
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          Patient Portal &middot; Current medications, schedule, adherence &amp; recommendations
        </p>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            ...cardStyle,
            flex: '1 1 160px',
            borderLeft: `4px solid ${kpi.color}`,
            ...(kpi.warn ? { border: '2px solid #f59e0b', borderLeft: '4px solid #f59e0b' } : {}),
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Current Medications Table */}
      <h2 style={sectionHeadingStyle}>Current Medications</h2>
      {medications.length > 0 ? (
        <div style={{ ...cardStyle, overflowX: 'auto', marginBottom: 16 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Patient</th>
                <th style={{ padding: '6px 10px' }}>Drug</th>
                <th style={{ padding: '6px 10px' }}>Brand</th>
                <th style={{ padding: '6px 10px' }}>Dose</th>
                <th style={{ padding: '6px 10px' }}>Frequency</th>
                <th style={{ padding: '6px 10px' }}>Class</th>
                <th style={{ padding: '6px 10px' }}>Side Effects</th>
              </tr>
            </thead>
            <tbody>
              {medications.map((med, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{med.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{med.drug_name || '—'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{med.brand || '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{med.dose_mg ? `${med.dose_mg} mg` : '—'}</td>
                  <td style={{ padding: '6px 10px' }}>{med.frequency || '—'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{med.drug_class || '—'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>
                    {(med.common_side_effects || []).slice(0, 3).join(', ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : <div style={cardStyle}>No medications on record.</div>}

      {/* Drug Distribution + Schedule */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Drug distribution pie */}
        {drugDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Drug Distribution</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={drugDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {drugDistrib.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Medication Schedule */}
        {schedule.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Daily Medication Schedule</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Morning</th>
                  <th style={{ padding: '6px 10px' }}>Noon</th>
                  <th style={{ padding: '6px 10px' }}>Evening</th>
                  <th style={{ padding: '6px 10px' }}>Bedtime</th>
                </tr>
              </thead>
              <tbody>
                {schedule.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {(s.morning || []).map(m => `${m.drug} ${m.dose}mg`).join(', ') || '—'}
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {(s.noon || []).map(m => `${m.drug} ${m.dose}mg`).join(', ') || '—'}
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {(s.evening || []).map(m => `${m.drug} ${m.dose}mg`).join(', ') || '—'}
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {(s.bedtime || []).map(m => `${m.drug} ${m.dose}mg`).join(', ') || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Adherence */}
      <h2 style={sectionHeadingStyle}>Adherence Summary</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {adherenceDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Adherence Levels</div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={adherenceDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  <Cell fill="#22c55e" />
                  <Cell fill="#f59e0b" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {adherenceResults.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Per-Patient Adherence</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>Score</th>
                  <th style={{ padding: '6px 10px' }}>Level</th>
                  <th style={{ padding: '6px 10px' }}>Seizures</th>
                  <th style={{ padding: '6px 10px' }}>Concern</th>
                </tr>
              </thead>
              <tbody>
                {adherenceResults.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.score)}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(
                        pt.level === 'high' ? '#22c55e' :
                        pt.level === 'medium' ? '#f59e0b' : '#ef4444'
                      )}>{pt.level || '—'}</span>
                    </td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.seizure_count)}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {pt.concern ? <span style={{ color: '#ef4444', fontWeight: 600 }}>{pt.concern}</span> : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Side Effect Profile */}
      <h2 style={sectionHeadingStyle}>Side Effect Profile</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {sideEffectChart.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 500px', minHeight: 300 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Top Side Effects Across All Medications ({fmt(sideEffects.total_unique_effects)} unique)
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={sideEffectChart} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill={COLORS[4]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {(sideEffects.overlapping_risks || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14, color: '#ef4444' }}>
              Overlapping Side Effect Risks
            </div>
            {sideEffects.overlapping_risks.map((risk, i) => (
              <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 13 }}>
                <span style={{ fontWeight: 600 }}>{risk.effect}</span>
                <span style={{ color: '#64748b' }}> — shared by {(risk.drugs || []).join(', ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recommendations */}
      <h2 style={sectionHeadingStyle}>Medication Recommendations</h2>
      {recommendations.length > 0 ? (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
          {recommendations.map((rec, i) => (
            <div key={i} style={{
              ...cardStyle,
              flex: '1 1 320px',
              borderLeft: `4px solid ${riskColor(rec.severity || rec.type)}`,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 14 }}>{rec.patient_id || 'All Patients'}</span>
                <span style={badgeStyle(riskColor(rec.severity || rec.type))}>
                  {rec.severity || rec.type || 'info'}
                </span>
              </div>
              <div style={{ fontSize: 13, color: '#334155', marginBottom: 4 }}>{rec.message}</div>
              {rec.action && (
                <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>{rec.action}</div>
              )}
            </div>
          ))}
        </div>
      ) : <div style={cardStyle}>No recommendations at this time.</div>}

      {/* Data source */}
      <div style={{ marginTop: 24, fontSize: 11, color: '#94a3b8', textAlign: 'center' }}>
        Data source: clinical.db medications table ({fmt(summary.total_prescriptions)} records) &middot; {fmt(summary.total_patients_on_meds)} patients
      </div>
    </div>
  )
}
