import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : v ?? '--')

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, color: '#fff', background: color || '#94a3b8'
    }}>{text}</span>
  )
}

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high' || s === 'contraindicated' || s === 'major') return '#ef4444'
  if (s === 'caution' || s === 'moderate' || s === 'medium') return '#f59e0b'
  if (s === 'low' || s === 'safe' || s === 'minor') return '#22c55e'
  return '#94a3b8'
}

export default function MedicationDashboard() {
  const [data, setData] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [main, df] = await Promise.all([
          axios.get(`${API_URL}/medication`),
          axios.get(`${API_URL}/medication/definitions`)
        ])
        setData(main.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message || 'Failed to load medication data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Medication Dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const summary = data.summary || {}

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'drugs', label: 'Drug Analytics' },
    { id: 'schedule', label: 'Schedule & Adherence' },
    { id: 'safety', label: 'Recommendations & Safety' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Medication Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(summary.total_prescriptions)} prescriptions across {fmt(summary.unique_drugs)} drugs | {fmt(summary.total_patients_on_meds)} patients on medication | Most common: {summary.most_common_drug || '--'}
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={data} summary={summary} />}
      {tab === 'drugs' && <DrugAnalyticsTab data={data} />}
      {tab === 'schedule' && <ScheduleAdherenceTab data={data} />}
      {tab === 'safety' && <SafetyTab data={data} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ data, summary }) {
  const myMeds = data.my_medications || {}
  const medications = (myMeds.patients || []).flatMap(p =>
    (p.medications || []).map(m => ({ ...m, patient_id: p.patient_id }))
  )

  const drugCounts = {}
  medications.forEach(m => {
    const name = m.drug_name || 'Unknown'
    drugCounts[name] = (drugCounts[name] || 0) + 1
  })
  const drugDistrib = Object.entries(drugCounts).map(([name, value]) => ({ name, value }))

  const kpis = [
    { label: 'Patients on Meds', value: summary.total_patients_on_meds, color: COLORS[0] },
    { label: 'Total Prescriptions', value: summary.total_prescriptions, color: COLORS[4] },
    { label: 'Unique Drugs', value: summary.unique_drugs, color: COLORS[2] },
    { label: 'Most Common Drug', value: summary.most_common_drug, color: COLORS[3] },
    { label: 'Polypharmacy (3+)', value: summary.polypharmacy_count, color: summary.polypharmacy_count > 0 ? '#ef4444' : COLORS[2] },
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', justifyContent: 'space-around' }}>
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Card>

      {drugDistrib.length > 0 && (
        <Card title="Drug Distribution">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={drugDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {drugDistrib.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip /><Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card title="Current Medications" span={2}>
        {medications.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
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
                    <td style={{ padding: '6px 10px' }}>{med.drug_name || '--'}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{med.brand || '--'}</td>
                    <td style={{ padding: '6px 10px' }}>{med.dose_mg ? `${med.dose_mg} mg` : '--'}</td>
                    <td style={{ padding: '6px 10px' }}>{med.frequency || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{med.drug_class || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>
                      {(med.common_side_effects || []).slice(0, 3).join(', ') || '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No medications on record.</div>}
      </Card>
    </div>
  )
}

function DrugAnalyticsTab({ data }) {
  const myMeds = data.my_medications || {}
  const medications = (myMeds.patients || []).flatMap(p =>
    (p.medications || []).map(m => ({ ...m, patient_id: p.patient_id }))
  )

  const drugStats = {}
  medications.forEach(m => {
    const name = m.drug_name || 'Unknown'
    if (!drugStats[name]) drugStats[name] = { name, count: 0, doses: [], patients: new Set(), brand: m.brand || '--', drug_class: m.drug_class || '--', sideEffects: new Set() }
    drugStats[name].count++
    if (m.dose_mg) drugStats[name].doses.push(m.dose_mg)
    drugStats[name].patients.add(m.patient_id)
    ;(m.common_side_effects || []).forEach(se => drugStats[name].sideEffects.add(se))
  })
  const drugList = Object.values(drugStats).map(d => ({
    ...d,
    patients: [...d.patients],
    patientCount: d.patients.size,
    avgDose: d.doses.length > 0 ? Math.round(d.doses.reduce((a, b) => a + b, 0) / d.doses.length) : null,
    minDose: d.doses.length > 0 ? Math.min(...d.doses) : null,
    maxDose: d.doses.length > 0 ? Math.max(...d.doses) : null,
    sideEffects: [...d.sideEffects]
  })).sort((a, b) => b.count - a.count)

  const freqCounts = {}
  medications.forEach(m => {
    const f = m.frequency || 'Unknown'
    freqCounts[f] = (freqCounts[f] || 0) + 1
  })
  const freqData = Object.entries(freqCounts).map(([name, value]) => ({ name, value }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Prescriptions by Drug">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={drugList} margin={{ left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill={COLORS[0]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Frequency Distribution">
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={freqData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
              {freqData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip /><Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Drug Analytics" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '6px 10px' }}>Drug</th>
                <th style={{ padding: '6px 10px' }}>Brand</th>
                <th style={{ padding: '6px 10px' }}>Class</th>
                <th style={{ padding: '6px 10px' }}>Rx Count</th>
                <th style={{ padding: '6px 10px' }}>Patients</th>
                <th style={{ padding: '6px 10px' }}>Avg Dose</th>
                <th style={{ padding: '6px 10px' }}>Dose Range</th>
                <th style={{ padding: '6px 10px' }}>Top Side Effects</th>
              </tr>
            </thead>
            <tbody>
              {drugList.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.name}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{d.brand}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{d.drug_class}</td>
                  <td style={{ padding: '6px 10px' }}>{d.count}</td>
                  <td style={{ padding: '6px 10px' }}>{d.patientCount}</td>
                  <td style={{ padding: '6px 10px' }}>{d.avgDose != null ? `${d.avgDose} mg` : '--'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 12 }}>{d.minDose != null ? `${d.minDose}-${d.maxDose} mg` : '--'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{d.sideEffects.slice(0, 3).join(', ') || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function ScheduleAdherenceTab({ data }) {
  const scheduleData = data.medication_schedule || {}
  const schedule = (scheduleData.patients || []).map(p => ({
    patient_id: p.patient_id,
    morning: (p.daily_schedule?.morning || []).map(m => `${m.drug_name} ${m.dose_mg}mg`),
    noon: (p.daily_schedule?.noon || []).map(m => `${m.drug_name} ${m.dose_mg}mg`),
    evening: (p.daily_schedule?.evening || []).map(m => `${m.drug_name} ${m.dose_mg}mg`),
    bedtime: (p.daily_schedule?.bedtime || []).map(m => `${m.drug_name} ${m.dose_mg}mg`),
  }))

  const adherenceData = data.adherence_summary || {}
  const adherenceResults = (adherenceData.patients || []).map(p => ({
    patient_id: p.patient_id,
    score: p.adherence_score_pct,
    level: p.adherence_level === 'needs_attention' ? 'low' : p.adherence_level === 'engaged' ? 'high' : (p.adherence_level || 'medium'),
    seizure_count: p.seizure_count,
    concern: p.concern_flag ? (p.concern_notes || []).join('; ') : null,
  }))

  const adherenceLevels = { high: 0, medium: 0, low: 0 }
  adherenceResults.forEach(r => {
    const lvl = (r.level || '').toLowerCase()
    if (lvl in adherenceLevels) adherenceLevels[lvl]++
  })
  const adherenceDistrib = Object.entries(adherenceLevels)
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }))
    .filter(d => d.value > 0)
  const adherenceColors = ['#22c55e', '#f59e0b', '#ef4444']

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Daily Medication Schedule" span={2}>
        {schedule.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
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
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{s.morning.join(', ') || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{s.noon.join(', ') || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{s.evening.join(', ') || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>{s.bedtime.join(', ') || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No schedule data available.</div>}
      </Card>

      {adherenceDistrib.length > 0 && (
        <Card title="Adherence Levels">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={adherenceDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                {adherenceDistrib.map((_, i) => <Cell key={i} fill={adherenceColors[i % adherenceColors.length]} />)}
              </Pie>
              <Tooltip /><Legend />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card title="Per-Patient Adherence" span={adherenceDistrib.length > 0 ? 1 : 2}>
        {adherenceResults.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
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
                      <Badge text={pt.level || '--'} color={pt.level === 'high' ? '#22c55e' : pt.level === 'medium' ? '#f59e0b' : '#ef4444'} />
                    </td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.seizure_count)}</td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {pt.concern ? <span style={{ color: '#ef4444', fontWeight: 600 }}>{pt.concern}</span> : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No adherence data available.</div>}
      </Card>
    </div>
  )
}

function SafetyTab({ data }) {
  const recData = data.medication_recommendations || {}
  const recommendations = (recData.patients || []).flatMap(p =>
    (p.warnings || []).map(w => ({ ...w, patient_id: p.patient_id }))
  )

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

  const sideEffectChart = sideEffectRanked.slice(0, 10).map(s => ({ name: s.effect, count: s.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {sideEffectChart.length > 0 && (
        <Card title={`Top Side Effects (${sideEffectRanked.length} unique)`} span={2}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={sideEffectChart} layout="vertical" margin={{ left: 120 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count" fill={COLORS[4]} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {overlappingRisks.length > 0 && (
        <Card title="Overlapping Side Effect Risks">
          <div style={{ display: 'grid', gap: 8 }}>
            {overlappingRisks.map((risk, i) => (
              <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 13 }}>
                <span style={{ fontWeight: 600, color: '#ef4444' }}>{risk.effect}</span>
                <span style={{ color: '#64748b' }}> -- shared by {(risk.drugs || []).join(', ')}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Medication Recommendations" span={overlappingRisks.length > 0 ? 1 : 2}>
        {recommendations.length > 0 ? (
          <div style={{ display: 'grid', gap: 12 }}>
            {recommendations.map((rec, i) => (
              <div key={i} style={{
                padding: 12, borderRadius: 8, background: '#f8fafc',
                borderLeft: `4px solid ${riskColor(rec.severity || rec.type)}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{rec.patient_id || 'All Patients'}</span>
                  <Badge text={rec.severity || rec.type || 'info'} color={riskColor(rec.severity || rec.type)} />
                </div>
                <div style={{ fontSize: 13, color: '#334155', marginBottom: 2 }}>{rec.message}</div>
                {rec.action && <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>{rec.action}</div>}
              </div>
            ))}
          </div>
        ) : <div style={{ color: '#94a3b8' }}>No recommendations at this time.</div>}
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div style={{ color: '#94a3b8' }}>No definitions available.</div>
  const sections = defs.sections || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((section, si) => (
        <Card key={si} title={section.title}>
          <div style={{ display: 'grid', gap: 12 }}>
            {(section.items || []).map((item, ii) => (
              <div key={ii}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
