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
  if (s === 'high' || s === 'urgent') return '#ef4444'
  if (s === 'medium' || s === 'moderate') return '#f59e0b'
  if (s === 'low' || s === 'safe') return '#22c55e'
  return '#94a3b8'
}

export default function DietitianDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/dietitian`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Dietitian Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const modules = data.modules || {}
  const keto = modules.ketogenic_diet_eligibility || {}
  const malnutrition = modules.malnutrition_screening || {}
  const nutrient = modules.nutrient_analysis || {}
  const medNutrition = modules.medication_nutrition_interaction || {}

  const ketoSummary = keto.cohort_summary || {}
  const malSummary = malnutrition.cohort_summary || {}
  const nutSummary = nutrient.cohort_summary || {}
  const medNutSummary = medNutrition.cohort_summary || {}

  // KPI tiles
  const kpis = [
    { label: 'Keto Assessed', value: keto.unique_patients, color: COLORS[0] },
    { label: 'Mean Eligibility Score', value: ketoSummary.mean_eligibility_score, color: COLORS[4] },
    { label: 'Malnutrition Screened', value: malnutrition.unique_patients, color: COLORS[2] },
    {
      label: 'Malnutrition Referrals', value: malSummary.patients_needing_referral, color: COLORS[1],
      urgent: (malSummary.patients_needing_referral || 0) > 0,
    },
    { label: 'Nutrient Depletions', value: nutSummary.patients_with_depletions, color: COLORS[3] },
    { label: 'Med-Nutrition Interactions', value: medNutSummary.patients_with_interactions, color: COLORS[5] },
  ]

  // Malnutrition risk distribution pie
  const malRiskDistrib = (malSummary.risk_distribution || []).map(d => ({
    name: d.level,
    value: d.count,
  })).filter(d => d.value > 0)

  // Keto eligibility distribution pie
  const ketoDistrib = (ketoSummary.category_distribution || []).map(d => ({
    name: d.category.length > 30 ? d.category.slice(0, 30) + '…' : d.category,
    value: d.count,
  })).filter(d => d.value > 0)

  // Nutrient gap frequency bar chart
  const nutGapData = (nutSummary.nutrient_gap_frequency || []).map(d => ({
    name: d.nutrient.length > 20 ? d.nutrient.slice(0, 20) + '…' : d.nutrient,
    patients: d.patients_affected,
  }))

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          Clinical Dietitian Dashboard
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.role || 'Clinical Dietitian / Nutritionist'} &middot; {fmt(keto.unique_patients)} keto-assessed &middot; {fmt(malnutrition.unique_patients)} malnutrition-screened
        </p>
      </div>

      {/* KPI Tiles */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
        {kpis.map((kpi, i) => (
          <div key={i} style={{
            ...cardStyle,
            flex: '1 1 160px',
            borderLeft: `4px solid ${kpi.color}`,
            ...(kpi.urgent ? { border: '2px solid #ef4444', borderLeft: '4px solid #ef4444' } : {}),
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Charts Row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {/* Malnutrition Risk Distribution */}
        {malRiskDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Malnutrition Risk Distribution (Mean Score: {fmt(malSummary.mean_risk_score)})
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={malRiskDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {malRiskDistrib.map((_, i) => {
                    const c = malRiskDistrib[i].name
                    return <Cell key={i} fill={riskColor(c)} />
                  })}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Keto Eligibility Distribution */}
        {ketoDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Ketogenic Diet Eligibility
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ketoDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {ketoDistrib.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Nutrient Gap Frequency */}
        {nutGapData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              AED-Induced Nutrient Depletions
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={nutGapData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="patients" fill={COLORS[3]} name="Patients Affected" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Ketogenic Diet Eligibility Table */}
      <h2 style={sectionHeadingStyle}>Ketogenic Diet Eligibility</h2>
      {(keto.patient_profiles || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            Patient Eligibility Profiles ({fmt(keto.unique_patients)} patients)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>Score</th>
                <th style={{ padding: '8px 12px' }}>Category</th>
                <th style={{ padding: '8px 12px' }}>Recommended Diet</th>
              </tr>
            </thead>
            <tbody>
              {keto.patient_profiles.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(pt.eligibility_score)}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>{pt.eligibility_category || '—'}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(COLORS[0])}>{pt.recommended_diet_type || '—'}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Malnutrition Screening Table */}
      <h2 style={sectionHeadingStyle}>Malnutrition Screening</h2>
      {(malnutrition.patient_profiles || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            Risk Profiles ({fmt(malnutrition.unique_patients)} patients &middot; {fmt(malSummary.patients_needing_referral)} needing referral)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>Risk Score</th>
                <th style={{ padding: '8px 12px' }}>Risk Level</th>
                <th style={{ padding: '8px 12px' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {malnutrition.patient_profiles.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(pt.risk_score)}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(riskColor(pt.risk_level))}>{pt.risk_level || '—'}</span>
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {pt.recommended_action || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Nutrient Analysis Table */}
      <h2 style={sectionHeadingStyle}>AED-Nutrient Depletion Analysis</h2>
      {(nutrient.patient_profiles || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Medications</th>
                <th style={{ padding: '8px 12px' }}>Depletions</th>
                <th style={{ padding: '8px 12px' }}>Details</th>
              </tr>
            </thead>
            <tbody>
              {nutrient.patient_profiles.filter(pt => pt.depletions_identified > 0).map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>
                    {(pt.medications || []).map(m => m.drug_name).join(', ') || '—'}
                  </td>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#ef4444' }}>
                    {fmt(pt.depletions_identified)}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>
                    {(pt.depletion_details || []).map(d =>
                      `${d.nutrient_depleted} (${d.medication})`
                    ).join('; ') || '—'}
                  </td>
                </tr>
              ))}
              {nutrient.patient_profiles.filter(pt => pt.depletions_identified > 0).length === 0 && (
                <tr><td colSpan={4} style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>No depletions identified</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Medication-Nutrition Interactions */}
      <h2 style={sectionHeadingStyle}>Medication-Nutrition Interactions</h2>
      {(medNutrition.patient_profiles || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Medications</th>
                <th style={{ padding: '8px 12px' }}>Interactions</th>
                <th style={{ padding: '8px 12px' }}>High Severity</th>
                <th style={{ padding: '8px 12px' }}>Counseling Points</th>
              </tr>
            </thead>
            <tbody>
              {medNutrition.patient_profiles.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px', fontSize: 12 }}>
                    {(pt.medications || []).map(m => m.drug_name || m.drug).join(', ') || '—'}
                  </td>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: (pt.interactions_found || 0) > 0 ? '#ef4444' : '#22c55e' }}>
                    {fmt(pt.interactions_found)}
                  </td>
                  <td style={{ padding: '8px 12px', color: (pt.high_severity_count || 0) > 0 ? '#ef4444' : undefined }}>
                    {fmt(pt.high_severity_count)}
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {(pt.dietary_counseling_points || []).join('; ') || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Diet Types Reference */}
      {(keto.diet_types_reference || []).length > 0 && (
        <>
          <h2 style={sectionHeadingStyle}>Dietary Therapy Reference</h2>
          <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Diet Type</th>
                  <th style={{ padding: '8px 12px' }}>Ratio</th>
                  <th style={{ padding: '8px 12px' }}>Carb</th>
                  <th style={{ padding: '8px 12px' }}>Indication</th>
                  <th style={{ padding: '8px 12px' }}>Suitability</th>
                </tr>
              </thead>
              <tbody>
                {keto.diet_types_reference.map((dt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{dt.name}</td>
                    <td style={{ padding: '8px 12px' }}>{dt.ratio || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>{dt.carb_pct || dt.carb_limit || '—'}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>{dt.indication || '—'}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>{dt.suitability || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: dietitian_module &middot; {data.role || 'Clinical Dietitian / Nutritionist'}
      </div>
    </div>
  )
}
