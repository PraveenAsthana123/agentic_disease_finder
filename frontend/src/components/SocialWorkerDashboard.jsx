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
  if (s === 'high' || s === 'urgent' || s === 'severe') return '#ef4444'
  if (s === 'medium' || s === 'moderate' || s === 'moderate-to-severe') return '#f59e0b'
  if (s === 'low' || s === 'safe' || s === 'little-or-none') return '#22c55e'
  return '#94a3b8'
}

export default function SocialWorkerDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/api/social-worker`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Medical Social Worker Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}
  const sdoh = data.sdoh_screening || {}
  const caregiver = data.caregiver_burden || {}
  const benefits = data.benefits_vocational || {}
  const barriers = data.treatment_barriers || {}

  const sdohSummary = sdoh.summary || {}
  const cgSummary = caregiver.summary || {}
  const benSummary = benefits.summary || {}
  const barSummary = barriers.summary || {}

  // KPI tiles
  const kpis = [
    { label: 'Total Patients', value: summary.total_patients, color: COLORS[0] },
    { label: 'High SDOH Vulnerability', value: summary.high_sdoh_vulnerability, color: COLORS[1],
      urgent: (summary.high_sdoh_vulnerability || 0) > 0 },
    { label: 'High Caregiver Burnout', value: summary.high_caregiver_burnout, color: COLORS[5],
      urgent: (summary.high_caregiver_burnout || 0) > 0 },
    { label: 'Driving Restricted', value: summary.driving_restricted, color: COLORS[3] },
    { label: 'Mean Vulnerability Score', value: summary.mean_vulnerability_score, color: COLORS[4] },
    { label: 'Mean ZBI Score', value: summary.mean_zbi_score, color: COLORS[6] },
    { label: 'Mean Barrier Score', value: summary.mean_barrier_score, color: COLORS[7] },
    { label: 'Respite Referrals Needed', value: summary.respite_referrals_needed, color: COLORS[2] },
  ]

  // SDOH priority distribution pie
  const sdohPriorityData = [
    { name: 'High', value: sdohSummary.high_priority || 0 },
    { name: 'Moderate', value: sdohSummary.moderate_priority || 0 },
    { name: 'Low', value: sdohSummary.low_priority || 0 },
  ].filter(d => d.value > 0)

  // Treatment barrier category frequency bar chart
  const barrierFreqData = (barSummary.barrier_category_frequency || []).map(d => ({
    name: d.category,
    patients: d.patients_affected,
  }))

  // Caregiver burden level distribution (aggregate from results)
  const cgResults = caregiver.results || []
  const burnoutCounts = {}
  cgResults.forEach(r => {
    const level = r.burnout_risk_level || 'unknown'
    burnoutCounts[level] = (burnoutCounts[level] || 0) + 1
  })
  const burnoutDistrib = Object.entries(burnoutCounts).map(([name, value]) => ({ name, value }))

  // Benefits: driving eligibility breakdown
  const benResults = benefits.results || []
  let drivingEligible = 0
  let drivingRestricted = 0
  benResults.forEach(r => {
    const drv = r.driving_eligibility || {}
    if (drv.standard_license_eligible) drivingEligible++
    else drivingRestricted++
  })
  const drivingData = [
    { name: 'Eligible', value: drivingEligible },
    { name: 'Restricted', value: drivingRestricted },
  ].filter(d => d.value > 0)

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          Medical Social Worker Dashboard
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.module || 'Medical Social Worker (MSW)'} &middot; {fmt(summary.total_patients)} patients &middot; SDOH + Caregiver Burden + Benefits + Barriers
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
        {/* SDOH Priority Distribution */}
        {sdohPriorityData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              SDOH Priority Distribution (Mean Vulnerability: {fmt(sdohSummary.mean_vulnerability_score)})
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sdohPriorityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {sdohPriorityData.map((d, i) => (
                    <Cell key={i} fill={riskColor(d.name)} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Caregiver Burnout Risk Distribution */}
        {burnoutDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Caregiver Burnout Risk (Mean ZBI: {fmt(cgSummary.mean_zbi)})
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={burnoutDistrib} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {burnoutDistrib.map((d, i) => (
                    <Cell key={i} fill={riskColor(d.name)} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Treatment Barrier Category Frequency */}
        {barrierFreqData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Treatment Barrier Categories
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barrierFreqData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="patients" fill={COLORS[3]} name="Patients Affected" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Driving Eligibility */}
        {drivingData.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 300px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Driving Eligibility ({fmt(benResults.length)} patients)
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={drivingData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {drivingData.map((d, i) => (
                    <Cell key={i} fill={d.name === 'Eligible' ? COLORS[2] : COLORS[1]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* SDOH Screening Table */}
      <h2 style={sectionHeadingStyle}>Social Determinants of Health (SDOH) Screening</h2>
      {(sdoh.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            {sdoh.analysis} ({fmt(sdoh.total_patients_screened)} patients screened)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>Vulnerability</th>
                <th style={{ padding: '8px 12px' }}>Priority</th>
                <th style={{ padding: '8px 12px' }}>Employment</th>
                <th style={{ padding: '8px 12px' }}>Housing</th>
                <th style={{ padding: '8px 12px' }}>Transport</th>
                <th style={{ padding: '8px 12px' }}>Financial</th>
                <th style={{ padding: '8px 12px' }}>Social</th>
                <th style={{ padding: '8px 12px' }}>Education</th>
              </tr>
            </thead>
            <tbody>
              {sdoh.results.filter(r => r.vulnerability_score > 0).map((pt, i) => {
                const ds = pt.domain_scores || {}
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(pt.vulnerability_score)}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={badgeStyle(riskColor(pt.priority_level))}>{pt.priority_level || '—'}</span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds.Employment)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds.Housing)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds.Transportation)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds.Financial)}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds['Social Support'])}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(ds.Education)}</td>
                  </tr>
                )
              })}
              {sdoh.results.filter(r => r.vulnerability_score > 0).length === 0 && (
                <tr><td colSpan={10} style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>No patients with elevated vulnerability</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Caregiver Burden Table */}
      <h2 style={sectionHeadingStyle}>Caregiver Burden Assessment (ZBI / CSI Proxy)</h2>
      {(caregiver.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            {caregiver.analysis} ({fmt(caregiver.total_patients_assessed)} patients)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>ZBI Score</th>
                <th style={{ padding: '8px 12px' }}>ZBI Level</th>
                <th style={{ padding: '8px 12px' }}>CSI Score</th>
                <th style={{ padding: '8px 12px' }}>Burnout Risk</th>
                <th style={{ padding: '8px 12px' }}>Respite</th>
              </tr>
            </thead>
            <tbody>
              {caregiver.results.filter(r => r.zbi_proxy_score > 0).map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(pt.zbi_proxy_score)}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(riskColor(pt.zbi_level))}>{pt.zbi_level || '—'}</span>
                  </td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.csi_proxy_score)}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(riskColor(pt.burnout_risk_level))}>{pt.burnout_risk_level || '—'}</span>
                  </td>
                  <td style={{ padding: '8px 12px' }}>{pt.respite_referral_flag ? 'Yes' : 'No'}</td>
                </tr>
              ))}
              {caregiver.results.filter(r => r.zbi_proxy_score > 0).length === 0 && (
                <tr><td colSpan={7} style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>No patients with elevated caregiver burden</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Benefits & Vocational Support Table */}
      <h2 style={sectionHeadingStyle}>Benefits & Vocational Support</h2>
      {(benefits.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            {benefits.analysis} ({fmt(benefits.total_patients_assessed)} patients &middot; {fmt(benSummary.driving_restricted)} driving-restricted)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>Readiness</th>
                <th style={{ padding: '8px 12px' }}>Label</th>
                <th style={{ padding: '8px 12px' }}>Driving</th>
                <th style={{ padding: '8px 12px' }}>Disability Review</th>
              </tr>
            </thead>
            <tbody>
              {benefits.results.map((pt, i) => {
                const drv = pt.driving_eligibility || {}
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{fmt(pt.employment_readiness_score)}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={badgeStyle(pt.employment_readiness_label === 'ready' ? COLORS[2] : COLORS[3])}>
                        {pt.employment_readiness_label || '—'}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={badgeStyle(drv.standard_license_eligible ? COLORS[2] : COLORS[1])}>
                        {drv.standard_license_eligible ? 'Eligible' : 'Restricted'}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{pt.eligible_for_disability_review ? 'Yes' : 'No'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Treatment Barriers Table */}
      <h2 style={sectionHeadingStyle}>Treatment-Barrier Detection</h2>
      {(barriers.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            {barriers.analysis} ({fmt(barriers.total_patients_assessed)} patients &middot; Mean barrier score: {fmt(barSummary.mean_barrier_score)})
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient</th>
                <th style={{ padding: '8px 12px' }}>Age</th>
                <th style={{ padding: '8px 12px' }}>Barriers</th>
                <th style={{ padding: '8px 12px' }}>Score</th>
                <th style={{ padding: '8px 12px' }}>Priority</th>
                <th style={{ padding: '8px 12px' }}>Barrier Types</th>
              </tr>
            </thead>
            <tbody>
              {barriers.results.filter(r => r.barrier_count > 0).map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.age)}</td>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#ef4444' }}>{fmt(pt.barrier_count)}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.barrier_score)}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span style={badgeStyle(riskColor(pt.priority))}>{pt.priority || '—'}</span>
                  </td>
                  <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {(pt.detected_barriers || []).map(b => b.category || b).join(', ') || '—'}
                  </td>
                </tr>
              ))}
              {barriers.results.filter(r => r.barrier_count > 0).length === 0 && (
                <tr><td colSpan={6} style={{ padding: '12px', textAlign: 'center', color: '#94a3b8' }}>No patients with detected treatment barriers</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: social_worker_module &middot; {data.module || 'Medical Social Worker (MSW)'}
      </div>
    </div>
  )
}
