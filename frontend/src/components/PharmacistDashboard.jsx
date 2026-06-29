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
  if (s === 'high' || s === 'contraindicated') return '#ef4444'
  if (s === 'caution' || s === 'moderate') return '#f59e0b'
  if (s === 'low' || s === 'safe') return '#22c55e'
  return '#94a3b8'
}

export default function PharmacistDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${API_URL}/pharmacist`)
      .then(r => { setData(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Pharmacist Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data) return <div style={{ padding: 40 }}>No data available</div>

  const summary = data.summary || {}
  const reconciliation = data.reconciliation || {}
  const interactions = data.interactions || {}
  const tdm = data.tdm || {}
  const adr = data.adr || {}
  const pregnancySafety = data.pregnancy_safety || {}
  const adherence = data.adherence || {}
  const adherenceSummary = adherence.summary || {}

  // Adherence distribution pie chart
  const adherenceDistrib = [
    { name: 'High', value: adherenceSummary.high_adherence || 0 },
    { name: 'Medium', value: adherenceSummary.medium_adherence || 0 },
    { name: 'Low', value: adherenceSummary.low_adherence || 0 },
  ].filter(d => d.value > 0)

  // KPI tiles
  const kpis = [
    { label: 'Total Patients', value: summary.total_patients, color: COLORS[0] },
    { label: 'Medication Records', value: summary.total_medication_records, color: COLORS[4] },
    {
      label: 'Drug Interactions Found', value: summary.total_interactions_found, color: COLORS[1],
      urgent: (summary.total_interactions_found || 0) > 0,
    },
    { label: 'ADR Overlaps', value: summary.overlapping_adr_count, color: COLORS[3] },
    {
      label: 'Low Adherence Patients', value: summary.low_adherence_patients, color: COLORS[5],
      urgent: (summary.low_adherence_patients || 0) > 0,
    },
    {
      label: 'Pregnancy Contraindicated', value: summary.contraindicated_in_pregnancy, color: COLORS[7],
      warn: (summary.contraindicated_in_pregnancy || 0) > 0,
    },
  ]

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#0f172a', margin: 0 }}>
          Clinical Pharmacist Dashboard
        </h1>
        <p style={{ color: '#64748b', marginTop: 4, fontSize: 14 }}>
          {data.module || 'Clinical Pharmacist (Epilepsy)'} &middot; {fmt(summary.total_patients)} patients &middot; {fmt(summary.total_medication_records)} medication records
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
            ...(kpi.warn ? { border: '2px solid #f59e0b', borderLeft: '4px solid #f59e0b' } : {}),
          }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#0f172a' }}>{fmt(kpi.value)}</div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Adherence Distribution */}
      <h2 style={sectionHeadingStyle}>Adherence Distribution</h2>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
        {adherenceDistrib.length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 320px', minHeight: 280 }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
              Adherence Levels (Avg MMAS-8: {fmt(adherenceSummary.avg_mmas8)} &middot; Avg MPR: {fmt(adherenceSummary.avg_mpr)})
            </div>
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

        {(adherence.results || []).length > 0 && (
          <div style={{ ...cardStyle, flex: '1 1 400px', overflowX: 'auto' }}>
            <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>Per-Patient Adherence</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Patient</th>
                  <th style={{ padding: '6px 10px' }}>MPR</th>
                  <th style={{ padding: '6px 10px' }}>MMAS-8</th>
                  <th style={{ padding: '6px 10px' }}>Level</th>
                  <th style={{ padding: '6px 10px' }}>Medications</th>
                </tr>
              </thead>
              <tbody>
                {adherence.results.map((pt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.mpr_estimate)}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(pt.mmas8_proxy_score)}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={badgeStyle(
                        pt.mmas8_level === 'high' ? '#22c55e' :
                        pt.mmas8_level === 'medium' ? '#f59e0b' : '#ef4444'
                      )}>{pt.mmas8_level || '—'}</span>
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 12 }}>
                      {(pt.medications || []).join(', ') || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Medication Reconciliation */}
      <h2 style={sectionHeadingStyle}>Medication Reconciliation</h2>
      {(reconciliation.reconciled || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <div style={{ fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
            Reconciled Patients ({fmt(reconciliation.total_patients)} patients &middot; {fmt(reconciliation.total_medication_records)} records)
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Unique Medications</th>
                <th style={{ padding: '8px 12px' }}>Duplicates Found</th>
                <th style={{ padding: '8px 12px' }}>Gaps Detected</th>
              </tr>
            </thead>
            <tbody>
              {reconciliation.reconciled.map((pt, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                  <td style={{ padding: '8px 12px' }}>{fmt(pt.unique_medications)}</td>
                  <td style={{ padding: '8px 12px', color: (pt.duplicates_found || []).length > 0 ? '#ef4444' : undefined }}>
                    {(pt.duplicates_found || []).length}
                  </td>
                  <td style={{ padding: '8px 12px', color: (pt.gaps_detected || []).length > 0 ? '#f59e0b' : undefined }}>
                    {(pt.gaps_detected || []).length}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Drug Interactions */}
      <h2 style={sectionHeadingStyle}>Drug Interactions</h2>
      {(interactions.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Drugs Checked</th>
                <th style={{ padding: '8px 12px' }}>Interactions</th>
                <th style={{ padding: '8px 12px' }}>Major</th>
                <th style={{ padding: '8px 12px' }}>Moderate</th>
                <th style={{ padding: '8px 12px' }}>Minor</th>
              </tr>
            </thead>
            <tbody>
              {interactions.results.map((pt, i) => {
                const sev = pt.severity_summary || {}
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>
                      {(pt.drugs_checked || []).join(', ') || '—'}
                    </td>
                    <td style={{ padding: '8px 12px', color: (pt.interactions_found || 0) > 0 ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
                      {fmt(pt.interactions_found)}
                    </td>
                    <td style={{ padding: '8px 12px', color: (sev.major || 0) > 0 ? '#ef4444' : undefined }}>
                      {fmt(sev.major)}
                    </td>
                    <td style={{ padding: '8px 12px', color: (sev.moderate || 0) > 0 ? '#f59e0b' : undefined }}>
                      {fmt(sev.moderate)}
                    </td>
                    <td style={{ padding: '8px 12px' }}>{fmt(sev.minor)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Therapeutic Drug Monitoring */}
      <h2 style={sectionHeadingStyle}>Therapeutic Drug Monitoring</h2>
      {(tdm.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Drug</th>
                <th style={{ padding: '8px 12px' }}>Brand</th>
                <th style={{ padding: '8px 12px' }}>Dose (mg)</th>
                <th style={{ padding: '8px 12px' }}>Frequency</th>
                <th style={{ padding: '8px 12px' }}>Therapeutic Range</th>
                <th style={{ padding: '8px 12px' }}>Recommendation</th>
              </tr>
            </thead>
            <tbody>
              {tdm.results.flatMap((pt, pi) =>
                (pt.tdm || []).map((med, mi) => (
                  <tr key={`${pi}-${mi}`} style={{ borderBottom: '1px solid #f1f5f9', background: (pi + mi) % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{med.drug || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>{med.brand || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>{fmt(med.dose_mg)}</td>
                    <td style={{ padding: '8px 12px' }}>{med.frequency || '—'}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>
                      {Array.isArray(med.therapeutic_range_mcg_ml)
                        ? `${med.therapeutic_range_mcg_ml[0]}–${med.therapeutic_range_mcg_ml[1]} mcg/mL`
                        : '—'}
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {med.recommendation || '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* ADR Profile */}
      <h2 style={sectionHeadingStyle}>ADR Profile</h2>
      {(adr.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Drug</th>
                <th style={{ padding: '8px 12px' }}>Brand</th>
                <th style={{ padding: '8px 12px' }}>Known ADRs</th>
                <th style={{ padding: '8px 12px' }}>Overlapping ADRs</th>
              </tr>
            </thead>
            <tbody>
              {adr.results.flatMap((pt, pi) =>
                (pt.medications || []).map((med, mi) => (
                  <tr key={`${pi}-${mi}`} style={{ borderBottom: '1px solid #f1f5f9', background: (pi + mi) % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{med.drug || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>{med.brand || '—'}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12 }}>
                      {(med.known_adrs || []).join(', ') || '—'}
                    </td>
                    {mi === 0 && (
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#ef4444', fontWeight: (pt.overlapping_adrs || []).length > 0 ? 600 : 400 }}
                          rowSpan={(pt.medications || []).length}>
                        {(pt.overlapping_adrs || []).join(', ') || 'None'}
                      </td>
                    )}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Pregnancy Safety */}
      <h2 style={sectionHeadingStyle}>Pregnancy Safety</h2>
      {(pregnancySafety.results || []).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: '8px 12px' }}>Patient ID</th>
                <th style={{ padding: '8px 12px' }}>Drug</th>
                <th style={{ padding: '8px 12px' }}>Brand</th>
                <th style={{ padding: '8px 12px' }}>Category</th>
                <th style={{ padding: '8px 12px' }}>Risk Level</th>
                <th style={{ padding: '8px 12px' }}>Guidance</th>
              </tr>
            </thead>
            <tbody>
              {pregnancySafety.results.flatMap((pt, pi) =>
                (pt.medications || []).map((med, mi) => (
                  <tr key={`${pi}-${mi}`} style={{ borderBottom: '1px solid #f1f5f9', background: (pi + mi) % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{pt.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{med.drug || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>{med.brand || '—'}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{ fontWeight: 600 }}>{med.pregnancy_category || '—'}</span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={badgeStyle(riskColor(med.risk_level))}>{med.risk_level || '—'}</span>
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: 12, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {med.guidance || '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: pharmacist_module &middot; ASM catalog ({fmt(summary.asm_catalog_size)}) &middot; Interaction KB ({fmt(summary.interaction_kb_size)})
      </div>
    </div>
  )
}
