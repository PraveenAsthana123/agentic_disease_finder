import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const ADHERENCE_COLORS = { High: '#10b981', Medium: '#f59e0b', Low: '#ef4444' }
const RISK_COLORS = { Contraindicated: '#ef4444', 'High Risk': '#f59e0b', Caution: '#3b82f6' }
const SEVERITY_COLORS = { major: '#ef4444', moderate: '#f59e0b', minor: '#3b82f6' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: (color || '#64748b') + '22', color: color || '#64748b',
      fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{text}</span>
  )
}

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

export default function ClinicalPharmacistDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/clinical-pharmacist/overview`),
          axios.get(`${API_URL}/api/clinical-pharmacist/breakdown`),
          axios.get(`${API_URL}/api/clinical-pharmacist/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load clinical pharmacist data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128138;</div>
      Loading clinical pharmacist data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No clinical pharmacist data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'medications', label: 'Medication Inventory' },
    { id: 'interactions', label: 'Drug Interactions' },
    { id: 'adherence', label: 'Adherence' },
    { id: 'safety', label: 'ADR & Pregnancy' }
  ]

  const kpi = overview.kpis || {}
  const adherenceDist = overview.adherence_distribution || []
  const drugClassDist = overview.drug_class_distribution || []
  const pregRiskDist = overview.pregnancy_risk_distribution || []
  const interactionSev = overview.interaction_severity || []
  const medInventory = (breakdown && breakdown.medication_inventory) || []
  const interactionDetails = (breakdown && breakdown.interaction_details) || []
  const adherenceDetails = (breakdown && breakdown.adherence_details) || []
  const adrProfiles = (breakdown && breakdown.adr_profiles) || []
  const pregDetails = (breakdown && breakdown.pregnancy_details) || []
  const tdmDetails = (breakdown && breakdown.tdm_details) || []

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Clinical Pharmacist Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(kpi.total_patients)} patients | {fmt(kpi.total_medication_records)} med records | {fmt(kpi.total_interactions)} interactions | {fmt(kpi.asm_catalog_size)} ASMs cataloged
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients" value={kpi.total_patients} color="#3b82f6" /></Card>
            <Card><KPI label="Medications" value={kpi.total_medication_records} color="#8b5cf6" /></Card>
            <Card><KPI label="Interactions" value={kpi.total_interactions} sub={`${kpi.major_interactions} major`} color="#ef4444" /></Card>
            <Card><KPI label="Avg MMAS-8" value={kpi.avg_mmas8} sub={`MPR: ${fmt(kpi.avg_mpr)}`} color="#f59e0b" /></Card>
            <Card><KPI label="Low Adherence" value={kpi.low_adherence_patients} sub="patients" color="#ec4899" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Drug Class Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={drugClassDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {drugClassDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Adherence Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={adherenceDist} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Patients">
                    {adherenceDist.map((d, i) => <Cell key={i} fill={ADHERENCE_COLORS[d.name] || '#64748b'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Pregnancy Risk Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={pregRiskDist} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Medications">
                    {pregRiskDist.map((d, i) => <Cell key={i} fill={RISK_COLORS[d.name] || '#64748b'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Interaction Severity">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={interactionSev} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Interactions">
                    {interactionSev.map((d, i) => <Cell key={i} fill={SEVERITY_COLORS[d.name.toLowerCase()] || '#64748b'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {tab === 'medications' && (
        <Card title={`Medication Inventory (${medInventory.length} patients)`}>
          {medInventory.map((p, i) => (
            <div key={i} style={{ marginBottom: 20, paddingBottom: 16, borderBottom: i < medInventory.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <h4 style={{ fontSize: 14, color: '#1e293b', marginBottom: 8 }}>{p.patient_id}</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px' }}>Drug</th>
                      <th style={{ padding: '6px 10px' }}>Brand</th>
                      <th style={{ padding: '6px 10px' }}>Class</th>
                      <th style={{ padding: '6px 10px' }}>Dose (mg)</th>
                      <th style={{ padding: '6px 10px' }}>Frequency</th>
                      <th style={{ padding: '6px 10px' }}>Therapeutic Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(p.medications || []).map((m, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.drug}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.brand}</td>
                        <td style={{ padding: '6px 10px' }}>{m.drug_class}</td>
                        <td style={{ padding: '6px 10px' }}>{fmt(m.dose_mg)}</td>
                        <td style={{ padding: '6px 10px' }}>{m.frequency}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.therapeutic_range ? m.therapeutic_range.join(' - ') + ' mcg/mL' : '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </Card>
      )}

      {tab === 'interactions' && (
        <Card title={`Drug Interaction Check (${interactionDetails.length} patients)`}>
          {interactionDetails.map((p, i) => (
            <div key={i} style={{ marginBottom: 20, paddingBottom: 16, borderBottom: i < interactionDetails.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
              <h4 style={{ fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{p.patient_id}</h4>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>
                Drugs checked: {(p.drugs_checked || []).join(', ')}
              </div>
              {(p.interactions || []).length > 0 ? (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '6px 10px' }}>Pair</th>
                      <th style={{ padding: '6px 10px' }}>Severity</th>
                      <th style={{ padding: '6px 10px' }}>Mechanism</th>
                      <th style={{ padding: '6px 10px' }}>Clinical Effect</th>
                    </tr>
                  </thead>
                  <tbody>
                    {p.interactions.map((ix, j) => (
                      <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{ix.drug_a} + {ix.drug_b}</td>
                        <td style={{ padding: '6px 10px' }}><Badge text={ix.severity} color={SEVERITY_COLORS[ix.severity]} /></td>
                        <td style={{ padding: '6px 10px' }}>{ix.mechanism || '--'}</td>
                        <td style={{ padding: '6px 10px' }}>{ix.clinical_effect || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ fontSize: 13, color: '#10b981' }}>No interactions detected</div>
              )}
              {p.severity_summary && (
                <div style={{ display: 'flex', gap: 12, marginTop: 8, fontSize: 12 }}>
                  <span>Major: <strong style={{ color: '#ef4444' }}>{p.severity_summary.major}</strong></span>
                  <span>Moderate: <strong style={{ color: '#f59e0b' }}>{p.severity_summary.moderate}</strong></span>
                  <span>Minor: <strong style={{ color: '#3b82f6' }}>{p.severity_summary.minor}</strong></span>
                </div>
              )}
            </div>
          ))}
        </Card>
      )}

      {tab === 'adherence' && (
        <>
          <Card title={`Adherence Details (${adherenceDetails.length} patients)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient</th>
                    <th style={{ padding: '8px 12px' }}>Medications</th>
                    <th style={{ padding: '8px 12px' }}>MMAS-8</th>
                    <th style={{ padding: '8px 12px' }}>Level</th>
                    <th style={{ padding: '8px 12px' }}>MPR</th>
                    <th style={{ padding: '8px 12px' }}>Adherent</th>
                    <th style={{ padding: '8px 12px' }}>Seizure Gap</th>
                  </tr>
                </thead>
                <tbody>
                  {adherenceDetails.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{a.patient_id}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{(a.medications || []).join(', ')}</td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.mmas8_proxy_score)}</td>
                      <td style={{ padding: '8px 12px' }}><Badge text={a.mmas8_level} color={ADHERENCE_COLORS[a.mmas8_level === 'high' ? 'High' : a.mmas8_level === 'medium' ? 'Medium' : 'Low']} /></td>
                      <td style={{ padding: '8px 12px' }}>{fmt(a.mpr_estimate)}</td>
                      <td style={{ padding: '8px 12px', color: a.mpr_adherent ? '#10b981' : '#ef4444' }}>{a.mpr_adherent ? 'Yes' : 'No'}</td>
                      <td style={{ padding: '8px 12px', color: a.seizure_gap_flag ? '#ef4444' : '#10b981' }}>{a.seizure_gap_flag ? 'Yes' : 'No'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`TDM Monitoring (${tdmDetails.length} patients)`} span={2}>
            {tdmDetails.map((t, i) => (
              <div key={i} style={{ marginBottom: 16, paddingBottom: 12, borderBottom: i < tdmDetails.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>{t.patient_id} — {t.medications_monitored} drugs monitored</div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {(t.tdm || []).map((d, j) => (
                    <div key={j} style={{ padding: '8px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                      <strong>{d.drug}</strong> ({d.brand}) — {d.dose_mg}mg {d.frequency}<br />
                      Range: {d.therapeutic_range_mcg_ml ? d.therapeutic_range_mcg_ml.join('-') + ' mcg/mL' : '--'}<br />
                      <span style={{ color: d.monitoring_status === 'range_available' ? '#10b981' : '#f59e0b' }}>{d.monitoring_status}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        </>
      )}

      {tab === 'safety' && (
        <>
          <Card title={`ADR Profiles (${adrProfiles.length} patients)`}>
            {adrProfiles.map((p, i) => (
              <div key={i} style={{ marginBottom: 16, paddingBottom: 12, borderBottom: i < adrProfiles.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <h4 style={{ fontSize: 14, color: '#1e293b', marginBottom: 8 }}>{p.patient_id}</h4>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {(p.medications || []).map((m, j) => (
                    <div key={j} style={{ padding: '8px 14px', background: '#fff7ed', borderRadius: 8, fontSize: 12, border: '1px solid #fed7aa' }}>
                      <strong>{m.drug}</strong> ({m.brand})<br />
                      <span style={{ color: '#64748b' }}>ADRs ({m.adr_count}):</span>{' '}
                      {(m.known_adrs || []).join(', ')}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Card>

          <Card title={`Pregnancy Safety (${pregDetails.length} patients)`}>
            {pregDetails.map((p, i) => (
              <div key={i} style={{ marginBottom: 16, paddingBottom: 12, borderBottom: i < pregDetails.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <h4 style={{ fontSize: 14, color: '#1e293b', marginBottom: 8 }}>{p.patient_id}</h4>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {(p.medications || []).map((m, j) => (
                    <div key={j} style={{ padding: '8px 14px', background: '#fef2f2', borderRadius: 8, fontSize: 12, border: '1px solid #fecaca' }}>
                      <strong>{m.drug}</strong> ({m.brand}) — Cat {m.pregnancy_category}<br />
                      <Badge text={m.risk_level} color={RISK_COLORS[m.risk_level === 'contraindicated' ? 'Contraindicated' : m.risk_level === 'high_risk' ? 'High Risk' : 'Caution']} /><br />
                      <span style={{ color: '#64748b', fontSize: 11 }}>{m.guidance}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        </>
      )}

      {defs && (
        <div style={{ marginTop: 20, padding: 16, background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
          <strong>Definitions:</strong> {(defs.concepts || []).map(c => c.term).join(', ')}
        </div>
      )}
    </div>
  )
}
