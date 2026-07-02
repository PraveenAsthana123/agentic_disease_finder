import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

export default function PrescriptionsDashboard() {
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
          axios.get(`${API_URL}/prescriptions/overview`),
          axios.get(`${API_URL}/prescriptions/breakdown`),
          axios.get(`${API_URL}/prescriptions/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load prescriptions data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128138;</div>
      Loading prescriptions data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No prescriptions data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'drugs', label: 'Drug Analytics' },
    { id: 'patients', label: 'Patient Prescriptions' },
    { id: 'recent', label: 'Recent Activity' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const k = overview.kpis

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Prescriptions Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        {fmt(k.total_prescriptions)} prescriptions across {fmt(k.unique_drugs)} drugs | {fmt(k.unique_patients)} patients covered | Coverage: {fmt(k.prescription_coverage_pct)}%
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

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'drugs' && <DrugAnalyticsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientPrescriptionsTab breakdown={breakdown} />}
      {tab === 'recent' && <RecentActivityTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const k = overview.kpis
  const coverageColor = k.prescription_coverage_pct >= 50 ? '#10b981' : k.prescription_coverage_pct >= 25 ? '#f59e0b' : '#ef4444'
  const polyColor = k.polypharmacy_patients > 0 ? '#f59e0b' : '#10b981'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Prescriptions" value={k.total_prescriptions} color="#3b82f6" /></Card>
      <Card><KPI label="Unique Drugs" value={k.unique_drugs} color="#8b5cf6" /></Card>
      <Card><KPI label="Patients Covered" value={k.unique_patients} sub={`of ${k.total_patients} total`} color="#10b981" /></Card>
      <Card><KPI label="Coverage" value={`${fmt(k.prescription_coverage_pct)}%`} color={coverageColor} sub="patients with meds" /></Card>

      <Card><KPI label="Avg Drugs / Patient" value={k.avg_drugs_per_patient} color="#06b6d4" /></Card>
      <Card><KPI label="Polypharmacy Patients" value={k.polypharmacy_patients} color={polyColor} sub="2+ drugs" /></Card>
      <Card><KPI label="Most Common Drug" value={k.most_common_drug} color="#1e293b" /></Card>
      <Card><KPI label="Most Common Frequency" value={k.most_common_frequency} color="#1e293b" /></Card>

      {/* Drug distribution pie */}
      <Card title="Drug Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={overview.drug_distribution || []} dataKey="count" nameKey="drug_name"
              cx="50%" cy="50%" outerRadius={85} label={({ drug_name, percentage }) => `${drug_name} (${percentage}%)`}>
              {(overview.drug_distribution || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Frequency distribution bar */}
      <Card title="Frequency Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overview.frequency_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="frequency" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Dose distribution bar */}
      <Card title="Dose Distribution (mg)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.dose_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="dose_range" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Prescriptions by date bar */}
      <Card title="Prescriptions by Date" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.prescriptions_by_date || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DrugAnalyticsTab({ breakdown }) {
  const drugs = breakdown?.per_drug || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Per-drug bar chart */}
      <Card title="Times Prescribed by Drug">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={drugs}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="drug_name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="total_prescribed" fill="#10b981" radius={[4, 4, 0, 0]} name="Times Prescribed" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Per-drug table */}
      <Card title={`Drug Details (${drugs.length} drugs)`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Drug Name</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Times Prescribed</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Avg Dose (mg)</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Dose Range (mg)</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Patients</th>
            </tr></thead>
            <tbody>{drugs.map((d, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{d.drug_name}</td>
                <td style={{ padding: 8, textAlign: 'right', fontWeight: 600, color: '#10b981' }}>{fmt(d.total_prescribed)}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(d.avg_dose_mg)}</td>
                <td style={{ padding: 8, textAlign: 'right', fontSize: 11, color: '#64748b' }}>
                  {d.dose_range?.min != null ? `${d.dose_range.min} - ${d.dose_range.max}` : '--'}
                </td>
                <td style={{ padding: 8, fontSize: 11 }}>{(d.patients || []).join(', ')}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </Card>

      {/* AED Combinations */}
      {(breakdown?.aed_combinations || []).length > 0 && (
        <Card title="AED Combinations (Polytherapy)">
          <div style={{ display: 'grid', gap: 12 }}>
            {(breakdown.aed_combinations || []).map((c, i) => (
              <div key={i} style={{ padding: 12, background: '#fefce8', borderRadius: 8, border: '1px solid #fde68a' }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{c.patient_id}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  {c.count} AEDs: {c.aeds.join(' + ')}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

function PatientPrescriptionsTab({ breakdown }) {
  const patients = breakdown?.per_patient || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient Prescriptions (${patients.length} patients)`}>
        <div style={{ display: 'grid', gap: 16 }}>
          {patients.map((p, i) => (
            <div key={i} style={{
              padding: 16, background: '#f8fafc', borderRadius: 8,
              border: p.has_polypharmacy ? '2px solid #f59e0b' : '1px solid #e2e8f0'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <span style={{ fontWeight: 700, fontSize: 15, color: '#1e293b' }}>{p.patient_id}</span>
                <span style={{ fontSize: 12, color: '#64748b' }}>{p.total_drugs} drug{p.total_drugs !== 1 ? 's' : ''}</span>
                {p.has_polypharmacy && (
                  <span style={{
                    fontSize: 10, fontWeight: 600, color: '#92400e', background: '#fef3c7',
                    padding: '2px 8px', borderRadius: 10
                  }}>POLYPHARMACY</span>
                )}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
                {(p.drugs || []).map((d, j) => (
                  <div key={j} style={{
                    padding: 10, background: '#fff', borderRadius: 6,
                    border: '1px solid #e2e8f0', fontSize: 12
                  }}>
                    <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{d.drug_name}</div>
                    <div style={{ color: '#64748b' }}>
                      {d.dose_mg != null ? `${d.dose_mg} mg` : 'dose N/A'} | {d.frequency || 'freq N/A'}
                    </div>
                    <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
                      {d.created_at ? d.created_at.substring(0, 10) : ''}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function RecentActivityTab({ breakdown }) {
  const recent = breakdown?.recent_prescriptions || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Recent Prescriptions (last ${recent.length})`}>
        <div style={{ maxHeight: 600, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Drug</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Dose (mg)</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Frequency</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Date</th>
            </tr></thead>
            <tbody>{recent.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{r.patient_id}</td>
                <td style={{ padding: 8 }}>{r.drug_name}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{fmt(r.dose_mg)}</td>
                <td style={{ padding: 8 }}>{r.frequency}</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>
                  {r.created_at ? r.created_at.substring(0, 10) : '--'}
                </td>
              </tr>
            ))}</tbody>
          </table>
        </div>
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
