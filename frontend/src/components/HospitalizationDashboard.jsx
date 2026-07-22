import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const TYPE_COLORS = {
  emergency: '#ef4444',
  planned: '#3b82f6',
  transfer: '#f97316',
  observation: '#8b5cf6'
}
const TYPE_LABELS = {
  emergency: 'Emergency',
  planned: 'Planned',
  transfer: 'Transfer',
  observation: 'Observation'
}

const WARD_COLORS = {
  'Epilepsy Monitoring Unit': '#3b82f6',
  'Neurology Ward': '#22c55e',
  'ICU': '#ef4444',
  'Emergency': '#f97316',
  'Surgical Recovery': '#8b5cf6'
}

const DISPOSITION_COLORS = {
  home: '#22c55e',
  rehabilitation: '#3b82f6',
  transferred: '#f97316',
  ama: '#ef4444'
}
const DISPOSITION_LABELS = {
  home: 'Home',
  rehabilitation: 'Rehabilitation',
  transferred: 'Transferred',
  ama: 'AMA'
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function TypeBadge({ type }) {
  const color = TYPE_COLORS[type] || '#94a3b8'
  const label = TYPE_LABELS[type] || type || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function WardBadge({ ward }) {
  const color = WARD_COLORS[ward] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{ward || '--'}</span>
  )
}

export default function HospitalizationDashboard() {
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
          axios.get(`${API_URL}/api/hospitalization/overview`),
          axios.get(`${API_URL}/api/hospitalization/breakdown`),
          axios.get(`${API_URL}/api/hospitalization/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Hospitalization data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No hospitalization data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'admissions', label: 'Admissions' },
    { id: 'patients', label: 'Patients' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const typeDistData = overview?.admission_type_distribution
    ? Object.entries(overview.admission_type_distribution).map(([k, v]) => ({
        name: TYPE_LABELS[k] || k, value: v, color: TYPE_COLORS[k] || '#94a3b8'
      }))
    : []

  const wardDistData = overview?.ward_distribution
    ? Object.entries(overview.ward_distribution).map(([k, v]) => ({
        name: k, count: v, color: WARD_COLORS[k] || '#94a3b8'
      }))
    : []

  const reasonDistData = overview?.admission_reason_distribution
    ? Object.entries(overview.admission_reason_distribution).map(([k, v]) => ({
        name: k.replace(/_/g, ' '), count: v
      })).sort((a, b) => b.count - a.count)
    : []

  const dispositionData = overview?.disposition_distribution
    ? Object.entries(overview.disposition_distribution).map(([k, v]) => ({
        name: DISPOSITION_LABELS[k] || k, value: v, color: DISPOSITION_COLORS[k] || '#94a3b8'
      }))
    : []

  const insuranceData = overview?.insurance_distribution
    ? Object.entries(overview.insurance_distribution).map(([k, v]) => ({
        name: k.replace(/_/g, ' '), count: v
      }))
    : []

  const timelineData = overview?.monthly_timeline || []

  /* Breakdown data prep */
  const perPatient = breakdown?.per_patient || []
  const currentlyAdmitted = breakdown?.currently_admitted || []
  const recentDischarges = breakdown?.recent_discharges || []
  const physicianStats = breakdown?.physician_stats || []
  const complicationData = breakdown?.complication_summary || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Hospitalization Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Admission tracking, length of stay, readmission analytics, and discharge outcomes
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Admissions" value={fmt(overview.total_admissions)} />
              <KPI label="Currently Admitted" value={fmt(overview.currently_admitted)} color="#f97316" />
              <KPI label="Avg LOS (days)" value={fmt(overview.avg_length_of_stay_days)} />
              <KPI label="Readmission Rate" value={overview.readmission_rate_pct != null ? fmt(overview.readmission_rate_pct) + '%' : '--'} color="#ef4444" />
              <KPI label="Seizure-Free Discharge" value={overview.seizure_free_discharge_rate_pct != null ? fmt(overview.seizure_free_discharge_rate_pct) + '%' : '--'} color="#22c55e" />
              <KPI label="Avg Cost" value={overview.avg_cost_per_admission != null ? '$' + fmt(Math.round(overview.avg_cost_per_admission)) : '--'} />
            </div>
          </Card>

          {/* Pie: Admission Type */}
          <Card title="Admission Type">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={typeDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {typeDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {typeDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Ward Distribution */}
          <Card title="Ward Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={wardDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Admissions">
                  {wardDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: Monthly Timeline */}
          <Card title="Monthly Trend (12 months)" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="admissions" stroke="#3b82f6" name="Admissions" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="discharges" stroke="#22c55e" name="Discharges" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="avg_los" stroke="#f97316" name="Avg LOS" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Admissions */}
      {tab === 'admissions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Currently Admitted Alert */}
          <Card title={`Currently Admitted (${currentlyAdmitted.length})`} span={2}>
            {currentlyAdmitted.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No patients currently admitted.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Admitted</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Ward</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Reason</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Physician</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Days So Far</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentlyAdmitted.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef3c7' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.admission_date || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><WardBadge ward={row.ward} /></td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{(row.admission_reason || '').replace(/_/g, ' ')}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.attending_physician || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 600, color: '#f97316' }}>{fmt(row.days_so_far)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Bar: Admission Reason */}
          <Card title="Admission Reasons" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(200, reasonDistData.length * 30)}>
              <BarChart data={reasonDistData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={180} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" name="Admissions" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Recent Discharges Table */}
          <Card title="Recent Discharges" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Admitted</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Discharged</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Ward</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>LOS</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Disposition</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizure-Free</th>
                  </tr>
                </thead>
                <tbody>
                  {recentDischarges.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.admission_date || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.discharge_date || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><WardBadge ward={row.ward} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 600 }}>{fmt(row.los_days)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                          background: (DISPOSITION_COLORS[row.disposition] || '#94a3b8') + '22',
                          color: DISPOSITION_COLORS[row.disposition] || '#94a3b8'
                        }}>{DISPOSITION_LABELS[row.disposition] || row.disposition || '--'}</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 16 }}>
                        {row.seizure_free === true || row.seizure_free === 1 ? '✓' : row.seizure_free === false || row.seizure_free === 0 ? '✗' : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Patients */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Patient Summary */}
          <Card title="Per-Patient Summary" span={2}>
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Admissions</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total Days</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg LOS</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Readmissions</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizure-Free %</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px', color: '#64748b' }}>Total Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatient.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.total_admissions)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.total_days)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.avg_los)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: row.readmissions > 0 ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
                        {fmt(row.readmissions)}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                          <div style={{ width: 60, height: 6, background: '#e2e8f0', borderRadius: 3 }}>
                            <div style={{ width: `${Math.min(100, row.seizure_free_rate || 0)}%`, height: 6, background: '#22c55e', borderRadius: 3 }} />
                          </div>
                          <span style={{ fontSize: 11 }}>{fmt(row.seizure_free_rate)}%</span>
                        </div>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'right', fontSize: 11 }}>${fmt(row.total_cost)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Physician Stats */}
          <Card title="Physician Statistics" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={physicianStats}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="physician" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="total_patients" fill="#3b82f6" name="Patients" />
                <Bar dataKey="avg_los" fill="#f97316" name="Avg LOS" />
                <Bar dataKey="seizure_free_rate" fill="#22c55e" name="Seizure-Free %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 4: Analytics */}
      {tab === 'analytics' && overview && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Pie: Disposition */}
          <Card title="Discharge Disposition">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={dispositionData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {dispositionData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {dispositionData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Insurance Type */}
          <Card title="Insurance Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={insuranceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#06b6d4" name="Admissions" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Complications */}
          <Card title="Complications" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={complicationData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="complication" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly Trend */}
          <Card title="Monthly Admissions & LOS Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="admissions" stroke="#3b82f6" name="Admissions" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="discharges" stroke="#22c55e" name="Discharges" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="avg_los" stroke="#f97316" name="Avg LOS" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Admission Types */}
          {defs.admission_types && (
            <Card title="Admission Types">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {Object.entries(defs.admission_types).map(([k, v], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: TYPE_COLORS[k] || '#334155', width: 150 }}>
                        {TYPE_LABELS[k] || k}
                      </td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Admission Reasons */}
          {defs.admission_reasons && (
            <Card title="Admission Reasons">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {Object.entries(defs.admission_reasons).map(([k, v], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155', width: 220 }}>{k.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Wards */}
          {defs.wards && (
            <Card title="Ward Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.wards).map(([k, v], i) => {
                  const color = WARD_COLORS[k] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: color + '11', border: `1px solid ${color}33`, borderRadius: 8 }}>
                      <h4 style={{ margin: '0 0 6px', fontSize: 13, color }}>{k}</h4>
                      <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{v}</p>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Disposition Types */}
          {defs.disposition_types && (
            <Card title="Discharge Disposition Types">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {Object.entries(defs.disposition_types).map(([k, v], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, color: DISPOSITION_COLORS[k] || '#334155', width: 150 }}>
                        {DISPOSITION_LABELS[k] || k}
                      </td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Clinical Glossary">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.glossary.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.term}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
