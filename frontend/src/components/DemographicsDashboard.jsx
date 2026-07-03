import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'demographics_analysis', label: 'Demographics Analysis' },
  { id: 'clinical_profile', label: 'Clinical Profile' },
  { id: 'patient_detail', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DemographicsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/demographics/overview`),
      axios.get(`${API_URL}/api/demographics/breakdown`),
      axios.get(`${API_URL}/api/demographics/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Demographics Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  /* Client-side aggregation for marital status from breakdown patients */
  const maritalCounts = {}
  ;(bd.patients || []).forEach(p => {
    const status = p.marital_status || 'Unknown'
    maritalCounts[status] = (maritalCounts[status] || 0) + 1
  })
  const maritalData = Object.entries(maritalCounts).map(([status, count]) => ({ status, count }))

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Demographics Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Patient demographics — age, sex, epilepsy type, insurance, BMI, ethnicity, education, and clinical profile
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total Patients"><KPI value={ov.total_patients} label="Patients enrolled" color="#3b82f6" /></Card>
          <Card title="Average Age"><KPI value={ov.avg_age != null ? ov.avg_age.toFixed(1) : '--'} label="Years" color="#8b5cf6" /></Card>
          <Card title="Male"><KPI value={ov.male_pct != null ? `${ov.male_pct.toFixed(1)}%` : '--'} label="Male patients" color="#06b6d4" /></Card>
          <Card title="Female"><KPI value={ov.female_pct != null ? `${ov.female_pct.toFixed(1)}%` : '--'} label="Female patients" color="#ec4899" /></Card>

          {/* KPI Row 2 */}
          <Card title="Average BMI"><KPI value={ov.avg_bmi != null ? ov.avg_bmi.toFixed(1) : '--'} label="Body mass index" color="#f59e0b" /></Card>
          <Card title="Interpreter Needed"><KPI value={ov.interpreter_needed_pct != null ? `${ov.interpreter_needed_pct.toFixed(1)}%` : '--'} label="Require interpreter" color="#e74c3c" /></Card>
          <Card title="Avg Years w/ Epilepsy"><KPI value={ov.avg_years_with_epilepsy != null ? ov.avg_years_with_epilepsy.toFixed(1) : '--'} label="Years with epilepsy" color="#10b981" /></Card>
          <Card title="Most Common Type"><KPI value={ov.most_common_epilepsy_type || '--'} label="Epilepsy type" color="#f97316" /></Card>

          {/* Age Distribution Bar Chart */}
          <Card title="Age Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.age_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.age_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Sex Distribution Pie Chart */}
          <Card title="Sex Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.sex_distribution || []} dataKey="count" nameKey="sex" cx="50%" cy="50%" outerRadius={80} label={e => `${e.sex}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.sex_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Epilepsy Type Distribution Bar Chart */}
          <Card title="Epilepsy Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.epilepsy_type_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.epilepsy_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Insurance Distribution Pie Chart */}
          <Card title="Insurance Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.insurance_distribution || []} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={80} label={e => `${e.type}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.insurance_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* BMI Categories Bar Chart */}
          <Card title="BMI Categories" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.bmi_categories || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(ov.bmi_categories || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* DEMOGRAPHICS ANALYSIS TAB */}
      {tab === 'demographics_analysis' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Ethnicity Distribution Bar Chart */}
          <Card title="Ethnicity Distribution" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.ethnicity_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="ethnicity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(bd.ethnicity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Race Distribution Bar Chart */}
          <Card title="Race Distribution" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.race_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="race" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(bd.race_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Language Distribution Pie Chart */}
          <Card title="Language Distribution" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={bd.language_distribution || []} dataKey="count" nameKey="language" cx="50%" cy="50%" outerRadius={80} label={e => `${e.language}: ${e.count}`} labelLine fontSize={11}>
                  {(bd.language_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Education Distribution Bar Chart */}
          <Card title="Education Distribution" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.education_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="education" fontSize={11} angle={-20} textAnchor="end" height={60} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(bd.education_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Employment Distribution Bar Chart */}
          <Card title="Employment Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.employment_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="employment" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(bd.employment_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* CLINICAL PROFILE TAB */}
      {tab === 'clinical_profile' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* Age Stats KPIs */}
          <Card title="Age — Min"><KPI value={ov.age_stats?.min != null ? ov.age_stats.min : '--'} label="Minimum age" color="#3b82f6" /></Card>
          <Card title="Age — Max"><KPI value={ov.age_stats?.max != null ? ov.age_stats.max : '--'} label="Maximum age" color="#8b5cf6" /></Card>
          <Card title="Age — Mean"><KPI value={ov.age_stats?.mean != null ? ov.age_stats.mean.toFixed(1) : '--'} label="Mean age" color="#06b6d4" /></Card>
          <Card title="Age — Median"><KPI value={ov.age_stats?.median != null ? ov.age_stats.median : '--'} label="Median age" color="#10b981" /></Card>

          {/* Epilepsy Onset Stats KPIs */}
          <Card title="Onset — Min"><KPI value={ov.epilepsy_onset_stats?.min != null ? ov.epilepsy_onset_stats.min : '--'} label="Min onset age" color="#f59e0b" /></Card>
          <Card title="Onset — Max"><KPI value={ov.epilepsy_onset_stats?.max != null ? ov.epilepsy_onset_stats.max : '--'} label="Max onset age" color="#e74c3c" /></Card>
          <Card title="Onset — Mean"><KPI value={ov.epilepsy_onset_stats?.mean != null ? ov.epilepsy_onset_stats.mean.toFixed(1) : '--'} label="Mean onset age" color="#ec4899" /></Card>
          <Card title="Onset — Median"><KPI value={ov.epilepsy_onset_stats?.median != null ? ov.epilepsy_onset_stats.median : '--'} label="Median onset age" color="#f97316" /></Card>

          {/* Referral Sources Bar Chart */}
          <Card title="Referral Sources" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.referral_sources || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" fontSize={11} angle={-20} textAnchor="end" height={60} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {(bd.referral_sources || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Marital Status Distribution Bar Chart */}
          <Card title="Marital Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={maritalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="status" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {maritalData.map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patient_detail' && (
        <Card title="Patient Demographics">
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9', position: 'sticky', top: 0 }}>
                  {['Patient ID', 'Full Name', 'Age', 'Sex', 'Epilepsy Type', 'Years w/ Epilepsy', 'Insurance', 'Employment'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.patient_id)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.full_name)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.age)}</td>
                    <td style={{ padding: '6px' }}>
                      <Badge text={p.sex || '--'} color={p.sex === 'Male' ? '#3b82f6' : p.sex === 'Female' ? '#ec4899' : '#64748b'} />
                    </td>
                    <td style={{ padding: '6px' }}>{fmt(p.epilepsy_type)}</td>
                    <td style={{ padding: '6px' }}>{p.years_with_epilepsy != null ? p.years_with_epilepsy.toFixed(1) : '--'}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.insurance_type)}</td>
                    <td style={{ padding: '6px' }}>{fmt(p.employment_status)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Demographics Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
