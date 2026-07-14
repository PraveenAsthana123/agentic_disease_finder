import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
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

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${(v * 100).toFixed(1)}%` : '--')
const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'demographics', label: 'Demographics' },
  { id: 'seizure_epi', label: 'Seizure Epidemiology' },
  { id: 'registry', label: 'Patient Registry' },
  { id: 'definitions', label: 'Definitions' },
]

const RISK_COLORS = {
  high: { bg: '#fee2e2', text: '#991b1b' },
  moderate: { bg: '#fef3c7', text: '#92400e' },
  low: { bg: '#dcfce7', text: '#166534' },
}

export default function PopulationHealthDashboard() {
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
      axios.get(`${API_URL}/api/population-health/overview`),
      axios.get(`${API_URL}/api/population-health/breakdown`),
      axios.get(`${API_URL}/api/population-health/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Population Health Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Population Health Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Population-level epilepsy analytics — demographics, seizure epidemiology, patient registry, data coverage
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Total Patients" value={fmt(ov?.total_patients)} sub="Enrolled" color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Mean Age" value={fmt(ov?.age_stats?.mean)} sub="Years" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Seizure Events" value={fmt(ov?.seizure_burden?.total_events)} sub="Total recorded" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Medications" value={fmt(ov?.medication_coverage?.total_prescriptions)} sub="Total prescriptions" color="#10b981" />
          </Card>

          {/* Age Group Distribution */}
          <Card title="Age Group Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={ov?.age_groups || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="group" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" fill="#3b82f6">
                  {(ov?.age_groups || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Gender Distribution */}
          <Card title="Gender Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={(ov?.gender_distribution || []).filter(d => d.count > 0)}
                  dataKey="count"
                  nameKey="gender"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.gender_distribution || []).filter(d => d.count > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Seizure Severity */}
          <Card title="Seizure Severity">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={(ov?.seizure_burden?.severity_distribution || []).filter(d => d.count > 0)}
                  dataKey="count"
                  nameKey="severity"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(ov?.seizure_burden?.severity_distribution || []).filter(d => d.count > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Top Comorbidity Prevalence */}
          <Card title="Top Comorbidity Prevalence" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={(ov?.comorbidity_prevalence || []).slice(0, 8)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="condition" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Prevalence" fill="#3b82f6">
                  {(ov?.comorbidity_prevalence || []).slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* AED Drug Distribution */}
          <Card title="AED Drug Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={ov?.medication_coverage?.drug_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="drug" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Prescriptions" fill="#10b981">
                  {(ov?.medication_coverage?.drug_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Data Coverage */}
          <Card title="Data Coverage" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
              {Object.entries(ov?.data_coverage || {}).map(([key, val]) => (
                <div key={key} style={{
                  background: '#f1f5f9', borderRadius: 8, padding: '10px 12px', textAlign: 'center'
                }}>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#1e293b' }}>{fmt(val)}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{key.replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Enrollment Trend */}
          <Card title="Enrollment Trend" span={4}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={ov?.enrollment_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="count" name="Enrollments" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── DEMOGRAPHICS TAB ─── */}
      {tab === 'demographics' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Age-Sex Pyramid */}
          <Card title="Age-Sex Pyramid" span={2}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={bd?.age_sex_pyramid || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="age_group" type="category" width={80} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="male" name="Male" stackId="a" fill="#3b82f6" />
                <Bar dataKey="female" name="Female" stackId="a" fill="#ec4899" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Department Distribution */}
          <Card title="Department Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={(bd?.geographic_distribution || []).filter(d => d.count > 0)}
                  dataKey="count"
                  nameKey="department"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(bd?.geographic_distribution || []).filter(d => d.count > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk Stratification Table */}
          <Card title="Risk Stratification" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Patient ID</th>
                    <th style={{ padding: '8px 12px' }}>Name</th>
                    <th style={{ padding: '8px 12px' }}>Risk Level</th>
                    <th style={{ padding: '8px 12px' }}>Factors</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.risk_stratification || []).map((p, i) => {
                    const rl = (p.risk_level || '').toLowerCase()
                    const rlStyle = RISK_COLORS[rl] || { bg: '#f1f5f9', text: '#475569' }
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>{p.name ?? '--'}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                            background: rlStyle.bg, color: rlStyle.text,
                          }}>{p.risk_level ?? '--'}</span>
                        </td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#475569' }}>
                          {Array.isArray(p.factors) ? p.factors.join(', ') : (p.factors ?? '--')}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── SEIZURE EPIDEMIOLOGY TAB ─── */}
      {tab === 'seizure_epi' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Total Events" value={fmt(bd?.seizure_characteristics?.total_events)} sub="Recorded seizures" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Mean per Patient" value={fmt(bd?.seizure_characteristics?.mean_per_patient)} sub="Average seizure count" color="#8b5cf6" />
          </Card>
          <Card span={2} />

          {/* Trigger Distribution */}
          <Card title="Trigger Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bd?.seizure_characteristics?.trigger_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="trigger" type="category" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Occurrences" fill="#ef4444">
                  {(bd?.seizure_characteristics?.trigger_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Awareness Distribution */}
          <Card title="Awareness Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={(bd?.seizure_characteristics?.awareness_distribution || []).filter(d => d.count > 0)}
                  dataKey="count"
                  nameKey="awareness"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {(bd?.seizure_characteristics?.awareness_distribution || []).filter(d => d.count > 0).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Rate tiles */}
          {['aura_rate', 'injury_rate', 'er_visit_rate', 'witnessed_rate'].map((key) => (
            <Card key={key}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>
                  {pct(bd?.seizure_characteristics?.[key])}
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
                  {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ─── PATIENT REGISTRY TAB ─── */}
      {tab === 'registry' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Patient Registry" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>ID</th>
                    <th style={{ padding: '8px 12px' }}>Name</th>
                    <th style={{ padding: '8px 12px' }}>Age</th>
                    <th style={{ padding: '8px 12px' }}>Gender</th>
                    <th style={{ padding: '8px 12px' }}>Seizures</th>
                    <th style={{ padding: '8px 12px' }}>Comorbidities</th>
                    <th style={{ padding: '8px 12px' }}>Medication</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd?.patient_registry || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontWeight: 600 }}>{p.id ?? p.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{p.name ?? '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(p.age)}</td>
                      <td style={{ padding: '8px 12px' }}>{p.gender ?? '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <span style={{
                          fontWeight: 600,
                          color: (p.seizures > 0) ? '#ef4444' : '#10b981',
                        }}>{fmt(p.seizures)}</span>
                      </td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#475569', maxWidth: 220 }}>
                        {Array.isArray(p.comorbidities) ? p.comorbidities.join(', ') : (p.comorbidities ?? '--')}
                      </td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#475569' }}>
                        {Array.isArray(p.medication) ? p.medication.join(', ') : (p.medication ?? '--')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Population Health Terminology" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px', width: 220 }}>Term</th>
                  <th style={{ padding: '8px 12px' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs?.terms || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{t.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{t.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Data Sources" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px', width: 180 }}>Source</th>
                  <th style={{ padding: '8px 12px', width: 80 }}>Rows</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs?.data_sources || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{s.source}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(s.rows)}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {defs?.methodology && (
            <Card title="Methodology" span={2}>
              <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.7, margin: 0 }}>{defs.methodology}</p>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
