import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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

export default function PatientDashboardPanel() {
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
          axios.get(`${API_URL}/api/patient-dashboard/overview`),
          axios.get(`${API_URL}/api/patient-dashboard/breakdown`),
          axios.get(`${API_URL}/api/patient-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load patient dashboard data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128202;</div>
      Loading patient dashboard...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No patient data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'assessments', label: 'Assessments' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'trends', label: 'Trends' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const ov = overview
  const bd = breakdown || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>Patient Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Cross-domain KPIs and trends for {fmt(ov.total_patients)} patients
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#1e293b' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Patients" value={ov.total_patients} color="#3b82f6" /></Card>
            <Card><KPI label="Avg Age" value={ov.avg_age} sub="years" /></Card>
            <Card><KPI label="Assessments" value={ov.total_assessments} sub={`${ov.unique_instruments} instruments`} color="#10b981" /></Card>
            <Card><KPI label="Assessment Coverage" value={`${ov.assessment_coverage_pct}%`} sub={`${ov.patients_with_assessments} of ${ov.total_patients}`} color="#8b5cf6" /></Card>
            <Card><KPI label="Appointments" value={ov.total_appointments} sub={`${ov.completion_rate_pct}% completed`} /></Card>
            <Card><KPI label="Prescriptions" value={ov.total_prescriptions} sub={`${ov.medication_coverage_pct}% coverage`} color="#f59e0b" /></Card>
            <Card><KPI label="Seizure Events" value={ov.total_seizure_events} sub={`${ov.patients_with_seizures} patients`} color="#ef4444" /></Card>
            <Card><KPI label="EEG Analyses" value={ov.total_analyses} sub={`${ov.avg_confidence_pct}% avg confidence`} color="#06b6d4" /></Card>
          </div>

          {/* Gender + Disease charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Gender Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={ov.gender_distribution.map(g => ({ ...g, name: g.gender || 'Unspecified' }))}
                    dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                    {ov.gender_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Disease Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ov.disease_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Seizure severity + Analysis distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {bd.seizure_severity && bd.seizure_severity.length > 0 && (
              <Card title="Seizure Severity">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={bd.seizure_severity} dataKey="count" nameKey="severity"
                      cx="50%" cy="50%" outerRadius={80} label={({ severity, count }) => `${severity}: ${count}`}>
                      {bd.seizure_severity.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            )}
            {bd.analysis_distribution && bd.analysis_distribution.length > 0 && (
              <Card title="EEG Analysis Results">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={bd.analysis_distribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* ── ASSESSMENTS TAB ── */}
      {tab === 'assessments' && (
        <div>
          {/* Instrument stats */}
          {bd.instrument_stats && bd.instrument_stats.length > 0 && (
            <Card title="Assessment by Instrument" span={2}>
              <ResponsiveContainer width="100%" height={Math.max(250, bd.instrument_stats.length * 30)}>
                <BarChart data={bd.instrument_stats} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="instrument" type="category" width={120} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Severity distribution */}
          {bd.severity_distribution && bd.severity_distribution.length > 0 && (
            <Card title="Severity Distribution" style={{ marginTop: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginTop: 16 }}>
                {bd.severity_distribution.map((s, i) => (
                  <Card key={i}>
                    <KPI label={s.level} value={s.count}
                      color={s.level === 'severe' ? '#ef4444' : s.level === 'moderate' ? '#f59e0b' : '#10b981'} />
                  </Card>
                ))}
              </div>
            </Card>
          )}

          {/* Instrument avg score table */}
          {bd.instrument_stats && bd.instrument_stats.length > 0 && (
            <Card title="Instrument Average Scores" style={{ marginTop: 16 }}>
              <div style={{ overflowX: 'auto', marginTop: 16 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Instrument</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Assessments</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Score %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.instrument_stats.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px' }}>{r.instrument}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(r.count)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(r.avg_score_pct)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Recent assessments */}
          {bd.recent_assessments && bd.recent_assessments.length > 0 && (
            <Card title="Recent Assessments (last 20)" style={{ marginTop: 16 }}>
              <div style={{ overflowX: 'auto', marginTop: 16 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Patient</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Instrument</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Score</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Max</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Level</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.recent_assessments.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px' }}>{r.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>{r.instrument}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(r.score)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(r.max_score)}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                            background: r.level === 'severe' ? '#fee2e2' : r.level === 'moderate' ? '#fef3c7' : '#ecfdf5',
                            color: r.level === 'severe' ? '#dc2626' : r.level === 'moderate' ? '#d97706' : '#059669',
                          }}>{r.level}</span>
                        </td>
                        <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{r.created_at}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── PATIENT PROFILES TAB ── */}
      {tab === 'patients' && (
        <div>
          {bd.patient_summary && bd.patient_summary.length > 0 && (
            <Card title={`Patient Profiles (${bd.patient_summary.length} most active)`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Patient</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Name</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Age</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gender</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Disease</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Assessments</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Visits</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Seizures</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Rx</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.patient_summary.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 11 }}>{p.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>{p.name || '—'}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.age)}</td>
                        <td style={{ padding: '8px 12px' }}>{p.gender || '—'}</td>
                        <td style={{ padding: '8px 12px' }}>{p.disease}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.assessments)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.completed_visits)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: p.seizure_events > 3 ? '#ef4444' : undefined, fontWeight: p.seizure_events > 3 ? 700 : undefined }}>{fmt(p.seizure_events)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(p.prescriptions)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── TRENDS TAB ── */}
      {tab === 'trends' && (
        <div>
          {bd.daily_appointment_trend && bd.daily_appointment_trend.length > 0 && (
            <Card title="Daily Appointment Trend">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={bd.daily_appointment_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="total" stroke="#3b82f6" name="Total" strokeWidth={2} />
                  <Line type="monotone" dataKey="completed" stroke="#10b981" name="Completed" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Seizure triggers */}
          {bd.seizure_triggers && bd.seizure_triggers.length > 0 && (
            <Card title="Seizure Triggers" style={{ marginTop: 16 }}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={bd.seizure_triggers}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="trigger" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Disease stats table */}
          {bd.disease_stats && bd.disease_stats.length > 0 && (
            <Card title="Per-Disease Statistics" style={{ marginTop: 16 }}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Disease</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Patients</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Avg Age</th>
                      <th style={{ padding: '8px 12px', textAlign: 'right' }}>Assessments</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.disease_stats.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px' }}>{d.disease}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.patients)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.avg_age)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.assessments)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div>
          {defs.sections && defs.sections.map((sec, si) => (
            <Card key={si} title={sec.title} style={{ marginBottom: 16 }}>
              <div style={{ marginTop: si > 0 ? 16 : 0 }}>
                {sec.items.map((item, ii) => (
                  <div key={ii} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: ii < sec.items.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4, lineHeight: 1.5 }}>{item.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
