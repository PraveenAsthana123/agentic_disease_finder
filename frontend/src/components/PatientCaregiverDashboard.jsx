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

const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6']

const badgeStyle = (color) => ({
  display: 'inline-block', padding: '2px 8px', borderRadius: 12, fontSize: 11, fontWeight: 600,
  background: color === 'green' ? '#f0fdf4' : color === 'blue' ? '#eff6ff' : color === 'red' ? '#fef2f2' : color === 'amber' ? '#fffbeb' : '#f1f5f9',
  color: color === 'green' ? '#16a34a' : color === 'blue' ? '#2563eb' : color === 'red' ? '#dc2626' : color === 'amber' ? '#d97706' : '#475569',
})

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
const thStyle = { textAlign: 'left', padding: 8, borderBottom: '2px solid #e2e8f0' }
const tdStyle = { padding: 8, borderBottom: '1px solid #f1f5f9', color: '#475569' }

const SEV_BADGE = {
  Mild: 'green',
  Moderate: 'amber',
  Severe: 'red',
}

const STATUS_BADGE = {
  completed: 'green',
  booked: 'blue',
  cancelled: 'red',
}

export default function PatientCaregiverDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/patient-caregiver/overview`),
      axios.get(`${API_URL}/api/patient-caregiver/breakdown`),
      axios.get(`${API_URL}/api/patient-caregiver/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Patient &amp; Caregiver dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'seizure_diary', label: 'Seizure Diary' },
    { id: 'mood', label: 'Mood & Wellbeing' },
    { id: 'medications', label: 'Medications' },
    { id: 'appointments', label: 'Appointments' },
    { id: 'profiles', label: 'Patient Profiles' },
    { id: 'alerts', label: 'Risk Alerts' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        Patient &amp; Caregiver Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Seizure tracking, mood assessment, quality of life, and self-management tools
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#0f766e' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Patients" value={k.total_patients} color="#0f766e" /></Card>
            <Card><KPI label="Seizure Events" value={k.seizure_events} color="#ef4444" /></Card>
            <Card><KPI label="Total Assessments" value={k.total_assessments} /></Card>
            <Card><KPI label="Total Medications" value={k.total_medications} /></Card>
            <Card><KPI label="Appointments" value={k.appointments} /></Card>
            <Card><KPI label="Avg Severity" value={k.avg_severity} color="#f59e0b" /></Card>
            <Card><KPI label="QoL Average" value={k.qol_average} color="#8b5cf6" /></Card>
            <Card><KPI label="Mood Score Avg" value={k.mood_score_avg} color="#06b6d4" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <Card title="Seizure Severity Distribution">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.seizure_summary || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="severity" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Trigger Distribution">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={(overview?.trigger_distribution || []).filter(d => d.count > 0)}
                       dataKey="count" nameKey="trigger" cx="50%" cy="50%"
                       outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview?.trigger_distribution || []).filter(d => d.count > 0).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Appointment Status">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={(overview?.appointment_status || []).filter(d => d.count > 0)}
                       dataKey="count" nameKey="status" cx="50%" cy="50%"
                       outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview?.appointment_status || []).filter(d => d.count > 0).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* Seizure Diary Tab */}
      {tab === 'seizure_diary' && (
        <>
          <Card title="Seizure Timeline">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={breakdown?.seizure_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" height={60} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ marginTop: 16 }}>
            <Card title={`Seizure Diary (${(breakdown?.seizure_diary || []).length} entries)`}>
              <div style={{ maxHeight: 500, overflowY: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Patient</th>
                      <th style={thStyle}>Date</th>
                      <th style={thStyle}>Duration (s)</th>
                      <th style={thStyle}>Severity</th>
                      <th style={thStyle}>Trigger</th>
                      <th style={thStyle}>Aura</th>
                      <th style={thStyle}>Injury</th>
                      <th style={thStyle}>ER Visit</th>
                      <th style={thStyle}>Rescue Med</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown?.seizure_diary || []).map((r, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{r.patient_id}</td>
                        <td style={tdStyle}>{r.event_date}</td>
                        <td style={{ ...tdStyle, textAlign: 'center' }}>{r.duration_sec ?? '--'}</td>
                        <td style={tdStyle}>
                          <span style={badgeStyle(SEV_BADGE[r.severity] || 'gray')}>{r.severity}</span>
                        </td>
                        <td style={tdStyle}>{r.trigger || '--'}</td>
                        <td style={tdStyle}>{r.aura ? 'Yes' : 'No'}</td>
                        <td style={tdStyle}>{r.injury ? 'Yes' : 'No'}</td>
                        <td style={tdStyle}>{r.er_visit ? 'Yes' : 'No'}</td>
                        <td style={tdStyle}>{r.rescue_med ? 'Yes' : 'No'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {/* Mood & Wellbeing Tab */}
      {tab === 'mood' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card>
              <KPI label="PHQ-9 Average" value={overview?.mood_overview?.phq9_avg} color="#ef4444" sub="Depression screening" />
            </Card>
            <Card>
              <KPI label="GAD-7 Average" value={overview?.mood_overview?.gad7_avg} color="#f59e0b" sub="Anxiety screening" />
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="PHQ-9 Level Distribution (Depression)">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.mood_overview?.phq9_levels || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="GAD-7 Level Distribution (Anxiety)">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.mood_overview?.gad7_levels || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Quality of Life Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={overview?.qol_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* Medications Tab */}
      {tab === 'medications' && (
        <Card title={`Medication List (${(breakdown?.medication_list || []).length} records)`}>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Drug Name</th>
                  <th style={thStyle}>Dose (mg)</th>
                  <th style={thStyle}>Frequency</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.medication_list || []).map((r, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{r.patient_id}</td>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{r.drug_name}</td>
                    <td style={{ ...tdStyle, textAlign: 'center' }}>{r.dose_mg ?? '--'}</td>
                    <td style={tdStyle}>{r.frequency || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Appointments Tab */}
      {tab === 'appointments' && (
        <Card title={`Appointments (${(breakdown?.appointment_list || []).length} records)`}>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient</th>
                  <th style={thStyle}>Provider</th>
                  <th style={thStyle}>Department</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Scheduled For</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.appointment_list || []).map((r, i) => (
                  <tr key={i}>
                    <td style={tdStyle}>{r.patient_id}</td>
                    <td style={tdStyle}>{r.provider || '--'}</td>
                    <td style={tdStyle}>{r.department || '--'}</td>
                    <td style={tdStyle}>{r.appt_type || '--'}</td>
                    <td style={tdStyle}>
                      <span style={badgeStyle(STATUS_BADGE[r.status] || 'gray')}>{r.status}</span>
                    </td>
                    <td style={tdStyle}>{r.scheduled_for || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Patient Profiles Tab */}
      {tab === 'profiles' && (
        <Card title={`Patient Profiles (${(breakdown?.patient_profiles || []).length} patients)`}>
          <div style={{ maxHeight: 600, overflowY: 'auto' }}>
            {(breakdown?.patient_profiles || []).map((p, i) => (
              <div key={i} style={{
                padding: 14, marginBottom: 10, borderRadius: 10,
                border: '1px solid #e2e8f0', background: '#fafafa',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                  <div>
                    <span style={{ fontWeight: 700, fontSize: 14 }}>{p.name || p.patient_id}</span>
                    <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                      {p.age ? `${p.age}y` : ''}{p.gender ? `, ${p.gender}` : ''}{p.disease ? ` | ${p.disease}` : ''}
                    </span>
                  </div>
                  <span style={{ fontSize: 12, color: '#64748b' }}>ID: {p.patient_id}</span>
                </div>

                <div style={{ display: 'flex', gap: 16, fontSize: 12, marginBottom: 8 }}>
                  <span>Seizures: <strong style={{ color: (p.seizure_count || 0) >= 3 ? '#dc2626' : '#475569' }}>{p.seizure_count ?? '--'}</strong></span>
                  <span>PHQ-9: <strong>{p.phq9_score ?? '--'}</strong></span>
                  <span>GAD-7: <strong>{p.gad7_score ?? '--'}</strong></span>
                  <span>QOLIE: <strong>{p.qolie_score ?? '--'}</strong></span>
                </div>

                {(p.medications || []).length > 0 && (
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                    <strong>Medications:</strong> {(p.medications || []).join(', ')}
                  </div>
                )}

                {(p.risk_factors || []).length > 0 && (
                  <div style={{ fontSize: 11, marginTop: 4 }}>
                    {(p.risk_factors || []).map((rf, j) => (
                      <span key={j} style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 8, margin: '1px 2px',
                        background: '#fef2f2', color: '#dc2626', fontSize: 10,
                      }}>{rf}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Risk Alerts Tab */}
      {tab === 'alerts' && (() => {
        const atRisk = (breakdown?.patient_profiles || []).filter(p => (p.risk_factors || []).length > 0)
        return (
          <Card title={`Risk Alerts (${atRisk.length} patients with risk factors)`}>
            <div style={{ maxHeight: 600, overflowY: 'auto' }}>
              {atRisk.length === 0 && (
                <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No risk alerts detected</div>
              )}
              {atRisk.sort((a, b) => (b.risk_factors || []).length - (a.risk_factors || []).length).map((p, i) => (
                <div key={i} style={{
                  padding: 14, marginBottom: 10, borderRadius: 10,
                  border: '1px solid #fecaca', background: '#fef2f2',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 14 }}>{p.name || p.patient_id}</span>
                      <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                        {p.age ? `${p.age}y` : ''}{p.gender ? `, ${p.gender}` : ''}
                      </span>
                    </div>
                    <span style={{
                      padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600,
                      background: (p.risk_factors || []).length >= 3 ? '#dc2626' : '#f59e0b',
                      color: '#fff',
                    }}>{(p.risk_factors || []).length} risk factors</span>
                  </div>

                  <div style={{ display: 'flex', gap: 16, fontSize: 12, marginBottom: 8 }}>
                    <span>Seizures: <strong>{p.seizure_count ?? '--'}</strong></span>
                    <span>PHQ-9: <strong>{p.phq9_score ?? '--'}</strong></span>
                    <span>GAD-7: <strong>{p.gad7_score ?? '--'}</strong></span>
                    <span>QOLIE: <strong>{p.qolie_score ?? '--'}</strong></span>
                  </div>

                  <div style={{ fontSize: 11, color: '#475569' }}>
                    <strong>Risk factors:</strong>
                    {(p.risk_factors || []).map((rf, j) => (
                      <span key={j} style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 8, margin: '1px 2px',
                        background: '#fecaca', color: '#dc2626', fontSize: 10, fontWeight: 500,
                      }}>{rf}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )
      })()}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Patient & Caregiver Concepts">
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ fontWeight: 600, color: '#0f766e', fontSize: 14 }}>{c.term}</div>
                <div style={{ color: '#475569', fontSize: 13, marginTop: 2 }}>{c.definition}</div>
              </div>
            ))}
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <Card title="Quality Metrics">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6 }}>Metric</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Target</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.quality_metrics || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6 }}>{m.metric}</td>
                      <td style={{ padding: 6, color: '#16a34a', fontWeight: 500 }}>{m.target}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card title="Compliance Standards">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6 }}>Standard</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Scope</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.compliance || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6, fontWeight: 600 }}>{c.standard}</td>
                      <td style={{ padding: 6, color: '#475569' }}>{c.scope}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>

          <div style={{ marginTop: 16 }}>
            <Card title="Remediation Strategies">
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
                {(defs.remediation || []).map((r, i) => (
                  <li key={i} style={{ marginBottom: 6, color: '#475569' }}>{r}</li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
