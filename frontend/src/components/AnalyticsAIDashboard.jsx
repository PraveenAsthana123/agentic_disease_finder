import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']

export default function AnalyticsAIDashboard() {
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
      axios.get(`${API}/analytics-ai/overview`),
      axios.get(`${API}/analytics-ai/breakdown`),
      axios.get(`${API}/analytics-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'demographics', label: 'Demographics' },
    { id: 'clinical-activity', label: 'Clinical Activity' },
    { id: 'patient-details', label: 'Patient Details' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Analytics AI dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No analytics data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Analytics AI</h2>
        <Badge
          text={`${overview.total_patients || 0} Patients`}
          color="#3b82f6"
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'demographics' && renderDemographics()}
      {tab === 'clinical-activity' && renderClinicalActivity()}
      {tab === 'patient-details' && renderPatientDetails()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const kpis = [
      { label: 'Total Patients', value: overview.total_patients || 0, color: '#3b82f6' },
      { label: 'Total Analyses', value: overview.total_analyses || 0, color: '#10b981' },
      { label: 'Total Assessments', value: overview.total_assessments || 0, color: '#8b5cf6' },
      { label: 'Seizure Events', value: overview.seizure_events || 0, color: '#ef4444' },
      { label: 'Avg Age', value: overview.avg_age != null ? overview.avg_age : 'N/A', color: '#f59e0b' },
      { label: 'Gender Ratio', value: overview.gender_ratio || 'N/A', color: '#ec4899' },
      { label: 'Diseases Tracked', value: overview.diseases_tracked || 0, color: '#06b6d4' },
      { label: 'Active Departments', value: overview.active_departments || 0, color: '#334155' },
    ]
    const diseaseDist = overview.disease_distribution || []
    const deptWorkload = overview.department_workload || []
    const signalQuality = overview.signal_quality_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Disease Distribution" span={2}>
          {diseaseDist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={diseaseDist} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={100} label={({ disease, count }) => `${disease}: ${count}`}>
                  {diseaseDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Department Workload" span={2}>
          {deptWorkload.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={deptWorkload} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="department" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Workload" radius={[4, 4, 0, 0]}>
                  {deptWorkload.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Signal Quality Distribution" span={4}>
          {signalQuality.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={signalQuality} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Recordings" radius={[4, 4, 0, 0]}>
                  {signalQuality.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderDemographics() {
    const ageHist = overview.age_histogram || []
    const genderBreakdown = overview.gender_breakdown || []
    const diseaseByGender = overview.disease_by_gender || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Age Distribution" span={2}>
          {ageHist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={ageHist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Gender Breakdown">
          {genderBreakdown.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={genderBreakdown} dataKey="count" nameKey="gender" cx="50%" cy="50%" outerRadius={100} label={({ gender, count }) => `${gender}: ${count}`}>
                  {genderBreakdown.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Disease by Gender">
          {diseaseByGender.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={diseaseByGender} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="male" name="Male" fill="#3b82f6" stackId="gender" radius={[0, 0, 0, 0]} />
                <Bar dataKey="female" name="Female" fill="#ec4899" stackId="gender" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderClinicalActivity() {
    const monthlyTrend = breakdown?.monthly_trend || []
    const instrumentDist = breakdown?.instrument_distribution || []
    const seizureSeverity = breakdown?.seizure_severity || []
    const appointmentStatus = breakdown?.appointment_status || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Monthly Activity Trend" span={2}>
          {monthlyTrend.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={monthlyTrend} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="analyses" name="Analyses" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="assessments" name="Assessments" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="seizures" name="Seizures" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Assessment Instrument Distribution" span={2}>
          {instrumentDist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={instrumentDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="instrument" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Assessments" radius={[4, 4, 0, 0]}>
                  {instrumentDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Seizure Severity Distribution">
          {seizureSeverity.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={seizureSeverity} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={100} label={({ severity, count }) => `${severity}: ${count}`}>
                  {seizureSeverity.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Appointment Status Breakdown">
          {appointmentStatus.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={appointmentStatus} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="status" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Appointments" radius={[4, 4, 0, 0]}>
                  {appointmentStatus.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderPatientDetails() {
    const patients = breakdown?.patient_summaries || []
    const medCoverage = breakdown?.medication_coverage || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Summaries (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Age</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Gender</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Disease</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Analyses</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Assessments</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Seizures</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Medications</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Appointments</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id || p.name || `Patient ${i + 1}`}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.age != null ? p.age : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={p.gender || 'N/A'} color={p.gender === 'Male' ? '#3b82f6' : p.gender === 'Female' ? '#ec4899' : '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#475569' }}>{p.disease || 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600 }}>{p.analyses != null ? p.analyses : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600 }}>{p.assessments != null ? p.assessments : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: p.seizures > 0 ? '#ef4444' : '#64748b' }}>{p.seizures != null ? p.seizures : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600 }}>{p.medications != null ? p.medications : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600 }}>{p.appointments != null ? p.appointments : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Medication Coverage">
          {medCoverage.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={medCoverage} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={100} label={({ status, count }) => `${status}: ${count}`}>
                  {medCoverage.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    const sections = definitions?.sections || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {sections.map((sec, si) => {
          const items = sec.items || []
          if (items.length === 0) return null
          return (
            <Card key={si} title={sec.title}>
              {items.map((item, ii) => (
                <div key={ii} style={{ marginBottom: 12, borderBottom: ii < items.length - 1 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term || item.standard || item.scenario || item.category || item.method}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition || item.description}</div>
                  {item.requirement && <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, marginTop: 4 }}><strong>Requirement:</strong> {item.requirement}</div>}
                  {item.how_met && <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>How Met:</strong> {item.how_met}</div>}
                  {item.strategy && <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, marginTop: 4 }}><strong>Strategy:</strong> {item.strategy}</div>}
                  {item.action && <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}><strong>Action:</strong> {item.action}</div>}
                </div>
              ))}
            </Card>
          )
        })}
      </div>
    )
  }
}
