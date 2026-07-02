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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const SEVERITY_COLORS = {
  normal: '#10b981',
  mild: '#f59e0b',
  moderate: '#f97316',
  severe: '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Reports' },
  { id: 'appointments', label: 'Appointments' },
  { id: 'seizures', label: 'Seizure Log' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PatientReportingDashboard() {
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
      axios.get(`${API_URL}/api/patient-reporting/overview`),
      axios.get(`${API_URL}/api/patient-reporting/breakdown`),
      axios.get(`${API_URL}/api/patient-reporting/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Patient Reporting data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No patient reporting data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Patient Reporting Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Scheduled patient monitoring summaries, report generation metrics, assessment coverage, appointment tracking
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'appointments' && <AppointmentsTab breakdown={breakdown} />}
      {tab === 'seizures' && <SeizureLogTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const instDist = overview.instrument_distribution || []
  const apptDist = overview.appt_status_distribution || []
  const sevDist = overview.severity_distribution || []
  const dailyActivity = overview.daily_activity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Instrument Distribution Bar */}
      <Card title="Assessment Instrument Distribution" span={2}>
        {instDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={instDist} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="instrument" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                {instDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument data</div>}
      </Card>

      {/* Appointment Status Pie */}
      <Card title="Appointment Status Distribution" span={2}>
        {apptDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={apptDist} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={100}
                label={({ status, count }) => `${status} (${count})`}>
                {apptDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No appointment data</div>}
      </Card>

      {/* Severity Distribution Bar */}
      <Card title="Assessment Severity Distribution" span={2}>
        {sevDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={sevDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {sevDist.map((entry, i) => (
                  <Cell key={i} fill={entry.color || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No severity data</div>}
      </Card>

      {/* Daily Activity Line Chart */}
      <Card title="Daily Report Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily activity data</div>}
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_reports || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient Report Inventory (${patients.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Name', 'Disease', 'Assessments', 'Appts', 'Seizures', 'Meds', 'MRI', 'Completeness', 'Latest Assessment', 'Instruments'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: ['Assessments', 'Appts', 'Seizures', 'Meds', 'MRI', 'Completeness'].includes(h) ? 'right' : 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}><Badge text={p.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.name || ''}</td>
                  <td style={{ padding: '6px 10px' }}>{p.disease ? <Badge text={p.disease} color="#8b5cf6" /> : ''}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.assessments}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.appointments}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.seizures}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.medications}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.mri_findings}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <Badge text={`${p.completeness}%`}
                      color={p.completeness >= 80 ? '#10b981' : p.completeness >= 50 ? '#f59e0b' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{p.latest_assessment || ''}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {(p.instruments || []).map((inst, j) => (
                        <Badge key={j} text={inst} color={COLORS[j % COLORS.length]} />
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function AppointmentsTab({ breakdown }) {
  const appointments = breakdown?.appointment_schedule || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Appointment Schedule (${appointments.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['ID', 'Patient', 'Provider', 'Department', 'Type', 'Status', 'Scheduled For', 'Duration'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Duration' ? 'right' : 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {appointments.map((a, i) => {
                const statusColor = a.status === 'completed' ? '#10b981' : a.status === 'scheduled' ? '#3b82f6' : a.status === 'cancelled' ? '#ef4444' : '#f59e0b'
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={a.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px', fontWeight: 500 }}>{a.provider || ''}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{a.department || ''}</td>
                    <td style={{ padding: '6px 10px' }}>{a.appt_type ? <Badge text={a.appt_type} color="#8b5cf6" /> : ''}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={a.status || 'unknown'} color={statusColor} /></td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{a.scheduled_for || ''}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{a.duration_min != null ? `${a.duration_min} min` : ''}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function SeizureLogTab({ breakdown }) {
  const seizures = breakdown?.seizure_log || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Seizure Diary Events (${seizures.length})`}>
        {seizures.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#fef2f2' }}>
                  {['ID', 'Patient', 'Date', 'Time', 'Duration (sec)', 'Location', 'Witnessed', 'Aura'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Duration (sec)' ? 'right' : 'left', borderBottom: '1px solid #fecaca', color: '#991b1b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {seizures.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fef2f2' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.id}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={s.patient_id} color="#3b82f6" /></td>
                    <td style={{ padding: '6px 10px' }}>{s.event_date || ''}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{s.event_time || ''}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{s.duration_sec != null ? s.duration_sec : ''}</td>
                    <td style={{ padding: '6px 10px' }}>{s.location ? <Badge text={s.location} color="#f97316" /> : ''}</td>
                    <td style={{ padding: '6px 10px' }}>{s.witnessed != null ? (s.witnessed ? 'Yes' : 'No') : ''}</td>
                    <td style={{ padding: '6px 10px' }}>{s.aura != null ? (s.aura ? 'Yes' : 'No') : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#10b981', fontSize: 13, padding: 20, textAlign: 'center' }}>No seizure events recorded</div>}
      </Card>
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Pipeline Events (${events.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['ID', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{e.id}</td>
                  <td style={{ padding: '6px 10px' }}>{e.patient_id ? <Badge text={e.patient_id} color="#3b82f6" /> : ''}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={e.component || 'system'} color="#8b5cf6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.action}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{e.actor || 'system'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || ''}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts_utc || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const concepts = definitions.concepts || []
  const qualityMetrics = definitions.quality_metrics || []
  const reportTypes = definitions.report_types || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Report Concepts */}
      <Card title={`Report Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Report Types */}
      <Card title={`Report Types (${reportTypes.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Type</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {reportTypes.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}><Badge text={r.type} color={COLORS[i % COLORS.length]} /></td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${qualityMetrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {qualityMetrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Standard</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Notes</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.ref}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Strategy</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.strategy}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
