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

const STATUS_COLORS = {
  completed: '#10b981',
  scheduled: '#3b82f6',
  cancelled: '#ef4444',
  'no-show': '#f97316',
  confirmed: '#8b5cf6',
  pending: '#f59e0b',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'subjects', label: 'Subject Inventory' },
  { id: 'protocol', label: 'Protocol Matrix' },
  { id: 'visits', label: 'Visit Log' },
  { id: 'outcomes', label: 'Outcomes Data' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ResearchCoordinatorDashboard() {
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
      axios.get(`${API_URL}/api/research-coordinator/overview`),
      axios.get(`${API_URL}/api/research-coordinator/breakdown`),
      axios.get(`${API_URL}/api/research-coordinator/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Research Coordinator data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No research coordinator data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Research Coordinator Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Study enrollment, protocol compliance, cohort management, visit tracking, outcomes collection, data completeness
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
      {tab === 'subjects' && <SubjectsTab breakdown={breakdown} />}
      {tab === 'protocol' && <ProtocolTab breakdown={breakdown} />}
      {tab === 'visits' && <VisitsTab breakdown={breakdown} />}
      {tab === 'outcomes' && <OutcomesTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = [
    { label: 'Enrolled Subjects', value: overview.total_subjects || 0, color: '#3b82f6' },
    { label: 'Total Assessments', value: overview.total_assessments || 0, color: '#8b5cf6' },
    { label: 'Total Visits', value: overview.total_visits || 0, color: '#06b6d4' },
    { label: 'Completed Visits', value: overview.completed_visits || 0, color: '#10b981' },
    { label: 'Visit Compliance', value: `${overview.visit_compliance_pct || 0}%`, color: overview.visit_compliance_pct >= 80 ? '#10b981' : '#f59e0b' },
    { label: 'Seizure Events', value: overview.total_seizure_events || 0, color: '#ef4444' },
    { label: 'EEG Uploads', value: overview.total_eeg_uploads || 0, color: '#f97316' },
    { label: 'Instruments Used', value: overview.instruments_used || 0, color: '#ec4899' },
  ]

  const diseaseDist = overview.disease_distribution || []
  const enrollByMonth = overview.enrollment_by_month || []
  const instCoverage = overview.instrument_coverage || []
  const visitStatusDist = overview.visit_status_distribution || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
      ))}

      {/* Disease Distribution Pie */}
      <Card title="Disease Distribution (Cohort)" span={2}>
        {diseaseDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={diseaseDist} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={100}
                label={({ disease, count }) => `${disease} (${count})`}>
                {diseaseDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No disease data</div>}
      </Card>

      {/* Visit Status Distribution */}
      <Card title="Visit Status Distribution" span={2}>
        {visitStatusDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={visitStatusDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="status" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {visitStatusDist.map((entry, i) => (
                  <Cell key={i} fill={STATUS_COLORS[entry.status] || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No visit data</div>}
      </Card>

      {/* Instrument Coverage Bar */}
      <Card title="Instrument Coverage (assessments per instrument)" span={2}>
        {instCoverage.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={instCoverage} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="instrument" width={120} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                {instCoverage.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No instrument data</div>}
      </Card>

      {/* Enrollment by Month */}
      <Card title="Enrollment Timeline" span={2}>
        {enrollByMonth.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={enrollByMonth}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No enrollment data</div>}
      </Card>
    </div>
  )
}

function SubjectsTab({ breakdown }) {
  if (!breakdown) return <div style={{ color: '#94a3b8' }}>No data</div>
  const subjects = breakdown.subject_inventory || []

  return (
    <Card title={`Subject Inventory (${subjects.length} enrolled)`}>
      <div style={{ maxHeight: 600, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8 }}>Patient ID</th>
              <th style={{ padding: 8 }}>Name</th>
              <th style={{ padding: 8 }}>Age</th>
              <th style={{ padding: 8 }}>Gender</th>
              <th style={{ padding: 8 }}>Disease</th>
              <th style={{ padding: 8 }}>Assessments</th>
              <th style={{ padding: 8 }}>Visits</th>
              <th style={{ padding: 8 }}>Seizure Events</th>
              <th style={{ padding: 8 }}>Uploads</th>
              <th style={{ padding: 8 }}>Enrolled</th>
            </tr>
          </thead>
          <tbody>
            {subjects.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{s.patient_id}</td>
                <td style={{ padding: 8 }}>{s.name || '-'}</td>
                <td style={{ padding: 8 }}>{s.age || '-'}</td>
                <td style={{ padding: 8 }}>{s.gender || '-'}</td>
                <td style={{ padding: 8 }}><Badge text={s.disease || 'Unknown'} color="#3b82f6" /></td>
                <td style={{ padding: 8 }}>{s.assessments_count}</td>
                <td style={{ padding: 8 }}>{s.visits_count}</td>
                <td style={{ padding: 8 }}>{s.seizure_events > 0 ? <Badge text={String(s.seizure_events)} color="#ef4444" /> : '0'}</td>
                <td style={{ padding: 8 }}>{s.uploads}</td>
                <td style={{ padding: 8, whiteSpace: 'nowrap' }}>{(s.enrollment_date || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function ProtocolTab({ breakdown }) {
  if (!breakdown) return <div style={{ color: '#94a3b8' }}>No data</div>
  const matrix = breakdown.protocol_matrix || []
  if (matrix.length === 0) return <Card title="Protocol Compliance Matrix"><div style={{ color: '#94a3b8' }}>No protocol data</div></Card>

  const instruments = Object.keys(matrix[0] || {}).filter(k => !['patient_id', 'name'].includes(k))

  return (
    <Card title={`Protocol Compliance Matrix (${matrix.length} subjects x ${instruments.length} instruments)`}>
      <div style={{ maxHeight: 600, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 6 }}>Patient</th>
              <th style={{ padding: 6 }}>Name</th>
              {instruments.map(inst => (
                <th key={inst} style={{ padding: 6, textAlign: 'center' }}>{inst}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6, fontWeight: 600 }}>{row.patient_id}</td>
                <td style={{ padding: 6 }}>{row.name || '-'}</td>
                {instruments.map(inst => (
                  <td key={inst} style={{ padding: 6, textAlign: 'center' }}>
                    {row[inst] > 0 ? (
                      <Badge text={String(row[inst])} color="#10b981" />
                    ) : (
                      <span style={{ color: '#cbd5e1' }}>-</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function VisitsTab({ breakdown }) {
  if (!breakdown) return <div style={{ color: '#94a3b8' }}>No data</div>
  const visits = breakdown.visit_log || []

  return (
    <Card title={`Visit Log (${visits.length} appointments)`}>
      <div style={{ maxHeight: 600, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8 }}>Patient</th>
              <th style={{ padding: 8 }}>Provider</th>
              <th style={{ padding: 8 }}>Type</th>
              <th style={{ padding: 8 }}>Status</th>
              <th style={{ padding: 8 }}>Scheduled</th>
              <th style={{ padding: 8 }}>Completed</th>
              <th style={{ padding: 8 }}>Duration</th>
            </tr>
          </thead>
          <tbody>
            {visits.map((v, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{v.patient_id}</td>
                <td style={{ padding: 8 }}>{v.provider || '-'}</td>
                <td style={{ padding: 8 }}><Badge text={v.appt_type || '-'} color="#06b6d4" /></td>
                <td style={{ padding: 8 }}>
                  <Badge text={v.status || 'unknown'} color={STATUS_COLORS[v.status] || '#6b7280'} />
                </td>
                <td style={{ padding: 8, whiteSpace: 'nowrap' }}>{(v.scheduled_for || '').slice(0, 16)}</td>
                <td style={{ padding: 8, whiteSpace: 'nowrap' }}>{v.completed_at ? v.completed_at.slice(0, 16) : '-'}</td>
                <td style={{ padding: 8 }}>{v.duration_min ? `${v.duration_min} min` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function OutcomesTab({ breakdown }) {
  if (!breakdown) return <div style={{ color: '#94a3b8' }}>No data</div>
  const seizures = breakdown.seizure_log || []
  const submissions = breakdown.data_submissions || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Seizure Events */}
      <Card title={`Seizure Events (${seizures.length} recorded)`}>
        {seizures.length > 0 ? (
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: 6 }}>Patient</th>
                  <th style={{ padding: 6 }}>Date</th>
                  <th style={{ padding: 6 }}>Duration (s)</th>
                  <th style={{ padding: 6 }}>Severity</th>
                  <th style={{ padding: 6 }}>Location</th>
                  <th style={{ padding: 6 }}>Trigger</th>
                </tr>
              </thead>
              <tbody>
                {seizures.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6, fontWeight: 600 }}>{s.patient_id}</td>
                    <td style={{ padding: 6, whiteSpace: 'nowrap' }}>{s.event_date || '-'}</td>
                    <td style={{ padding: 6 }}>{s.duration_sec || '-'}</td>
                    <td style={{ padding: 6 }}>
                      <Badge text={s.severity || 'unknown'} color={
                        s.severity === 'severe' ? '#ef4444' : s.severity === 'moderate' ? '#f59e0b' : '#10b981'
                      } />
                    </td>
                    <td style={{ padding: 6 }}>{s.location || '-'}</td>
                    <td style={{ padding: 6 }}>{s.trigger || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No seizure events recorded</div>}
      </Card>

      {/* Data Submissions */}
      <Card title={`Data Submissions (${submissions.length} EEG uploads + analyses)`}>
        {submissions.length > 0 ? (
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: 6 }}>Upload ID</th>
                  <th style={{ padding: 6 }}>Patient</th>
                  <th style={{ padding: 6 }}>File</th>
                  <th style={{ padding: 6 }}>Disease</th>
                  <th style={{ padding: 6 }}>Prediction</th>
                  <th style={{ padding: 6 }}>Confidence</th>
                  <th style={{ padding: 6 }}>Signal Quality</th>
                </tr>
              </thead>
              <tbody>
                {submissions.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6 }}>{d.upload_id || d.id}</td>
                    <td style={{ padding: 6, fontWeight: 600 }}>{d.patient_id}</td>
                    <td style={{ padding: 6, maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {d.file_name || '-'}
                    </td>
                    <td style={{ padding: 6 }}><Badge text={d.disease || '-'} color="#3b82f6" /></td>
                    <td style={{ padding: 6 }}>{d.predicted_label || '-'}</td>
                    <td style={{ padding: 6 }}>
                      {d.confidence != null ? (
                        <span style={{ color: d.confidence >= 0.8 ? '#10b981' : d.confidence >= 0.6 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                          {(d.confidence * 100).toFixed(1)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td style={{ padding: 6 }}>{d.signal_quality || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data submissions</div>}
      </Card>
    </div>
  )
}

function PipelineTab({ breakdown }) {
  if (!breakdown) return <div style={{ color: '#94a3b8' }}>No data</div>
  const events = breakdown.pipeline_events || []
  const dailyActivity = breakdown.daily_activity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Daily Activity */}
      <Card title="Daily Pipeline Activity" span={1}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No activity data</div>}
      </Card>

      {/* Event Log */}
      <Card title={`Pipeline Event Log (${events.length} events)`}>
        <div style={{ maxHeight: 500, overflow: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                <th style={{ padding: 6 }}>ID</th>
                <th style={{ padding: 6 }}>Component</th>
                <th style={{ padding: 6 }}>Action</th>
                <th style={{ padding: 6 }}>Actor</th>
                <th style={{ padding: 6 }}>Detail</th>
                <th style={{ padding: 6 }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{ev.id}</td>
                  <td style={{ padding: 6 }}><Badge text={ev.component || '-'} color="#3b82f6" /></td>
                  <td style={{ padding: 6 }}>{ev.action}</td>
                  <td style={{ padding: 6 }}>{ev.actor || '-'}</td>
                  <td style={{ padding: 6, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {ev.detail || '-'}
                  </td>
                  <td style={{ padding: 6, whiteSpace: 'nowrap' }}>{ev.ts_local || ev.ts_utc || '-'}</td>
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
  if (!definitions) return <div style={{ color: '#94a3b8' }}>No definitions</div>
  const concepts = definitions.concepts || []
  const metrics = definitions.quality_metrics || []
  const phases = definitions.study_phases || []
  const compliance = definitions.compliance_refs || []
  const remediation = definitions.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Concepts */}
      <Card title={`Research Coordination Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 220 }}>Concept</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{c.name}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Study Phases */}
      <Card title={`Study Phases (${phases.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 150 }}>Phase</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {phases.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{p.name}</td>
                <td style={{ padding: 8, color: '#475569' }}>{p.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${metrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 220 }}>Metric</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{m.name}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Compliance References */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8 }}>Reference</th>
              <th style={{ padding: 8 }}>Scope</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600 }}>{c.name}</td>
                <td style={{ padding: 8, color: '#475569' }}>{c.scope}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: 8, width: 200 }}>Strategy</th>
              <th style={{ padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 600, verticalAlign: 'top' }}>{r.name}</td>
                <td style={{ padding: 8, color: '#475569', lineHeight: 1.5 }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
