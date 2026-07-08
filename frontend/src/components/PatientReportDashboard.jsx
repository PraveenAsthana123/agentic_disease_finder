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

const LEVEL_COLORS = {
  normal: '#10b981', minimal: '#10b981',
  mild: '#f59e0b', moderate: '#f97316',
  severe: '#ef4444', critical: '#dc2626',
  low: '#3b82f6', high: '#ef4444'
}

const SECTION_ICONS = {
  brain: '\u{1F9E0}',
  clipboard: '\u{1F4CB}',
  activity: '\u{26A1}',
  image: '\u{1F9E0}',
  pill: '\u{1F48A}'
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'generate', label: 'Generate Report' },
  { id: 'preview', label: 'Report Preview' },
  { id: 'coverage', label: 'Data Coverage' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PatientReportDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [patientReport, setPatientReport] = useState(null)
  const [reportLoading, setReportLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/patient-report/overview`),
      axios.get(`${API_URL}/api/patient-report/breakdown`),
      axios.get(`${API_URL}/api/patient-report/definitions`),
    ])
      .then(([o, b, d]) => {
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const generateReport = async (patientId) => {
    setSelectedPatient(patientId)
    setReportLoading(true)
    try {
      const res = await axios.get(`${API_URL}/api/patient-report/breakdown?patient_id=${patientId}`)
      setPatientReport(res.data.patient_report)
      setTab('preview')
    } catch (e) {
      setError(e.message)
    } finally {
      setReportLoading(false)
    }
  }

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading patient report data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Patient-Facing Report Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Plain-language health reports for patients — simplified summaries from real clinical data
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 24, borderBottom: '2px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 700 : 500, fontSize: 13, cursor: 'pointer',
            marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(kpis, overview)}
      {tab === 'generate' && renderGenerate(breakdown, generateReport, reportLoading)}
      {tab === 'preview' && renderPreview(patientReport, selectedPatient)}
      {tab === 'coverage' && renderCoverage(overview)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(kpis, overview) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Patients" value={kpis.total_patients} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Reportable Patients" value={kpis.reportable_patients}
          sub={`${kpis.report_coverage_pct}% coverage`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Total Assessments" value={kpis.total_assessments}
          sub={`${kpis.instruments_used} instruments`} color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="Avg Medication Adherence" value={`${kpis.avg_medication_adherence_pct}%`}
          sub={`${kpis.seizure_events_logged} seizure events`} color="#f59e0b" />
      </Card>

      <Card title="Assessment Level Distribution" span={2}>
        {overview.level_distribution && overview.level_distribution.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={overview.level_distribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="level" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" name="Count">
                {overview.level_distribution.map((entry, i) => (
                  <Cell key={i} fill={LEVEL_COLORS[entry.level] || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      <Card title="Instrument Usage" span={2}>
        {overview.instrument_distribution && overview.instrument_distribution.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={overview.instrument_distribution.slice(0, 8)} dataKey="value"
                nameKey="code" cx="50%" cy="50%" outerRadius={80} label={({ code, value }) => `${code}: ${value}`}>
                {overview.instrument_distribution.slice(0, 8).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v, n, p) => [v, p.payload.name]} />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      <Card title="Monthly Assessment Trend" span={2}>
        {overview.monthly_trend && overview.monthly_trend.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={overview.monthly_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>

      <Card title="Disease Distribution" span={2}>
        {overview.disease_distribution && overview.disease_distribution.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={overview.disease_distribution} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
              <Tooltip />
              <Bar dataKey="value" fill="#8b5cf6" name="Patients" />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
      </Card>
    </div>
  )
}

function renderGenerate(breakdown, generateReport, reportLoading) {
  if (!breakdown) return <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>No data</div>
  const patients = breakdown.patient_list || []
  return (
    <div>
      <Card title={`Patient List (${breakdown.total_reportable} with reportable data)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={thStyle}>Patient ID</th>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Age</th>
                <th style={thStyle}>Condition</th>
                <th style={thStyle}>Available Sections</th>
                <th style={thStyle}>Action</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={tdStyle}><strong>{p.patient_id}</strong></td>
                  <td style={tdStyle}>{p.name || 'N/A'}</td>
                  <td style={tdStyle}>{p.age || 'N/A'}</td>
                  <td style={tdStyle}>{p.disease ? p.disease.charAt(0).toUpperCase() + p.disease.slice(1) : 'N/A'}</td>
                  <td style={tdStyle}>
                    {p.available_sections && p.available_sections.length > 0
                      ? p.available_sections.map((s, j) => (
                        <Badge key={j} text={s} color="#3b82f6" />
                      ))
                      : <span style={{ color: '#94a3b8' }}>No data</span>
                    }
                  </td>
                  <td style={tdStyle}>
                    {p.has_data ? (
                      <button
                        onClick={() => generateReport(p.patient_id)}
                        disabled={reportLoading}
                        style={{
                          padding: '4px 12px', borderRadius: 6, border: 'none',
                          background: '#3b82f6', color: '#fff', fontSize: 12,
                          cursor: reportLoading ? 'not-allowed' : 'pointer',
                          opacity: reportLoading ? 0.6 : 1
                        }}
                      >
                        {reportLoading ? 'Generating...' : 'Generate Report'}
                      </button>
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: 12 }}>Insufficient data</span>
                    )}
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

function renderPreview(report, patientId) {
  if (!report) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: 40, color: '#64748b' }}>
          <p style={{ fontSize: 15 }}>No report generated yet</p>
          <p style={{ fontSize: 13 }}>Go to the "Generate Report" tab and select a patient</p>
        </div>
      </Card>
    )
  }

  if (report.error) {
    return <Card><div style={{ color: '#ef4444', padding: 20 }}>{report.error}</div></Card>
  }

  return (
    <div>
      {/* Report Header */}
      <Card>
        <div style={{ textAlign: 'center', borderBottom: '2px solid #e2e8f0', paddingBottom: 16, marginBottom: 16 }}>
          <h3 style={{ fontSize: 20, color: '#1e293b', margin: 0 }}>Your Health Summary</h3>
          <p style={{ color: '#64748b', fontSize: 13, margin: '4px 0' }}>
            Patient: <strong>{report.patient_name}</strong> ({report.patient_id})
          </p>
          <p style={{ color: '#94a3b8', fontSize: 11 }}>
            Generated: {report.generated_at ? new Date(report.generated_at).toLocaleDateString() : 'N/A'}
          </p>
        </div>
        <div style={{
          background: '#fef3c7', border: '1px solid #fbbf24', borderRadius: 8,
          padding: 12, fontSize: 12, color: '#92400e', lineHeight: 1.5
        }}>
          {report.disclaimer}
        </div>
      </Card>

      {/* Report Sections */}
      {(report.sections || []).map((section, i) => (
        <Card key={i} title={`${SECTION_ICONS[section.icon] || ''} ${section.title}`}>
          <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: '0 0 12px' }}>
            {section.summary}
          </p>

          {/* Details table */}
          {section.details && (
            <div style={{ marginBottom: 12 }}>
              {section.details.map((d, j) => (
                <div key={j} style={{
                  display: 'flex', justifyContent: 'space-between', padding: '6px 0',
                  borderBottom: '1px solid #f1f5f9', fontSize: 13
                }}>
                  <span style={{ color: '#64748b' }}>{d.label}</span>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{d.value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Assessment results */}
          {section.assessments && (
            <div style={{ marginBottom: 12 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={thStyle}>Assessment</th>
                    <th style={thStyle}>Result</th>
                    <th style={thStyle}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {section.assessments.map((a, j) => (
                    <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={tdStyle}>{a.test}</td>
                      <td style={tdStyle}>
                        <Badge text={a.result} color={LEVEL_COLORS[a.level] || '#64748b'} />
                      </td>
                      <td style={tdStyle}>{a.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Seizure events */}
          {section.recent_events && (
            <div style={{ marginBottom: 12 }}>
              <h4 style={{ fontSize: 13, color: '#475569', margin: '8px 0' }}>Recent Events</h4>
              {section.recent_events.map((e, j) => (
                <div key={j} style={{
                  display: 'flex', gap: 16, padding: '6px 0',
                  borderBottom: '1px solid #f1f5f9', fontSize: 12
                }}>
                  <span style={{ color: '#64748b' }}>Date: {e.date}</span>
                  <span style={{ color: '#64748b' }}>Duration: {e.duration_sec}s</span>
                  <span style={{ color: '#64748b' }}>Severity: {e.severity}</span>
                </div>
              ))}
            </div>
          )}

          {/* Plain note */}
          {section.plain_note && (
            <div style={{
              background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: 6,
              padding: 10, fontSize: 12, color: '#0369a1', lineHeight: 1.5, marginTop: 8
            }}>
              {section.plain_note}
            </div>
          )}
        </Card>
      ))}

      {/* Closing / Next Steps */}
      {report.closing && (
        <Card title="Next Steps">
          <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: '0 0 12px' }}>
            {report.closing.message}
          </p>
          {report.closing.next_steps && (
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {report.closing.next_steps.map((step, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 4, lineHeight: 1.5 }}>
                  {step}
                </li>
              ))}
            </ul>
          )}
        </Card>
      )}
    </div>
  )
}

function renderCoverage(overview) {
  if (!overview) return null
  const cov = overview.data_coverage || {}
  const items = [
    { label: 'EEG Analyses', value: cov.eeg_analyses, icon: '\u{1F9E0}', color: '#3b82f6' },
    { label: 'Assessments', value: cov.assessments, icon: '\u{1F4CB}', color: '#8b5cf6' },
    { label: 'Seizure Diary', value: cov.seizure_diary, icon: '\u{26A1}', color: '#f59e0b' },
    { label: 'MRI Findings', value: cov.mri_findings, icon: '\u{1F50D}', color: '#10b981' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {items.map((item, i) => (
        <Card key={i}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ fontSize: 32 }}>{item.icon}</div>
            <div>
              <div style={{ fontSize: 24, fontWeight: 700, color: item.color }}>{item.value}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{item.label} (patients with data)</div>
            </div>
          </div>
        </Card>
      ))}
      <Card title="Report Completeness by Data Source" span={2}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {items.map((item, i) => {
            const total = overview.kpis?.total_patients || 1
            const pct = Math.round((item.value / total) * 100)
            return (
              <div key={i} style={{ flex: 1, minWidth: 120 }}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{item.label}</div>
                <div style={{ background: '#e2e8f0', borderRadius: 4, height: 20, overflow: 'hidden' }}>
                  <div style={{
                    width: `${pct}%`, height: '100%', background: item.color,
                    borderRadius: 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 10, color: '#fff', fontWeight: 600
                  }}>{pct}%</div>
                </div>
              </div>
            )
          })}
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(definitions) {
  if (!definitions) return null
  return (
    <div>
      {(definitions.sections || []).map((section, i) => (
        <Card key={i} title={section.heading}>
          <div>
            {(section.items || []).map((item, j) => (
              <div key={j} style={{
                padding: '8px 0', borderBottom: '1px solid #f1f5f9'
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{item.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2, lineHeight: 1.5 }}>
                  {item.definition}
                </div>
              </div>
            ))}
          </div>
        </Card>
      ))}
      {definitions.health_literacy_note && (
        <Card>
          <div style={{
            background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 6,
            padding: 12, fontSize: 12, color: '#166534', lineHeight: 1.5
          }}>
            {definitions.health_literacy_note}
          </div>
        </Card>
      )}
    </div>
  )
}

const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', fontWeight: 600 }
const tdStyle = { padding: '8px 10px', fontSize: 12 }
