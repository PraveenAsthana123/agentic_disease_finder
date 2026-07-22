import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#ef4444', '#22c55e', '#f59e0b', '#7c4dff', '#ec4899', '#6366f1', '#14b8a6']
const fmt = v => (typeof v === 'number' ? v.toLocaleString() : v ?? '--')

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

const badgeStyle = (color) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: color,
})

const qualityColor = (score) => {
  if (score >= 90) return '#22c55e'
  if (score >= 75) return '#1e88e5'
  if (score >= 50) return '#f59e0b'
  return '#ef4444'
}

const coverageColor = (pct) => {
  if (pct >= 80) return '#22c55e'
  if (pct >= 50) return '#f59e0b'
  return '#ef4444'
}

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }
const tdStyle = (i) => ({ padding: '8px 12px', borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' })

export default function ClinicalDataManagerDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/clinical-data-manager/overview`),
      axios.get(`${API_URL}/api/clinical-data-manager/breakdown`),
      axios.get(`${API_URL}/api/clinical-data-manager/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Clinical Data Manager Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No data available</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'quality', label: 'Data Quality' },
    { id: 'coverage', label: 'Modality Coverage' },
    { id: 'tasks', label: 'Task Catalog' },
    { id: 'inventory', label: 'Dataset Inventory' },
    { id: 'patients', label: 'Patient Coverage' },
    { id: 'lineage', label: 'Data Lineage' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}
  const qd = overview?.quality_dimensions || []
  const mc = overview?.modality_coverage || []
  const mm = overview?.missing_matrix || []
  const arc = overview?.ai_readiness_components || {}
  const lineage = overview?.lineage || []
  const tasks = breakdown?.tasks || []
  const tables = breakdown?.dataset_inventory || []
  const patientCov = breakdown?.patient_coverage || []
  const instruments = breakdown?.instrument_distribution || []

  // Radar data for AI readiness
  const radarData = Object.entries(arc).map(([key, val]) => ({
    dimension: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    score: val ?? 0,
    fullMark: 100,
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        Clinical Data Manager Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Data quality dimensions, AI readiness, modality coverage, dataset inventory, lineage, and task catalog
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#1e88e5' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Tab: Overview ── */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Patients" value={k.total_patients} color="#1e88e5" /></Card>
            <Card><KPI label="Total Records" value={fmt(k.total_records)} color="#7c4dff" /></Card>
            <Card><KPI label="Tables" value={k.total_tables} color="#14b8a6" /></Card>
            <Card><KPI label="Uploads" value={k.total_uploads} color="#6366f1" /></Card>
            <Card><KPI label="EEG Analyses" value={k.total_analyses} color="#22c55e" /></Card>
            <Card><KPI label="Assessments" value={k.total_assessments} color="#f59e0b" /></Card>
            <Card><KPI label="Audit Events" value={fmt(k.audit_events)} color="#ec4899" /></Card>
            <Card><KPI label="AI Readiness" value={k.ai_readiness_score} sub={k.ai_readiness_grade} color={qualityColor(k.ai_readiness_score)} /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* Quality Dimensions Bar */}
            <Card title="Data Quality Dimensions">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={qd} layout="vertical" margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} />
                  <YAxis type="category" dataKey="dimension" width={95} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="score" radius={[0, 6, 6, 0]}>
                    {qd.map((entry, i) => (
                      <Cell key={i} fill={qualityColor(entry.score)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* AI Readiness Radar */}
            <Card title="AI Readiness Components">
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 10 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Score" dataKey="score" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
                  <Tooltip formatter={v => `${v}%`} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            {/* Modality Coverage Bar */}
            <Card title="Modality Coverage (% patients)">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={mc}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="modality" tick={{ fontSize: 10 }} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={v => `${v}%`} />
                  <Bar dataKey="pct" radius={[6, 6, 0, 0]}>
                    {mc.map((entry, i) => (
                      <Cell key={i} fill={coverageColor(entry.pct)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Missing Data Bar */}
            <Card title="Missing Data by Modality">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={mm}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="modality" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="present" stackId="a" fill="#22c55e" name="Present" />
                  <Bar dataKey="missing" stackId="a" fill="#ef4444" name="Missing" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ── Tab: Data Quality ── */}
      {tab === 'quality' && (
        <>
          <Card title="Data Quality Dimensions — Detail" span={2}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Dimension</th>
                  <th style={thStyle}>Score</th>
                  <th style={thStyle}>Grade</th>
                  <th style={thStyle}>Basis</th>
                </tr>
              </thead>
              <tbody>
                {qd.map((d, i) => (
                  <tr key={i}>
                    <td style={tdStyle(i)}><strong>{d.dimension}</strong></td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(qualityColor(d.score))}>{d.score != null ? `${d.score}%` : 'N/A'}</span>
                    </td>
                    <td style={tdStyle(i)}>
                      {d.score >= 90 ? 'Excellent' : d.score >= 75 ? 'Good' : d.score >= 50 ? 'Fair' : 'Poor'}
                    </td>
                    <td style={tdStyle(i)}>{d.basis}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }}>
            <Card title="AI Readiness Score">
              <div style={{ display: 'flex', alignItems: 'center', gap: 24, padding: 16 }}>
                <div style={{ fontSize: 48, fontWeight: 700, color: qualityColor(k.ai_readiness_score) }}>
                  {k.ai_readiness_score}
                </div>
                <div>
                  <div style={{ fontSize: 18, fontWeight: 600, color: '#334155' }}>{k.ai_readiness_grade}</div>
                  <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
                    Composite of completeness, uniqueness, validity, label coverage, and signal quality
                  </div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12, marginTop: 12 }}>
                {Object.entries(arc).map(([key, val]) => (
                  <div key={key} style={{ textAlign: 'center', padding: 8, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: qualityColor(val ?? 0) }}>{val != null ? `${val}%` : 'N/A'}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>{key.replace(/_/g, ' ')}</div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Tab: Modality Coverage ── */}
      {tab === 'coverage' && (
        <>
          <Card title="Modality Coverage (patients with data in each modality)">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Modality</th>
                  <th style={thStyle}>Patients</th>
                  <th style={thStyle}>Coverage</th>
                  <th style={thStyle}>Missing</th>
                  <th style={thStyle}>Gap</th>
                </tr>
              </thead>
              <tbody>
                {mc.map((m, i) => (
                  <tr key={i}>
                    <td style={tdStyle(i)}><strong>{m.modality}</strong></td>
                    <td style={tdStyle(i)}>{m.patients} / {k.total_patients}</td>
                    <td style={tdStyle(i)}><span style={badgeStyle(coverageColor(m.pct))}>{m.pct}%</span></td>
                    <td style={tdStyle(i)}>{k.total_patients - m.patients}</td>
                    <td style={tdStyle(i)}>
                      <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: 120 }}>
                        <div style={{ background: coverageColor(m.pct), borderRadius: 4, height: 8, width: `${m.pct}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {instruments.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <Card title="Assessment Instrument Distribution">
                <ResponsiveContainer width="100%" height={Math.max(260, instruments.length * 28)}>
                  <BarChart data={instruments.slice(0, 20)} layout="vertical" margin={{ left: 140 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="instrument" width={135} tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#7c4dff" radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>
          )}
        </>
      )}

      {/* ── Tab: Task Catalog ── */}
      {tab === 'tasks' && (
        <Card title={`CDM Task Catalog (${tasks.length} tasks)`}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>#</th>
                <th style={thStyle}>Task</th>
                <th style={thStyle}>AI Feature</th>
                <th style={thStyle}>Deliverable</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Steps</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t, i) => (
                <tr key={i}>
                  <td style={tdStyle(i)}>{i + 1}</td>
                  <td style={tdStyle(i)}><strong>{t.name}</strong></td>
                  <td style={tdStyle(i)}>{t.ai_feature || '--'}</td>
                  <td style={tdStyle(i)}>{t.deliverable || '--'}</td>
                  <td style={tdStyle(i)}>
                    <span style={badgeStyle(t.status === 'built' ? '#22c55e' : t.status === 'partial' ? '#f59e0b' : '#94a3b8')}>
                      {t.status}
                    </span>
                  </td>
                  <td style={tdStyle(i)}>
                    <ul style={{ margin: 0, paddingLeft: 16, fontSize: 11, color: '#64748b' }}>
                      {(t.steps || []).slice(0, 3).map((s, j) => <li key={j}>{s}</li>)}
                      {(t.steps || []).length > 3 && <li>+{t.steps.length - 3} more</li>}
                    </ul>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* ── Tab: Dataset Inventory ── */}
      {tab === 'inventory' && (
        <Card title={`Dataset Inventory (${tables.length} tables, ${fmt(k.total_records)} rows)`}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Table</th>
                <th style={thStyle}>Rows</th>
                <th style={thStyle}>Columns</th>
                <th style={thStyle}>Schema (first 8 cols)</th>
              </tr>
            </thead>
            <tbody>
              {tables.map((t, i) => (
                <tr key={i}>
                  <td style={tdStyle(i)}><strong>{t.table}</strong></td>
                  <td style={tdStyle(i)}>{fmt(t.rows)}</td>
                  <td style={tdStyle(i)}>{t.columns}</td>
                  <td style={tdStyle(i)}>
                    <span style={{ fontSize: 11, color: '#64748b' }}>
                      {(t.column_names || []).join(', ')}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* ── Tab: Patient Coverage ── */}
      {tab === 'patients' && (
        <Card title="Per-Patient Modality Coverage Matrix">
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Patient ID</th>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Age</th>
                <th style={thStyle}>EEG</th>
                <th style={thStyle}>Assessment</th>
                <th style={thStyle}>Seizure Diary</th>
                <th style={thStyle}>MRI</th>
                <th style={thStyle}>Medication</th>
                <th style={thStyle}>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {patientCov.map((p, i) => (
                <tr key={i}>
                  <td style={tdStyle(i)}><strong>{p.patient_id}</strong></td>
                  <td style={tdStyle(i)}>{p.name}</td>
                  <td style={tdStyle(i)}>{p.age ?? '--'}</td>
                  <td style={tdStyle(i)}>{p.eeg ? '\u2705' : '\u274c'}</td>
                  <td style={tdStyle(i)}>{p.assessment ? '\u2705' : '\u274c'}</td>
                  <td style={tdStyle(i)}>{p.seizure_diary ? '\u2705' : '\u274c'}</td>
                  <td style={tdStyle(i)}>{p.mri ? '\u2705' : '\u274c'}</td>
                  <td style={tdStyle(i)}>{p.medication ? '\u2705' : '\u274c'}</td>
                  <td style={tdStyle(i)}>
                    <span style={badgeStyle(coverageColor(p.coverage_pct))}>{p.coverage_pct}%</span>
                    <span style={{ fontSize: 11, color: '#64748b', marginLeft: 6 }}>{p.modalities}/5</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* ── Tab: Data Lineage ── */}
      {tab === 'lineage' && (
        <Card title="Data Lineage — Pipeline Stages">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
            {lineage.map((stage, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '12px 0' }}>
                <div style={{
                  width: 36, height: 36, borderRadius: '50%', background: '#1e88e5',
                  color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 700, fontSize: 14, flexShrink: 0,
                }}>
                  {i + 1}
                </div>
                {i < lineage.length - 1 && (
                  <div style={{ position: 'absolute', left: 41, top: 48, width: 2, height: 24, background: '#cbd5e1' }} />
                )}
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{stage.stage}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{stage.description}</div>
                </div>
                <span style={badgeStyle(stage.status === 'active' ? '#22c55e' : '#94a3b8')}>{stage.status}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ── Tab: Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Clinical Data Management Concepts">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.concepts || []).map((c, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600, width: 200 }}>{c.term}</td>
                    <td style={tdStyle(i)}>{c.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
            <Card title="Quality Metrics Targets">
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Metric</th>
                    <th style={thStyle}>Target</th>
                    <th style={thStyle}>Method</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.quality_metrics || []).map((m, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{m.metric}</td>
                      <td style={tdStyle(i)}><span style={badgeStyle('#1e88e5')}>{m.target}</span></td>
                      <td style={tdStyle(i)}>{m.method}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card title="Compliance References">
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Standard</th>
                    <th style={thStyle}>Scope</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.compliance_references || []).map((r, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{r.standard}</td>
                      <td style={tdStyle(i)}>{r.scope}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>

          <div style={{ marginTop: 16 }}>
            <Card title="Remediation Strategies">
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Gap</th>
                    <th style={thStyle}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.remediation || []).map((r, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600, color: '#ef4444' }}>{r.gap}</td>
                      <td style={tdStyle(i)}>{r.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}

      {/* Footer */}
      <div style={{ marginTop: 24, padding: '12px 0', borderTop: '1px solid #e2e8f0', fontSize: 11, color: '#94a3b8', textAlign: 'center' }}>
        Clinical Data Manager Dashboard — Real data from clinical.db ({k.total_patients} patients, {fmt(k.total_records)} records, {k.total_tables} tables)
        {overview?.generated_at && ` — Generated: ${new Date(overview.generated_at).toLocaleString()}`}
      </div>
    </div>
  )
}
