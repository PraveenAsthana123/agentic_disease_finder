import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const riskColor = (level) => {
  const s = (level || '').toLowerCase()
  if (s === 'high' || s === 'critical') return '#ef4444'
  if (s === 'moderate' || s === 'medium') return '#f59e0b'
  if (s === 'low' || s === 'minimal') return '#22c55e'
  return '#94a3b8'
}

const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }
const tdStyle = (i) => ({ padding: '8px 12px', borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#f8fafc' })

export default function DataStewardDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/data-steward/overview`),
      axios.get(`${API_URL}/api/data-steward/breakdown`),
      axios.get(`${API_URL}/api/data-steward/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Steward / Privacy Officer Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>No data available</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'datasets', label: 'Dataset Registration' },
    { id: 'phi', label: 'PHI Assessment' },
    { id: 'eeg_video', label: 'EEG/Video Privacy' },
    { id: 'deidentification', label: 'De-identification' },
    { id: 'access', label: 'Access Control' },
    { id: 'data_sharing', label: 'Data Sharing' },
    { id: 'audit', label: 'Audit Trail' },
    { id: 'incidents', label: 'Incidents' },
    { id: 'retention', label: 'Retention' },
    { id: 'risk', label: 'Risk' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}
  const charts = overview?.charts || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        Data Steward / Privacy Officer Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        PHI assessment, de-identification status, access control, audit trails, incident monitoring, and data retention oversight
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
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Datasets" value={fmt(k.total_datasets)} color="#1e88e5" /></Card>
            <Card><KPI label="Total Patients" value={fmt(k.total_patients)} color="#7c4dff" /></Card>
            <Card><KPI label="PHI Fields Detected" value={fmt(k.phi_fields_detected)} color="#ef4444" /></Card>
            <Card><KPI label="De-identified" value={k.deidentified_pct != null ? `${k.deidentified_pct}%` : '--'} color="#22c55e" /></Card>
            <Card><KPI label="Unique Actors" value={fmt(k.unique_actors)} color="#f59e0b" /></Card>
            <Card><KPI label="Audit Events" value={fmt(k.audit_events)} color="#6366f1" /></Card>
            <Card><KPI label="Data Sharing Actions" value={fmt(k.data_sharing_actions)} color="#ec4899" /></Card>
            <Card><KPI label="Privacy Risk Score" value={k.privacy_risk_score != null ? `${k.privacy_risk_score}/100` : '--'} color="#14b8a6" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
            {/* PHI Exposure by Field */}
            <Card title="PHI Exposure by Field">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={charts.phi_exposure_by_field || []} layout="vertical" margin={{ left: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="field" width={70} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="records_exposed" fill="#ef4444" radius={[0, 4, 4, 0]}>
                    {(charts.phi_exposure_by_field || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Access by Actor */}
            <Card title="Access by Actor">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={charts.access_by_actor || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="actor" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]}>
                    {(charts.access_by_actor || []).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Actions Timeline */}
            <Card title="Actions Timeline">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={charts.actions_timeline || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Risk Distribution */}
            <Card title="Risk Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={charts.risk_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                    {(charts.risk_distribution || []).map((entry, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Risk Factors */}
          {(overview.risk_factors || []).length > 0 && (
            <Card title="Privacy Risk Factors">
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {overview.risk_factors.map((rf, i) => (
                  <span key={i} style={badgeStyle(rf.score > 15 ? '#ef4444' : rf.score > 5 ? '#f59e0b' : '#22c55e')}>
                    {rf.factor}: {rf.score}/{rf.max}
                  </span>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* ── Tab: Dataset Registration ── */}
      {tab === 'datasets' && (() => {
        const ds = breakdown?.dataset_registration || {}
        const recs = ds.records || []
        const sum = ds.summary || {}
        return (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
              <Card><KPI label="Total Uploads" value={fmt(sum.total_uploads)} color="#1e88e5" /></Card>
              <Card><KPI label="Unique Patients" value={fmt(sum.unique_patients_with_data)} color="#7c4dff" /></Card>
              <Card><KPI label="File Types" value={fmt((sum.file_type_distribution || []).length)} color="#22c55e" /></Card>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
              <Card title="File Type Distribution">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={sum.file_type_distribution || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="file_type" tick={{ fontSize: 11 }} />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                      {(sum.file_type_distribution || []).map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>

              <Card title="Registration Timeline">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={sum.registration_timeline || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </div>

            <Card title={`Registered Datasets (${recs.length})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>File Name</th>
                      <th style={thStyle}>Patient ID</th>
                      <th style={thStyle}>File Type</th>
                      <th style={thStyle}>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recs.map((row, i) => (
                      <tr key={i}>
                        <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.file_name || '--'}</td>
                        <td style={tdStyle(i)}>{row.patient_id || '--'}</td>
                        <td style={tdStyle(i)}><span style={badgeStyle(COLORS[0])}>{row.file_type || '--'}</span></td>
                        <td style={{ ...tdStyle(i), fontSize: 11 }}>{row.created_at || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {recs.length === 0 && (
                <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No dataset registrations available</div>
              )}
            </Card>
          </>
        )
      })()}

      {/* ── Tab: PHI Assessment ── */}
      {tab === 'phi' && (
        <Card title={`PHI Assessment (${(breakdown?.per_patient_phi || []).length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Patient ID</th>
                  <th style={thStyle}>Name</th>
                  <th style={thStyle}>PHI Fields</th>
                  <th style={thStyle}>PHI Count</th>
                  <th style={thStyle}>De-identified</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.per_patient_phi || []).map((row, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.patient_id || '--'}</td>
                    <td style={tdStyle(i)}>{row.name || '--'}</td>
                    <td style={tdStyle(i)}>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(row.fields_exposed || []).map((f, j) => (
                          <span key={j} style={badgeStyle('#f59e0b')}>{f}</span>
                        ))}
                      </div>
                    </td>
                    <td style={tdStyle(i)}>{fmt(row.phi_field_count)}</td>
                    <td style={tdStyle(i)}>
                      <span style={badgeStyle(row.is_deidentified ? '#22c55e' : '#ef4444')}>
                        {row.is_deidentified ? 'Yes' : 'No'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {(breakdown?.per_patient_phi || []).length === 0 && (
            <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No PHI assessment data available</div>
          )}
        </Card>
      )}

      {/* ── Tab: EEG/Video Privacy ── */}
      {tab === 'eeg_video' && (
        <>
          <Card title="EEG/Video File Privacy Assessment">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>File Name</th>
                    <th style={thStyle}>Patient ID</th>
                    <th style={thStyle}>File Type</th>
                    <th style={thStyle}>Risk Level</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.eeg_video_privacy || []).map((row, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.file_name || '--'}</td>
                      <td style={tdStyle(i)}>{row.patient_id || '--'}</td>
                      <td style={tdStyle(i)}>{row.extension || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(riskColor(row.privacy_risk))}>{row.privacy_risk || '--'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.eeg_video_privacy || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No EEG/video privacy data available</div>
            )}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="File Type Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown?.file_type_analysis || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="extension" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                  {(breakdown?.file_type_analysis || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* ── Tab: De-identification ── */}
      {tab === 'deidentification' && (() => {
        const deid = breakdown?.deidentification_detail || {}
        const perPat = deid.per_patient || []
        const sum = deid.summary || {}
        return (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
              <Card><KPI label="Total Patients" value={fmt(sum.total_patients)} color="#1e88e5" /></Card>
              <Card><KPI label="Fully De-identified" value={fmt(sum.fully_deidentified)} color="#22c55e" /></Card>
              <Card><KPI label="Partially De-identified" value={fmt(sum.partially_deidentified)} color="#f59e0b" /></Card>
              <Card><KPI label="Not De-identified" value={fmt(sum.not_deidentified)} color="#ef4444" /></Card>
            </div>

            <Card title="Safe Harbor Fields Checked">
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 }}>
                {(sum.safe_harbor_fields_checked || []).map((f, i) => (
                  <span key={i} style={badgeStyle('#6366f1')}>{f}</span>
                ))}
              </div>
            </Card>

            <div style={{ marginTop: 16 }} />

            <Card title="De-identification Status by Patient">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={perPat.slice(0, 30)} layout="vertical" margin={{ left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="patient_id" width={55} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="safe_harbor_compliance_pct" fill="#22c55e" radius={[0, 4, 4, 0]}>
                    {perPat.slice(0, 30).map((entry, i) => (
                      <Cell key={i} fill={entry.safe_harbor_compliance_pct >= 80 ? '#22c55e' : entry.safe_harbor_compliance_pct >= 50 ? '#f59e0b' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <div style={{ marginTop: 16 }} />

            <Card title={`Per-Patient De-identification Detail (${perPat.length})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Patient ID</th>
                      <th style={thStyle}>Identifiers Present</th>
                      <th style={thStyle}>Compliance %</th>
                      <th style={thStyle}>Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perPat.map((row, i) => (
                      <tr key={i}>
                        <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.patient_id || '--'}</td>
                        <td style={tdStyle(i)}>
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {(row.identifiers_present || []).map((f, j) => (
                              <span key={j} style={badgeStyle('#ef4444')}>{f}</span>
                            ))}
                            {(row.identifiers_present || []).length === 0 && <span style={{ color: '#22c55e', fontSize: 12 }}>None</span>}
                          </div>
                        </td>
                        <td style={tdStyle(i)}>
                          <span style={badgeStyle(row.safe_harbor_compliance_pct >= 80 ? '#22c55e' : row.safe_harbor_compliance_pct >= 50 ? '#f59e0b' : '#ef4444')}>
                            {row.safe_harbor_compliance_pct != null ? `${row.safe_harbor_compliance_pct}%` : '--'}
                          </span>
                        </td>
                        <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 300 }}>{row.recommendation || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )
      })()}

      {/* ── Tab: Access Control ── */}
      {tab === 'access' && (() => {
        const ap = breakdown?.actor_permissions || {}
        const actorRows = Object.entries(ap).map(([actor, actions]) => ({
          actor,
          actions,
          total: (actions || []).reduce((s, a) => s + (a.count || 0), 0),
        }))
        return (
        <>
          <Card title="Actor Permissions">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Actions</th>
                    <th style={thStyle}>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {actorRows.map((row, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.actor || '--'}</td>
                      <td style={tdStyle(i)}>
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                          {(row.actions || []).map((a, j) => (
                            <span key={j} style={badgeStyle(COLORS[j % COLORS.length])}>{a.action}: {a.count}</span>
                          ))}
                        </div>
                      </td>
                      <td style={tdStyle(i)}>{fmt(row.total)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {actorRows.length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No actor permission data available</div>
            )}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Access by Actor">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={charts.access_by_actor || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="actor" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]}>
                  {(charts.access_by_actor || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
        )
      })()}

      {/* ── Tab: Data Sharing ── */}
      {tab === 'data_sharing' && (() => {
        const sh = breakdown?.data_sharing_detail || {}
        const recs = sh.records || []
        const sum = sh.summary || {}
        return (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
              <Card><KPI label="Total Sharing Actions" value={fmt(sum.total_sharing_actions)} color="#1e88e5" /></Card>
              <Card><KPI label="Unique Actors" value={fmt((sum.sharing_by_actor || []).length)} color="#7c4dff" /></Card>
              <Card><KPI label="Action Types" value={fmt((sum.sharing_by_action_type || []).length)} color="#22c55e" /></Card>
            </div>

            {(sum.sharing_by_actor || []).length > 0 && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 20 }}>
                <Card title="Sharing by Actor">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={sum.sharing_by_actor || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="actor" tick={{ fontSize: 10 }} />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#7c4dff" radius={[4, 4, 0, 0]}>
                        {(sum.sharing_by_actor || []).map((_, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                <Card title="Sharing Timeline">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={sum.sharing_timeline || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                      <YAxis allowDecimals={false} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Card>
              </div>
            )}

            <Card title={`Data Sharing Log (${recs.length} events)`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Timestamp</th>
                      <th style={thStyle}>Actor</th>
                      <th style={thStyle}>Action</th>
                      <th style={thStyle}>Patient</th>
                      <th style={thStyle}>Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recs.map((row, i) => (
                      <tr key={i}>
                        <td style={{ ...tdStyle(i), fontSize: 11 }}>{row.ts_utc || '--'}</td>
                        <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.actor || '--'}</td>
                        <td style={tdStyle(i)}><span style={badgeStyle('#ec4899')}>{row.action || '--'}</span></td>
                        <td style={tdStyle(i)}>{row.patient_id || '--'}</td>
                        <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>{row.detail || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {recs.length === 0 && (
                <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No data sharing events recorded</div>
              )}
            </Card>
          </>
        )
      })()}

      {/* ── Tab: Audit Trail ── */}
      {tab === 'audit' && (
        <>
          <Card title={`Audit Log (${(breakdown?.access_log || []).length} entries)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Timestamp</th>
                    <th style={thStyle}>Actor</th>
                    <th style={thStyle}>Component</th>
                    <th style={thStyle}>Action</th>
                    <th style={thStyle}>Patient</th>
                    <th style={thStyle}>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.access_log || []).map((entry, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontSize: 11 }}>{entry.ts_utc || '--'}</td>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{entry.actor || '--'}</td>
                      <td style={tdStyle(i)}>{entry.component || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(entry.action === 'write' || entry.action === 'delete' ? '#ef4444' : '#1e88e5')}>
                          {entry.action || '--'}
                        </span>
                      </td>
                      <td style={tdStyle(i)}>{entry.patient_id || '--'}</td>
                      <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 250 }}>{entry.detail || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.access_log || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No audit log entries available</div>
            )}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Actions Timeline">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={charts.actions_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* ── Tab: Incidents ── */}
      {tab === 'incidents' && (() => {
        const ic = breakdown?.incident_candidates || {}
        const allIncidents = [
          ...(ic.multi_actor_access || []).map(r => ({ ...r, type: 'Multi-Actor Access' })),
          ...(ic.high_frequency_access || []).map(r => ({ ...r, type: 'High Frequency Access' })),
          ...(ic.broad_permission_actors || []).map(r => ({ ...r, type: 'Broad Permissions' })),
        ]
        return (
          <Card title={`Incident Candidates (${allIncidents.length} found)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Patient / Actor</th>
                    <th style={thStyle}>Reason</th>
                    <th style={thStyle}>Severity</th>
                  </tr>
                </thead>
                <tbody>
                  {allIncidents.map((row, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.type || '--'}</td>
                      <td style={tdStyle(i)}>{row.patient_id || row.actor || '--'}</td>
                      <td style={{ ...tdStyle(i), fontSize: 11, maxWidth: 350 }}>{row.reason || '--'}</td>
                      <td style={tdStyle(i)}>
                        <span style={badgeStyle(riskColor(row.severity))}>{row.severity || '--'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {allIncidents.length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No incident candidates detected</div>
            )}
          </Card>
        )
      })()}

      {/* ── Tab: Retention ── */}
      {tab === 'retention' && (
        <>
          <Card title="Data Retention Analysis">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={breakdown?.retention_analysis || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="record_count" fill="#1e88e5" radius={[4, 4, 0, 0]}>
                  {(breakdown?.retention_analysis || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Retention Buckets">
            <div style={{ overflowX: 'auto' }}>
              <table style={tableStyle}>
                <thead>
                  <tr>
                    <th style={thStyle}>Retention Period</th>
                    <th style={thStyle}>Record Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.retention_analysis || []).map((row, i) => (
                    <tr key={i}>
                      <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.period || '--'}</td>
                      <td style={tdStyle(i)}>{fmt(row.record_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(breakdown?.retention_analysis || []).length === 0 && (
              <div style={{ color: '#94a3b8', fontSize: 13, textAlign: 'center', padding: 20 }}>No retention data available</div>
            )}
          </Card>
        </>
      )}

      {/* ── Tab: Risk ── */}
      {tab === 'risk' && (() => {
        const ra = breakdown?.risk_assessment || {}
        const perPat = ra.per_patient || []
        const dist = ra.risk_distribution || []
        return (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
              {dist.map((d, i) => (
                <Card key={i}><KPI label={d.risk_level} value={fmt(d.count)} color={riskColor(d.risk_level)} /></Card>
              ))}
            </div>

            <Card title="Privacy Risk Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={dist} dataKey="count" nameKey="risk_level" cx="50%" cy="50%" outerRadius={100} label>
                    {dist.map((entry, i) => (
                      <Cell key={i} fill={riskColor(entry.risk_level)} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <div style={{ marginTop: 16 }} />

            <Card title={`Per-Patient Risk Assessment (${perPat.length})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={tableStyle}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Patient ID</th>
                      <th style={thStyle}>Risk Score</th>
                      <th style={thStyle}>Risk Level</th>
                      <th style={thStyle}>Risk Factors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perPat.map((row, i) => (
                      <tr key={i}>
                        <td style={{ ...tdStyle(i), fontWeight: 600 }}>{row.patient_id || '--'}</td>
                        <td style={tdStyle(i)}>
                          <span style={{ fontWeight: 700, color: riskColor(row.risk_level) }}>{row.risk_score ?? '--'}</span>
                          <span style={{ fontSize: 11, color: '#94a3b8' }}>/100</span>
                        </td>
                        <td style={tdStyle(i)}>
                          <span style={badgeStyle(riskColor(row.risk_level))}>{row.risk_level || '--'}</span>
                        </td>
                        <td style={tdStyle(i)}>
                          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                            {(row.risk_factors || []).map((f, j) => (
                              <span key={j} style={{ ...badgeStyle('#64748b'), fontSize: 10 }}>{f}</span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )
      })()}

      {/* ── Tab: Definitions ── */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="Data Stewardship / Privacy Concepts">
            {(defs.terms || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < (defs.terms || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{c.term}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{c.definition}</div>
              </div>
            ))}
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Quality Metrics">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Metric</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Target</th>
                </tr>
              </thead>
              <tbody>
                {(defs.quality_metrics || []).map((m, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{m.metric}</td>
                    <td style={tdStyle(i)}>{m.description}</td>
                    <td style={tdStyle(i)}><span style={badgeStyle('#22c55e')}>{m.target}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Compliance References">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Standard</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.compliance_references || []).map((r, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle(i), fontWeight: 600 }}>{r.standard}</td>
                    <td style={tdStyle(i)}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <div style={{ marginTop: 16 }} />

          <Card title="Remediation Strategies">
            {(defs.remediation_strategies || []).map((s, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < (defs.remediation_strategies || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{s.strategy}</div>
                <div style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>{s.description}</div>
              </div>
            ))}
          </Card>
        </>
      )}

      {/* Footer */}
      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 12, marginTop: 32, paddingBottom: 24 }}>
        Source: data_steward_module &middot; Datasets ({fmt(k.total_datasets)}) &middot; Patients ({fmt(k.total_patients)}) &middot; Audit Events ({fmt(k.audit_events)})
      </div>
    </div>
  )
}
