import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function StatusBadge({ status }) {
  const map = {
    'Completed': '#10b981',
    'Passed': '#3b82f6',
    'In Progress': '#f59e0b',
    'Planned': '#8b5cf6',
    'Failed - Remediation': '#ef4444',
  }
  const color = map[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{status}</span>
  )
}

function AUCBadge({ auc }) {
  if (auc == null) return <span style={{ color: '#94a3b8' }}>--</span>
  const color = auc >= 0.9 ? '#10b981' : auc >= 0.8 ? '#3b82f6' : auc >= 0.7 ? '#f59e0b' : '#ef4444'
  const label = auc >= 0.9 ? 'Outstanding' : auc >= 0.8 ? 'Excellent' : auc >= 0.7 ? 'Acceptable' : 'Poor'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12
    }}>{auc.toFixed(3)} ({label})</span>
  )
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

export default function ValidationStudiesDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/validation-studies/overview`),
      axios.get(`${API_URL}/validation-studies/breakdown`),
      axios.get(`${API_URL}/validation-studies/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
      setLoading(false)
    }).catch(e => {
      setError(e.message)
      setLoading(false)
    })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading validation studies...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const tabs = ['overview', 'studies', 'submissions', 'definitions']
  const k = overview.kpis || {}

  const thStyle = { padding: '8px 12px', textAlign: 'left', fontSize: 13, color: '#475569', borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }
  const tdStyle = { padding: '8px 12px', fontSize: 13, borderBottom: '1px solid #e2e8f0' }

  return (
    <div style={{ padding: '24px 32px', background: '#f1f5f9', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Validation Studies Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Clinical validation, software verification &amp; analytical performance tracking across {k.total_sites} international sites
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 20px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t ? '#3b82f6' : '#e2e8f0',
            color: tab === t ? '#fff' : '#64748b',
            fontWeight: 600, fontSize: 13
          }}>
            {t === 'overview' ? 'Overview' : t === 'studies' ? 'Studies' : t === 'submissions' ? 'Submissions' : 'Definitions'}
          </button>
        ))}
      </div>

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Studies" value={k.total_studies} />
              <KPI label="Submissions" value={k.total_submissions} />
              <KPI label="Pass Rate" value={k.pass_rate_pct != null ? k.pass_rate_pct + '%' : '--'} color={k.pass_rate_pct >= 50 ? '#10b981' : '#ef4444'} />
              <KPI label="Avg Sensitivity" value={k.avg_sensitivity} color="#3b82f6" />
              <KPI label="Avg AUC-ROC" value={k.avg_auc_roc} color={k.avg_auc_roc >= 0.9 ? '#10b981' : '#f59e0b'} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Avg Specificity" value={k.avg_specificity} color="#8b5cf6" />
              <KPI label="Avg Sample Size" value={k.avg_sample_size} />
              <KPI label="Sites" value={k.total_sites} />
              <KPI label="Principal Investigators" value={k.total_pis} />
            </div>
          </Card>

          {/* Study type distribution pie */}
          <Card title="Study Type Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={overview.study_type_distribution || []} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={90} label={({ type, pct }) => `${type.split(' ')[0]} ${pct}%`}>
                  {(overview.study_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Status distribution bar */}
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={overview.status_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="status" width={140} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Site distribution bar */}
          <Card title="Studies by Site">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={overview.site_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="site" tick={{ fontSize: 11, angle: -20 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Performance by study type table */}
          <Card title="Performance by Study Type" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Study Type</th>
                    <th style={thStyle}>Studies</th>
                    <th style={thStyle}>Avg Sensitivity</th>
                    <th style={thStyle}>Avg Specificity</th>
                    <th style={thStyle}>Avg AUC-ROC</th>
                    <th style={thStyle}>Avg Sample</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.performance_by_type || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><strong>{r.type}</strong></td>
                      <td style={tdStyle}>{r.studies}</td>
                      <td style={tdStyle}>{fmt(r.avg_sensitivity)}</td>
                      <td style={tdStyle}>{fmt(r.avg_specificity)}</td>
                      <td style={tdStyle}><AUCBadge auc={r.avg_auc_roc} /></td>
                      <td style={tdStyle}>{r.avg_sample_size?.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Performance by site table */}
          <Card title="Performance by Site">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Site</th>
                    <th style={thStyle}>Studies</th>
                    <th style={thStyle}>Avg AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.performance_by_site || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><strong>{r.site}</strong></td>
                      <td style={tdStyle}>{r.studies}</td>
                      <td style={tdStyle}><AUCBadge auc={r.avg_auc_roc} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── STUDIES TAB ─── */}
      {tab === 'studies' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Failed studies alert */}
          {(breakdown.failed_studies || []).length > 0 && (
            <Card title={`Failed Studies — Remediation Required (${breakdown.failed_studies.length})`}>
              <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12, marginBottom: 8 }}>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Study ID</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Submission</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Type</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Site</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>PI</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Sample</th>
                        <th style={{ ...thStyle, background: '#fef2f2', color: '#991b1b' }}>Findings</th>
                      </tr>
                    </thead>
                    <tbody>
                      {breakdown.failed_studies.map((r, i) => (
                        <tr key={i}>
                          <td style={tdStyle}><strong>{r.study_id}</strong></td>
                          <td style={tdStyle}>{r.submission_id}</td>
                          <td style={tdStyle}>{r.study_type}</td>
                          <td style={tdStyle}>{r.site}</td>
                          <td style={tdStyle}>{r.principal_investigator}</td>
                          <td style={tdStyle}>{r.sample_size?.toLocaleString()}</td>
                          <td style={tdStyle}>{r.findings}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          )}

          {/* In-progress studies */}
          {(breakdown.in_progress_studies || []).length > 0 && (
            <Card title={`In-Progress Studies (${breakdown.in_progress_studies.length})`}>
              <div style={{ background: '#fffbeb', borderRadius: 8, padding: 12 }}>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Study ID</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Submission</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Type</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Title</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Site</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>PI</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Sample</th>
                        <th style={{ ...thStyle, background: '#fffbeb', color: '#92400e' }}>Started</th>
                      </tr>
                    </thead>
                    <tbody>
                      {breakdown.in_progress_studies.map((r, i) => (
                        <tr key={i}>
                          <td style={tdStyle}><strong>{r.study_id}</strong></td>
                          <td style={tdStyle}>{r.submission_id}</td>
                          <td style={tdStyle}>{r.study_type}</td>
                          <td style={tdStyle}>{r.title}</td>
                          <td style={tdStyle}>{r.site}</td>
                          <td style={tdStyle}>{r.principal_investigator}</td>
                          <td style={tdStyle}>{r.sample_size?.toLocaleString()}</td>
                          <td style={tdStyle}>{r.start_date}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          )}

          {/* Top performing studies */}
          <Card title="Top Performing Studies (by AUC-ROC)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Study ID</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Site</th>
                    <th style={thStyle}>PI</th>
                    <th style={thStyle}>Sensitivity</th>
                    <th style={thStyle}>Specificity</th>
                    <th style={thStyle}>AUC-ROC</th>
                    <th style={thStyle}>Sample</th>
                    <th style={thStyle}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.top_performing || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><strong>{r.study_id}</strong></td>
                      <td style={tdStyle}>{r.study_type}</td>
                      <td style={tdStyle}>{r.site}</td>
                      <td style={tdStyle}>{r.principal_investigator}</td>
                      <td style={tdStyle}>{fmt(r.sensitivity)}</td>
                      <td style={tdStyle}>{fmt(r.specificity)}</td>
                      <td style={tdStyle}><AUCBadge auc={r.auc_roc} /></td>
                      <td style={tdStyle}>{r.sample_size?.toLocaleString()}</td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* PI workload */}
          <Card title="Principal Investigator Workload">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Principal Investigator</th>
                    <th style={thStyle}>Studies</th>
                    <th style={thStyle}>Passed</th>
                    <th style={thStyle}>Failed</th>
                    <th style={thStyle}>Avg Sample Size</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.pi_workload || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><strong>{r.principal_investigator}</strong></td>
                      <td style={tdStyle}>{r.studies}</td>
                      <td style={tdStyle}>{r.passed}</td>
                      <td style={{ ...tdStyle, color: r.failed > 0 ? '#ef4444' : undefined, fontWeight: r.failed > 0 ? 600 : undefined }}>{r.failed}</td>
                      <td style={tdStyle}>{r.avg_sample_size ? Number(r.avg_sample_size).toLocaleString() : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* All studies */}
          <Card title={`All Studies (${(breakdown.all_studies || []).length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Study ID</th>
                    <th style={thStyle}>Submission</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Site</th>
                    <th style={thStyle}>Sensitivity</th>
                    <th style={thStyle}>Specificity</th>
                    <th style={thStyle}>AUC-ROC</th>
                    <th style={thStyle}>Sample</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Dates</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.all_studies || []).map((r, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><strong>{r.study_id}</strong></td>
                      <td style={tdStyle}>{r.submission_id}</td>
                      <td style={tdStyle}>{r.study_type}</td>
                      <td style={tdStyle}>{r.site}</td>
                      <td style={tdStyle}>{fmt(r.sensitivity)}</td>
                      <td style={tdStyle}>{fmt(r.specificity)}</td>
                      <td style={tdStyle}><AUCBadge auc={r.auc_roc} /></td>
                      <td style={tdStyle}>{r.sample_size?.toLocaleString()}</td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                      <td style={tdStyle}>{r.start_date} — {r.end_date || 'ongoing'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── SUBMISSIONS TAB ─── */}
      {tab === 'submissions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Per-Submission Summary (${(breakdown.per_submission || []).length} submissions)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Submission ID</th>
                    <th style={thStyle}>Total Studies</th>
                    <th style={thStyle}>Passed</th>
                    <th style={thStyle}>Failed</th>
                    <th style={thStyle}>In Progress</th>
                    <th style={thStyle}>Planned</th>
                    <th style={thStyle}>Avg Sensitivity</th>
                    <th style={thStyle}>Avg AUC</th>
                    <th style={thStyle}>Completion</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_submission || []).map((r, i) => {
                    const completionPct = r.total_studies > 0 ? Math.round((r.passed / r.total_studies) * 100) : 0
                    const barColor = r.failed > 0 ? '#ef4444' : completionPct >= 80 ? '#10b981' : completionPct >= 50 ? '#f59e0b' : '#94a3b8'
                    return (
                      <tr key={i}>
                        <td style={tdStyle}><strong>{r.submission_id}</strong></td>
                        <td style={tdStyle}>{r.total_studies}</td>
                        <td style={{ ...tdStyle, color: '#10b981', fontWeight: 600 }}>{r.passed}</td>
                        <td style={{ ...tdStyle, color: r.failed > 0 ? '#ef4444' : undefined, fontWeight: r.failed > 0 ? 600 : undefined }}>{r.failed}</td>
                        <td style={{ ...tdStyle, color: r.in_progress > 0 ? '#f59e0b' : undefined }}>{r.in_progress}</td>
                        <td style={tdStyle}>{r.planned}</td>
                        <td style={tdStyle}>{fmt(r.avg_sensitivity)}</td>
                        <td style={tdStyle}><AUCBadge auc={r.avg_auc} /></td>
                        <td style={tdStyle}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ flex: 1, background: '#e2e8f0', borderRadius: 6, height: 8 }}>
                              <div style={{ width: `${completionPct}%`, background: barColor, borderRadius: 6, height: 8 }} />
                            </div>
                            <span style={{ fontSize: 12, fontWeight: 600, color: barColor }}>{completionPct}%</span>
                          </div>
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

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Study types */}
          <Card title="Study Types" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {(definitions.study_types || []).map((d, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.type}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{d.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Metrics */}
          <Card title="Performance Metrics">
            {(definitions.metrics || []).map((d, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12, marginBottom: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.metric}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{d.description}</div>
              </div>
            ))}
          </Card>

          {/* Statuses */}
          <Card title="Study Statuses">
            {(definitions.statuses || []).map((d, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12, marginBottom: 8 }}>
                <div style={{ marginBottom: 4 }}><StatusBadge status={d.status} /></div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{d.description}</div>
              </div>
            ))}
          </Card>

          {/* Regulatory context */}
          <Card title="Regulatory Context" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {(definitions.regulatory_context || []).map((d, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.item}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{d.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Glossary */}
          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {(definitions.glossary || []).map((d, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{d.definition}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
