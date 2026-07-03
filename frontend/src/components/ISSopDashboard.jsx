import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const STATUS_COLORS = {
  published: '#2ecc71',
  draft: '#f39c12',
  under_review: '#3498db',
  retired: '#95a5a6'
}

const SEVERITY_COLORS = {
  critical: '#e74c3c',
  high: '#e67e22',
  medium: '#f39c12',
  low: '#2ecc71'
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

function StatusBadge({ status }) {
  const key = status?.toLowerCase()
  return <Badge text={status || '--'} color={STATUS_COLORS[key] || '#64748b'} />
}

function SeverityBadge({ severity }) {
  const key = severity?.toLowerCase()
  return <Badge text={severity || '--'} color={SEVERITY_COLORS[key] || '#64748b'} />
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'procedures', label: 'Procedure Index' },
  { id: 'audit', label: 'Compliance Audit' },
  { id: 'detail', label: 'SOP Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ISSopDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/is-sop/overview`),
      axios.get(`${API_URL}/api/is-sop/breakdown`),
      axios.get(`${API_URL}/api/is-sop/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const toggleExpand = id => setExpanded(prev => ({ ...prev, [id]: !prev[id] }))

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading IS SOP Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  const complianceColor = v => {
    if (v == null) return '#64748b'
    if (v >= 90) return '#2ecc71'
    if (v >= 70) return '#f39c12'
    return '#e74c3c'
  }

  const isOverdue = dateStr => {
    if (!dateStr) return false
    return new Date(dateStr) < new Date()
  }

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        IS SOP Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Information Security Standard Operating Procedures — compliance tracking, audit findings, and procedure management
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row 1 */}
          <Card title="Total SOPs"><KPI value={ov.total_sops} label="Procedures" color="#3b82f6" /></Card>
          <Card title="Published"><KPI value={ov.published} label="Active SOPs" color="#2ecc71" /></Card>
          <Card title="Reviews Due"><KPI value={ov.reviews_due} label="Pending review" color="#f59e0b" /></Card>
          <Card title="Overdue"><KPI value={ov.overdue} label="Past due date" color="#e74c3c" /></Card>

          {/* KPI Row 2 */}
          <Card title="Avg Compliance %"><KPI value={ov.avg_compliance != null ? `${ov.avg_compliance.toFixed(1)}%` : '--'} label="Compliance score" color="#10b981" /></Card>
          <Card title="Open Findings"><KPI value={ov.open_findings} label="Unresolved" color="#ef4444" /></Card>
          <Card title="Total Audits"><KPI value={ov.total_audits} label="Audit records" color="#8b5cf6" /></Card>
          <Card title="Closed Findings"><KPI value={ov.closed_findings} label="Resolved" color="#06b6d4" /></Card>

          {/* Status Distribution Pie Chart */}
          <Card title="Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.status_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={e => `${e.status}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.status_distribution || []).map((e, i) => (
                    <Cell key={i} fill={STATUS_COLORS[e.status?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Category Distribution Bar Chart */}
          <Card title="Category Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.category_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="SOPs">
                  {(ov.category_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Compliance by Category Bar Chart */}
          <Card title="Compliance by Category" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.compliance_by_category || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="avg_compliance" name="Avg Compliance %" radius={[4, 4, 0, 0]}>
                  {(ov.compliance_by_category || []).map((e, i) => (
                    <Cell key={i} fill={complianceColor(e.avg_compliance)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Severity Distribution Pie Chart */}
          <Card title="Finding Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.severity_distribution || []} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={e => `${e.severity}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={SEVERITY_COLORS[e.severity?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PROCEDURE INDEX TAB */}
      {tab === 'procedures' && (
        <Card title="SOP Procedure Index">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['SOP ID', 'Title', 'Version', 'Status', 'Category', 'Owner', 'Compliance Score', 'Next Review Due', 'Revision Count'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.sops || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(s.sop_id)}</td>
                    <td style={{ padding: '6px' }}>{fmt(s.title)}</td>
                    <td style={{ padding: '6px' }}>{fmt(s.version)}</td>
                    <td style={{ padding: '6px' }}><StatusBadge status={s.status} /></td>
                    <td style={{ padding: '6px' }}>{fmt(s.category)}</td>
                    <td style={{ padding: '6px' }}>{fmt(s.owner)}</td>
                    <td style={{ padding: '6px', fontWeight: 600, color: complianceColor(s.compliance_score) }}>
                      {s.compliance_score != null ? `${s.compliance_score}%` : '--'}
                    </td>
                    <td style={{ padding: '6px', color: isOverdue(s.next_review_due) ? '#e74c3c' : '#475569', fontWeight: isOverdue(s.next_review_due) ? 600 : 400 }}>
                      {fmt(s.next_review_due)}
                    </td>
                    <td style={{ padding: '6px' }}>{fmt(s.revision_count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* COMPLIANCE AUDIT TAB */}
      {tab === 'audit' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Finding Type Distribution Pie Chart */}
          <Card title="Finding Type Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={ov.finding_type_distribution || []} dataKey="count" nameKey="finding_type" cx="50%" cy="50%" outerRadius={90} label={e => `${e.finding_type}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.finding_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Audit Records Table */}
          <Card title="Audit Records">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['Audit ID', 'SOP ID', 'Date', 'Auditor', 'Finding Type', 'Severity', 'Status', 'Finding Description', 'Corrective Action'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.audits || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(a.audit_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(a.sop_id)}</td>
                      <td style={{ padding: '6px' }}>{fmt(a.date)}</td>
                      <td style={{ padding: '6px' }}>{fmt(a.auditor)}</td>
                      <td style={{ padding: '6px' }}><Badge text={a.finding_type || '--'} color="#3498db" /></td>
                      <td style={{ padding: '6px' }}><SeverityBadge severity={a.severity} /></td>
                      <td style={{ padding: '6px' }}><StatusBadge status={a.status} /></td>
                      <td style={{ padding: '6px', maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fmt(a.finding_description)}</td>
                      <td style={{ padding: '6px', maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{fmt(a.corrective_action)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* SOP DETAIL TAB */}
      {tab === 'detail' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Top 5 Non-Compliant SOPs */}
          <Card title="Top 5 Non-Compliant SOPs">
            {(() => {
              const sorted = [...(bd.sops || [])].filter(s => s.compliance_score != null).sort((a, b) => a.compliance_score - b.compliance_score).slice(0, 5)
              return sorted.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
                  {sorted.map((s, i) => (
                    <div key={i} style={{ textAlign: 'center', padding: 16, borderRadius: 8, background: '#ef444410', border: '1px solid #ef444430' }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>{s.sop_id}</div>
                      <div style={{ fontSize: 24, fontWeight: 700, color: complianceColor(s.compliance_score) }}>{s.compliance_score}%</div>
                      <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>{s.title}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: '#64748b', fontSize: 13 }}>No compliance data available.</p>
              )
            })()}
          </Card>

          {/* Expandable SOP Detail Rows */}
          <Card title="Per-SOP Detail with Associated Audits">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    {['', 'SOP ID', 'Title', 'Version', 'Status', 'Category', 'Compliance', 'Owner'].map(h => (
                      <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.sops || []).map((s, i) => {
                    const isExp = expanded[s.sop_id]
                    const sopAudits = (bd.audits || []).filter(a => a.sop_id === s.sop_id)
                    return (
                      <React.Fragment key={i}>
                        <tr
                          style={{ borderBottom: '1px solid #e2e8f0', cursor: 'pointer' }}
                          onClick={() => toggleExpand(s.sop_id)}
                        >
                          <td style={{ padding: '6px', fontSize: 14, transition: 'transform 0.2s', transform: isExp ? 'rotate(90deg)' : 'rotate(0deg)', display: 'inline-block', width: 20 }}>&#9654;</td>
                          <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(s.sop_id)}</td>
                          <td style={{ padding: '6px' }}>{fmt(s.title)}</td>
                          <td style={{ padding: '6px' }}>{fmt(s.version)}</td>
                          <td style={{ padding: '6px' }}><StatusBadge status={s.status} /></td>
                          <td style={{ padding: '6px' }}>{fmt(s.category)}</td>
                          <td style={{ padding: '6px', fontWeight: 600, color: complianceColor(s.compliance_score) }}>
                            {s.compliance_score != null ? `${s.compliance_score}%` : '--'}
                          </td>
                          <td style={{ padding: '6px' }}>{fmt(s.owner)}</td>
                        </tr>
                        {isExp && (
                          <tr>
                            <td colSpan={8} style={{ padding: '12px 20px', background: '#f8fafc' }}>
                              <div>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, marginBottom: 16 }}>
                                  {/* SOP Details */}
                                  <div>
                                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>SOP Details</h4>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                                      <div><span style={{ color: '#64748b' }}>SOP ID:</span> {fmt(s.sop_id)}</div>
                                      <div><span style={{ color: '#64748b' }}>Version:</span> {fmt(s.version)}</div>
                                      <div><span style={{ color: '#64748b' }}>Status:</span> <StatusBadge status={s.status} /></div>
                                      <div><span style={{ color: '#64748b' }}>Category:</span> {fmt(s.category)}</div>
                                      <div><span style={{ color: '#64748b' }}>Owner:</span> {fmt(s.owner)}</div>
                                      <div><span style={{ color: '#64748b' }}>Compliance:</span> <span style={{ fontWeight: 600, color: complianceColor(s.compliance_score) }}>{s.compliance_score != null ? `${s.compliance_score}%` : '--'}</span></div>
                                      <div><span style={{ color: '#64748b' }}>Next Review:</span> <span style={{ color: isOverdue(s.next_review_due) ? '#e74c3c' : '#475569' }}>{fmt(s.next_review_due)}</span></div>
                                      <div><span style={{ color: '#64748b' }}>Revision Count:</span> {fmt(s.revision_count)}</div>
                                    </div>
                                  </div>
                                  {/* Description */}
                                  <div>
                                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Description</h4>
                                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{fmt(s.description)}</div>
                                  </div>
                                </div>

                                {/* Associated Audits */}
                                <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Associated Audits ({sopAudits.length})</h4>
                                {sopAudits.length > 0 ? (
                                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                                    <thead>
                                      <tr style={{ background: '#e2e8f0' }}>
                                        {['Audit ID', 'Date', 'Auditor', 'Finding Type', 'Severity', 'Status', 'Description'].map(h => (
                                          <th key={h} style={{ padding: '6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {sopAudits.map((a, ai) => (
                                        <tr key={ai} style={{ borderBottom: '1px solid #e2e8f0' }}>
                                          <td style={{ padding: '4px 6px' }}>{fmt(a.audit_id)}</td>
                                          <td style={{ padding: '4px 6px' }}>{fmt(a.date)}</td>
                                          <td style={{ padding: '4px 6px' }}>{fmt(a.auditor)}</td>
                                          <td style={{ padding: '4px 6px' }}><Badge text={a.finding_type || '--'} color="#3498db" /></td>
                                          <td style={{ padding: '4px 6px' }}><SeverityBadge severity={a.severity} /></td>
                                          <td style={{ padding: '4px 6px' }}><StatusBadge status={a.status} /></td>
                                          <td style={{ padding: '4px 6px' }}>{fmt(a.finding_description)}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                ) : (
                                  <p style={{ color: '#94a3b8', fontSize: 12 }}>No audits recorded for this SOP.</p>
                                )}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Information Security SOP Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
