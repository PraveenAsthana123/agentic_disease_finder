import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const styles = {
    'Approved':                 { bg: '#dcfce7', color: '#16a34a' },
    'Under Review':             { bg: '#dbeafe', color: '#1d4ed8' },
    'Pre-submission':           { bg: '#f1f5f9', color: '#475569' },
    'Additional Info Requested':{ bg: '#fef3c7', color: '#d97706' },
    'Withdrawn':                { bg: '#fee2e2', color: '#dc2626' },
  }
  const s = styles[status] || { bg: '#f1f5f9', color: '#475569' }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color
    }}>{status}</span>
  )
}

function RiskBadge({ risk }) {
  const styles = {
    'Class I':   { bg: '#dcfce7', color: '#16a34a' },
    'Class IIa': { bg: '#dbeafe', color: '#1d4ed8' },
    'Class IIb': { bg: '#fef3c7', color: '#d97706' },
    'Class III': { bg: '#fee2e2', color: '#dc2626' },
  }
  const s = styles[risk] || { bg: '#f1f5f9', color: '#475569' }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color
    }}>{risk}</span>
  )
}

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'submissions', label: 'All Submissions' },
  { id: 'reviewers',   label: 'Reviewer Workload' },
  { id: 'validation',  label: 'Validation Scores' },
  { id: 'definitions', label: 'Glossary' },
]

export default function RegulatorySubmissionsDashboard() {
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
      axios.get(`${API_URL}/api/regulatory-submissions/overview`),
      axios.get(`${API_URL}/api/regulatory-submissions/breakdown`),
      axios.get(`${API_URL}/api/regulatory-submissions/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Regulatory Submissions data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#0f172a' }}>Regulatory Submissions</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        FDA / CE Mark submission lifecycle, validation scores, and reviewer workload
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            background: tab === t.id ? '#1e293b' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* KPIs */}
          <Card title="Key Metrics" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 16 }}>
              <KPI label="Total Submissions" value={fmt(k.total_submissions)} />
              <KPI label="Products" value={fmt(k.total_products)} />
              <KPI label="Pathways" value={fmt(k.total_pathways)} />
              <KPI label="Reviewers" value={fmt(k.total_reviewers)} />
              <KPI label="Approved" value={fmt(k.approved_count)} color="#16a34a" />
              <KPI label="Approval Rate" value={k.approval_rate != null ? `${fmt(k.approval_rate)}%` : '--'} color="#3b82f6" />
              <KPI label="Avg Validation" value={k.avg_validation_score != null ? fmt(k.avg_validation_score) : '--'} color="#8b5cf6" />
            </div>
          </Card>

          {/* Status distribution pie */}
          <Card title="Status Distribution">
            {overview?.status_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={overview.status_distribution} dataKey="count" nameKey="status"
                    cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {overview.status_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Pathway distribution pie */}
          <Card title="Pathway Distribution">
            {overview?.pathway_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={overview.pathway_distribution} dataKey="count" nameKey="pathway"
                    cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {overview.pathway_distribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Risk class distribution bar */}
          <Card title="Risk Class Distribution">
            {overview?.risk_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview.risk_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="risk_class" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Product breakdown bar */}
          <Card title="Submissions by Product">
            {overview?.product_breakdown?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview.product_breakdown} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis dataKey="product" type="category" tick={{ fontSize: 11 }} width={120} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Phase distribution bar */}
          <Card title="Phase Distribution">
            {overview?.phase_distribution?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview.phase_distribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="phase" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Monthly timeline */}
          <Card title="Submissions Timeline" span={2}>
            {overview?.monthly_timeline?.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={overview.monthly_timeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="submissions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>
        </div>
      )}

      {/* ── ALL SUBMISSIONS TAB ── */}
      {tab === 'submissions' && (
        <div style={{ display: 'grid', gap: 16 }}>
          {/* Recent / all submissions table */}
          <Card title={`All Submissions (${breakdown?.recent_submissions?.length || 0})`} span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>ID</th>
                    <th style={{ padding: '8px 6px' }}>Product</th>
                    <th style={{ padding: '8px 6px' }}>Pathway</th>
                    <th style={{ padding: '8px 6px' }}>Classification</th>
                    <th style={{ padding: '8px 6px' }}>Status</th>
                    <th style={{ padding: '8px 6px' }}>Risk</th>
                    <th style={{ padding: '8px 6px' }}>Phase</th>
                    <th style={{ padding: '8px 6px' }}>Reviewer</th>
                    <th style={{ padding: '8px 6px' }}>Submitted</th>
                    <th style={{ padding: '8px 6px' }}>Target</th>
                    <th style={{ padding: '8px 6px' }}>Validation</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.recent_submissions || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 12 }}>{r.submission_id}</td>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.product}</td>
                      <td style={{ padding: '8px 6px' }}>{r.pathway}</td>
                      <td style={{ padding: '8px 6px', fontSize: 11 }}>{r.classification || '--'}</td>
                      <td style={{ padding: '8px 6px' }}><StatusBadge status={r.status} /></td>
                      <td style={{ padding: '8px 6px' }}><RiskBadge risk={r.risk_class} /></td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.phase || '--'}</td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.reviewer}</td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.submitted_date || '--'}</td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.target_date || '--'}</td>
                      <td style={{ padding: '8px 6px', fontWeight: 600, color: r.validation_score >= 0.9 ? '#16a34a' : r.validation_score >= 0.8 ? '#3b82f6' : '#d97706' }}>
                        {r.validation_score != null ? r.validation_score.toFixed(2) : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Overdue submissions */}
          {breakdown?.overdue_submissions?.length > 0 && (
            <Card title={`Overdue Submissions (${breakdown.overdue_submissions.length})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 6px' }}>ID</th>
                      <th style={{ padding: '8px 6px' }}>Product</th>
                      <th style={{ padding: '8px 6px' }}>Pathway</th>
                      <th style={{ padding: '8px 6px' }}>Status</th>
                      <th style={{ padding: '8px 6px' }}>Target Date</th>
                      <th style={{ padding: '8px 6px' }}>Reviewer</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.overdue_submissions.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                        <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 12 }}>{r.submission_id}</td>
                        <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.product}</td>
                        <td style={{ padding: '8px 6px' }}>{r.pathway}</td>
                        <td style={{ padding: '8px 6px' }}><StatusBadge status={r.status} /></td>
                        <td style={{ padding: '8px 6px', color: '#dc2626', fontWeight: 600 }}>{r.target_date}</td>
                        <td style={{ padding: '8px 6px' }}>{r.reviewer}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Pathway × Status crosstab */}
          <Card title="Pathway × Status Crosstab">
            {breakdown?.pathway_status_crosstab?.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 6px' }}>Pathway</th>
                      <th style={{ padding: '8px 6px' }}>Status</th>
                      <th style={{ padding: '8px 6px' }}>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.pathway_status_crosstab.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 6px' }}>{r.pathway}</td>
                        <td style={{ padding: '8px 6px' }}><StatusBadge status={r.status} /></td>
                        <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>
        </div>
      )}

      {/* ── REVIEWER WORKLOAD TAB ── */}
      {tab === 'reviewers' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* Reviewer workload bar */}
          <Card title="Reviewer Workload" span={2}>
            {breakdown?.reviewer_workload?.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={breakdown.reviewer_workload} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis dataKey="reviewer" type="category" tick={{ fontSize: 11 }} width={140} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="total" fill="#3b82f6" name="Total" radius={[0, 6, 6, 0]} />
                  <Bar dataKey="approved" fill="#10b981" name="Approved" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Reviewer table */}
          <Card title="Reviewer Detail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Reviewer</th>
                    <th style={{ padding: '8px 6px' }}>Total</th>
                    <th style={{ padding: '8px 6px' }}>Approved</th>
                    <th style={{ padding: '8px 6px' }}>Avg Validation</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.reviewer_workload || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.reviewer}</td>
                      <td style={{ padding: '8px 6px' }}>{r.total}</td>
                      <td style={{ padding: '8px 6px', color: '#16a34a' }}>{r.approved}</td>
                      <td style={{ padding: '8px 6px' }}>{r.avg_validation_score != null ? r.avg_validation_score.toFixed(2) : '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Per product summary */}
          <Card title="Per Product Summary" span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Product</th>
                    <th style={{ padding: '8px 6px' }}>Pathways</th>
                    <th style={{ padding: '8px 6px' }}>Statuses</th>
                    <th style={{ padding: '8px 6px' }}>Submissions</th>
                    <th style={{ padding: '8px 6px' }}>Avg Validation</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.per_product || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.product}</td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.pathways}</td>
                      <td style={{ padding: '8px 6px', fontSize: 12 }}>{r.statuses}</td>
                      <td style={{ padding: '8px 6px' }}>{r.submissions}</td>
                      <td style={{ padding: '8px 6px', fontWeight: 600, color: r.avg_validation_score >= 0.9 ? '#16a34a' : r.avg_validation_score >= 0.8 ? '#3b82f6' : '#d97706' }}>
                        {r.avg_validation_score != null ? r.avg_validation_score.toFixed(3) : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── VALIDATION SCORES TAB ── */}
      {tab === 'validation' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* Validation score bar */}
          <Card title="Validation Scores by Product" span={2}>
            {breakdown?.validation_scores?.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={breakdown.validation_scores}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="product" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                  <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip formatter={v => v.toFixed(3)} />
                  <Bar dataKey="score" fill="#8b5cf6" radius={[6, 6, 0, 0]}>
                    {(breakdown.validation_scores || []).map((entry, i) => (
                      <Cell key={i} fill={entry.score >= 0.9 ? '#10b981' : entry.score >= 0.8 ? '#3b82f6' : '#f59e0b'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <p style={{ color: '#94a3b8' }}>No data</p>}
          </Card>

          {/* Validation table */}
          <Card title="Score Detail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Product</th>
                    <th style={{ padding: '8px 6px' }}>Score</th>
                    <th style={{ padding: '8px 6px' }}>Pathway</th>
                    <th style={{ padding: '8px 6px' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.validation_scores || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.product}</td>
                      <td style={{ padding: '8px 6px', fontWeight: 700,
                        color: r.score >= 0.9 ? '#16a34a' : r.score >= 0.8 ? '#3b82f6' : '#d97706'
                      }}>{r.score != null ? r.score.toFixed(3) : '--'}</td>
                      <td style={{ padding: '8px 6px' }}>{r.pathway}</td>
                      <td style={{ padding: '8px 6px' }}><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── GLOSSARY TAB ── */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          {/* Pathways */}
          <Card title="Regulatory Pathways">
            {(definitions.pathways || []).map((p, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{p.description}</div>
              </div>
            ))}
          </Card>

          {/* Statuses */}
          <Card title="Submission Statuses">
            {(definitions.statuses || []).map((s, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{s.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.description}</div>
              </div>
            ))}
          </Card>

          {/* Risk classes */}
          <Card title="Risk Classes">
            {(definitions.risk_classes || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{r.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{r.description}</div>
              </div>
            ))}
          </Card>

          {/* Phases */}
          <Card title="Regulatory Phases">
            {(definitions.phases || []).map((p, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{p.description}</div>
              </div>
            ))}
          </Card>

          {/* Classifications */}
          <Card title="SaMD Classifications">
            {(definitions.classifications || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{c.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{c.description}</div>
              </div>
            ))}
          </Card>

          {/* Field descriptions */}
          <Card title="Field Descriptions">
            {(definitions.field_descriptions || []).map((f, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <span style={{ fontWeight: 600, fontSize: 12, fontFamily: 'monospace', color: '#6366f1' }}>{f.field}</span>
                <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>{f.description}</span>
              </div>
            ))}
          </Card>

          {/* Glossary */}
          <Card title="Glossary" span={2}>
            {(definitions.glossary || []).map((g, i) => (
              <div key={i} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #f1f5f9' }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{g.definition}</div>
              </div>
            ))}
          </Card>

          {/* Clinical notes */}
          {definitions.clinical_notes?.length > 0 && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
                {definitions.clinical_notes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
              </ul>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
