import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'

const COLORS = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']

const STATUS_COLORS = {
  'Approved': '#22c55e',
  'Under Review': '#3b82f6',
  'Submitted': '#f59e0b',
  'Pre-submission': '#8b5cf6',
  'Additional Info Requested': '#ef4444'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status || 'unknown'}</span>
  )
}

function RiskBadge({ risk }) {
  const colors = { 'Class I': '#22c55e', 'Class IIa': '#f59e0b', 'Class IIb': '#f97316', 'Class III': '#ef4444' }
  const color = colors[risk] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{risk || '--'}</span>
  )
}

export default function RegulatorySubmissionsDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdownData, setBreakdownData] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoint = tab === 'overview' ? 'overview' : tab === 'breakdown' ? 'breakdown' : 'definitions'
    axios.get(`${API_URL}/regulatory-submissions/${endpoint}`)
      .then(res => {
        if (tab === 'overview') setOverview(res.data)
        else if (tab === 'breakdown') setBreakdownData(res.data)
        else setDefinitions(res.data)
        setLoading(false)
      })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'definitions', label: 'Definitions' }
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading regulatory submissions...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const data = tab === 'overview' ? overview : tab === 'breakdown' ? breakdownData : definitions
  if (!data || data.available === false) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{data?.note || 'No data available'}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Regulatory Submissions</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>FDA / CE pathway tracking &amp; submission lifecycle analytics</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview tab */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {/* KPI row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 12 }}>
              <KPI label="Total Submissions" value={fmt(overview.kpis?.total_submissions)} />
              <KPI label="Products" value={fmt(overview.kpis?.total_products)} />
              <KPI label="Pathways" value={fmt(overview.kpis?.total_pathways)} />
              <KPI label="Reviewers" value={fmt(overview.kpis?.total_reviewers)} />
              <KPI label="Approved" value={fmt(overview.kpis?.approved_count)} color="#22c55e" />
              <KPI label="Approval Rate" value={overview.kpis?.approval_rate != null ? overview.kpis.approval_rate + '%' : '--'} />
              <KPI label="Avg Validation" value={fmt(overview.kpis?.avg_validation_score)} color="#3b82f6" />
            </div>
          </Card>

          {/* Status pie */}
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.status_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status}: ${count}`}>
                  {(overview.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Pathway bar chart */}
          <Card title="Submissions by Pathway">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.pathway_distribution || []} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="pathway" width={80} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk class bar */}
          <Card title="Risk Class Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.risk_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="risk_class" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Product breakdown */}
          <Card title="Products">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.product_breakdown || []} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="product" width={100} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Phase distribution */}
          <Card title="Phase Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.phase_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="phase" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly timeline */}
          <Card title="Monthly Submission Timeline" span={3}>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={overview.monthly_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="submissions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Breakdown tab */}
      {tab === 'breakdown' && breakdownData && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {/* Reviewer workload */}
          <Card title="Reviewer Workload" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Reviewer</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Total</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Approved</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Avg Score</th>
                </tr>
              </thead>
              <tbody>
                {(breakdownData.reviewer_workload || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}>{r.reviewer}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.total}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#22c55e', fontWeight: 600 }}>{r.approved}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.avg_validation_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Per-product */}
          <Card title="Per-Product Summary">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Product</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Subs</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Avg Score</th>
                </tr>
              </thead>
              <tbody>
                {(breakdownData.per_product || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 500 }}>{p.product}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.submissions}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(p.avg_validation_score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Overdue alert */}
          {(breakdownData.overdue_submissions || []).length > 0 && (
            <Card title="Overdue / At-Risk Submissions" span={3}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Submission</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Product</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Pathway</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Status</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Target Date</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Reviewer</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdownData.overdue_submissions || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.submission_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.product}</td>
                      <td style={{ padding: '6px 10px' }}>{r.pathway}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={r.status} /></td>
                      <td style={{ padding: '6px 10px', color: '#ef4444', fontWeight: 600 }}>{r.target_date}</td>
                      <td style={{ padding: '6px 10px' }}>{r.reviewer}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Pathway-Status cross-tab */}
          <Card title="Pathway × Status Cross-Tab" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Pathway</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Status</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {(breakdownData.pathway_status_crosstab || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}>{r.pathway}</td>
                    <td style={{ padding: '6px 10px' }}><StatusBadge status={r.status} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Validation scores */}
          <Card title="Validation Scores">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Product</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Score</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdownData.validation_scores || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px' }}>{r.product}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: r.score >= 0.8 ? '#22c55e' : r.score >= 0.6 ? '#f59e0b' : '#ef4444' }}>{r.score?.toFixed(3)}</td>
                    <td style={{ padding: '6px 10px' }}><StatusBadge status={r.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Recent submissions table */}
          <Card title="All Submissions" span={3}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>ID</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Product</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Pathway</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Classification</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Status</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Risk</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Phase</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Submitted</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Score</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Reviewer</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdownData.recent_submissions || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.submission_id}</td>
                      <td style={{ padding: '6px 10px', fontWeight: 500 }}>{r.product}</td>
                      <td style={{ padding: '6px 10px' }}>{r.pathway}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.classification}</td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={r.status} /></td>
                      <td style={{ padding: '6px 10px' }}><RiskBadge risk={r.risk_class} /></td>
                      <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.phase}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.submitted_date}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600, color: r.validation_score >= 0.8 ? '#22c55e' : r.validation_score ? '#f59e0b' : '#94a3b8' }}>{r.validation_score?.toFixed(3) || '--'}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.reviewer}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {/* Pathways */}
          <Card title="Regulatory Pathways">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.pathways || []).map((p, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{p.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Statuses */}
          <Card title="Submission Statuses">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.statuses || []).map((s, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{s.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Risk classes */}
          <Card title="Risk Classes">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.risk_classes || []).map((r, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{r.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{r.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Phases */}
          <Card title="Regulatory Phases">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.phases || []).map((p, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{p.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{p.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Classifications */}
          <Card title="SaMD Classifications">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.classifications || []).map((c, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{c.name}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{c.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Clinical notes */}
          <Card title="Clinical Notes">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {(definitions.clinical_notes || []).map((n, i) => (
                <div key={i} style={{ padding: 8, background: '#f0fdf4', borderRadius: 6, fontSize: 12, color: '#166534' }}>{n}</div>
              ))}
            </div>
          </Card>

          {/* Glossary */}
          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              {(definitions.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{g.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Field descriptions */}
          <Card title="Field Descriptions">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.field_descriptions || []).map((f, i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', fontFamily: 'monospace' }}>{f.field}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{f.description}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
