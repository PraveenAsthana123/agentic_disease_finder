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
    'Approved': '#10b981', 'Conditionally Approved': '#06b6d4',
    'Under Review': '#f59e0b', 'Submitted': '#3b82f6',
    'Additional Info Requested': '#ef4444', 'Pre-submission': '#64748b',
    'Passed': '#10b981', 'Completed': '#3b82f6', 'In Progress': '#f59e0b',
    'Planned': '#94a3b8', 'Failed - Remediation': '#ef4444'
  }
  const color = map[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 11, textTransform: 'uppercase'
    }}>{status}</span>
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

export default function RegulatoryDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/regulatory/overview`),
          axios.get(`${API_URL}/regulatory/breakdown`),
          axios.get(`${API_URL}/regulatory/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load regulatory data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading regulatory data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Regulatory data not available</div>

  const tabs = ['overview', 'submissions', 'validation', 'audit', 'definitions']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Clinical Validation &amp; Regulatory Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        FDA/CE/MDR pathway tracking — {fmt(kpis.total_submissions)} submissions across {fmt(kpis.products_tracked)} products
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Submissions" value={kpis.total_submissions} />
              <KPI label="Approved" value={kpis.approved} color="#10b981" />
              <KPI label="Under Review" value={kpis.under_review} color="#f59e0b" />
              <KPI label="Approval Rate" value={kpis.approval_rate_pct} sub="%" color="#10b981" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Total Studies" value={kpis.total_studies} />
              <KPI label="Study Pass Rate" value={kpis.study_pass_rate_pct} sub="%" color="#3b82f6" />
              <KPI label="Avg Sensitivity" value={kpis.avg_sensitivity} color="#8b5cf6" />
              <KPI label="Avg AUC-ROC" value={kpis.avg_auc_roc} color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Pathway Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.pathway_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="pathway" type="category" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="count" name="Submissions" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.status_distribution || []} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ status, count }) => `${status} (${count})`} labelLine={false}>
                  {(overview.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Risk Class Distribution" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={overview.risk_class_distribution || []} dataKey="count" nameKey="risk_class" cx="50%" cy="50%" outerRadius={70} label={({ risk_class, count }) => `${risk_class} (${count})`} labelLine={false}>
                  {(overview.risk_class_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Phase Distribution" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={overview.phase_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="phase" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Submissions" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Submissions tab ── */}
      {tab === 'submissions' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Product Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Product', 'Submissions', 'Approved', 'Avg Validation Score'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.product_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.product}</td>
                      <td style={{ padding: '10px 12px' }}>{p.submissions}</td>
                      <td style={{ padding: '10px 12px' }}>{p.approved}</td>
                      <td style={{ padding: '10px 12px' }}>{fmt(p.avg_validation_score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="All Submissions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['ID', 'Product', 'Pathway', 'Risk', 'Phase', 'Status', 'Reviewer', 'Submitted', 'Val Score'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.submissions || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.submission_id}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 500 }}>{s.product_name}</td>
                      <td style={{ padding: '8px 10px' }}>{s.pathway}</td>
                      <td style={{ padding: '8px 10px' }}>{s.risk_class}</td>
                      <td style={{ padding: '8px 10px' }}>{s.phase}</td>
                      <td style={{ padding: '8px 10px' }}><StatusBadge status={s.status} /></td>
                      <td style={{ padding: '8px 10px' }}>{s.reviewer}</td>
                      <td style={{ padding: '8px 10px' }}>{s.submitted_date}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.validation_score)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Reviewer Workload">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Reviewer', 'Submissions', 'Approved'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.reviewer_workload || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{r.reviewer}</td>
                      <td style={{ padding: '10px 12px' }}>{r.submissions}</td>
                      <td style={{ padding: '10px 12px' }}>{r.approved}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Validation tab ── */}
      {tab === 'validation' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Study Type Performance">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Study Type', 'Count', 'Avg Sensitivity', 'Avg Specificity', 'Avg AUC-ROC'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.study_type_performance || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 500 }}>{s.study_type}</td>
                      <td style={{ padding: '10px 12px' }}>{s.count}</td>
                      <td style={{ padding: '10px 12px' }}>{fmt(s.avg_sensitivity)}</td>
                      <td style={{ padding: '10px 12px' }}>{fmt(s.avg_specificity)}</td>
                      <td style={{ padding: '10px 12px' }}>{fmt(s.avg_auc_roc)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Validation Studies">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Study ID', 'Submission', 'Type', 'Status', 'N', 'Sens.', 'Spec.', 'AUC', 'PI', 'Site'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.validation_studies || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.study_id}</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.submission_id}</td>
                      <td style={{ padding: '8px 10px' }}>{s.study_type}</td>
                      <td style={{ padding: '8px 10px' }}><StatusBadge status={s.status} /></td>
                      <td style={{ padding: '8px 10px' }}>{s.sample_size || '--'}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.sensitivity)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.specificity)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.auc_roc)}</td>
                      <td style={{ padding: '8px 10px' }}>{s.principal_investigator}</td>
                      <td style={{ padding: '8px 10px' }}>{s.site}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Audit tab ── */}
      {tab === 'audit' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Audit Category Distribution" span={1}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.audit_categories || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Recent Audit Trail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Timestamp', 'Submission', 'Action', 'Actor', 'Category', 'Document'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.audit_trail || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.timestamp}</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.submission_id}</td>
                      <td style={{ padding: '8px 10px' }}>{a.action}</td>
                      <td style={{ padding: '8px 10px' }}>{a.actor}</td>
                      <td style={{ padding: '8px 10px' }}>{a.category}</td>
                      <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.document_ref}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Regulatory Pathways" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {(defs.regulatory_pathways || []).map((p, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{p.pathway}</div>
                  <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{p.description}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Timeline: {p.timeline} | Evidence: {p.evidence}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Risk Classifications" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Class', 'Description', 'Examples', 'Controls'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(defs.risk_classifications || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{r.class}</td>
                      <td style={{ padding: '10px 12px' }}>{r.description}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{r.examples}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{r.regulatory_controls}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Validation Criteria" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Metric', 'Description', 'Threshold', 'Standard'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(defs.validation_criteria || []).map((v, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{v.metric}</td>
                      <td style={{ padding: '10px 12px' }}>{v.description}</td>
                      <td style={{ padding: '10px 12px', color: '#10b981', fontWeight: 500 }}>{v.threshold}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{v.standard}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Regulatory Standards" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Standard', 'Title', 'Scope'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', borderBottom: '2px solid #e2e8f0', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(defs.regulatory_standards || []).map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{s.standard}</td>
                      <td style={{ padding: '10px 12px' }}>{s.title}</td>
                      <td style={{ padding: '10px 12px', fontSize: 12 }}>{s.scope}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{g.term}</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{g.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="References" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: '#475569' }}>
              {(defs.references || []).map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
