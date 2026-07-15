import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const STATUS_COLORS = { published: '#10b981', under_review: '#f59e0b', draft: '#3b82f6', retired: '#94a3b8' }
const FINDING_COLORS = { compliant: '#10b981', observation: '#3b82f6', minor_nonconformance: '#f59e0b', major_nonconformance: '#ef4444' }
const SEVERITY_COLORS = { low: '#10b981', medium: '#f59e0b', high: '#f97316', critical: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'procedures', label: 'Procedures & Audits' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SOPComplianceDashboard() {
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
      axios.get(`${API_URL}/api/sop-compliance/overview`),
      axios.get(`${API_URL}/api/sop-compliance/breakdown`),
      axios.get(`${API_URL}/api/sop-compliance/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading SOP Compliance data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>SOP Compliance Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Standard Operating Procedure compliance tracking — audit findings, procedure status, standards coverage
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'procedures' && breakdown && <ProceduresTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

/* -- Badge helpers -------------------------------------------------------- */
function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11,
      fontWeight: 600, background: `${color}18`, color, textTransform: 'capitalize',
    }}>{(status || '').replace(/_/g, ' ')}</span>
  )
}

function FindingBadge({ type }) {
  const color = FINDING_COLORS[type] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11,
      fontWeight: 600, background: `${color}18`, color, textTransform: 'capitalize',
    }}>{(type || '').replace(/_/g, ' ')}</span>
  )
}

function SeverityBadge({ severity }) {
  const color = SEVERITY_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11,
      fontWeight: 600, background: `${color}18`, color, textTransform: 'capitalize',
    }}>{severity}</span>
  )
}

function ComplianceBar({ score }) {
  const color = score >= 80 ? '#10b981' : score >= 60 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 8, borderRadius: 4, background: '#e2e8f0' }}>
        <div style={{ width: `${Math.min(score, 100)}%`, height: '100%', borderRadius: 4, background: color }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color, minWidth: 36, textAlign: 'right' }}>{score}%</span>
    </div>
  )
}

/* -- Overview Tab --------------------------------------------------------- */
function OverviewTab({ data }) {
  const avgScore = data.avg_compliance_score ?? 0
  const scoreColor = avgScore >= 80 ? '#10b981' : avgScore >= 60 ? '#f59e0b' : '#ef4444'

  const statusDist = data.status_distribution || {}
  const statusPie = Object.entries(statusDist).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v, key: k }))

  const categoryData = (data.category_breakdown || []).map(c => ({
    ...c,
    fill: c.avg_score >= 80 ? '#10b981' : c.avg_score >= 60 ? '#f59e0b' : '#ef4444',
  }))

  const standardsData = (data.standards_coverage || []).map(s => ({ standard: s.standard, count: s.sop_count }))

  const findingDist = data.finding_type_distribution || {}
  const findingPie = Object.entries(findingDist).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v, key: k }))

  const sevDist = data.severity_distribution || {}
  const sevPie = Object.entries(sevDist).filter(([, v]) => v > 0).map(([k, v]) => ({ name: k, value: v, key: k }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      <Card title="Total SOPs">
        <KPI label="Procedures" value={data.total_procedures} color="#3b82f6" />
      </Card>
      <Card title="Avg Compliance">
        <KPI label="Compliance score" value={`${avgScore}%`} color={scoreColor} />
      </Card>
      <Card title="Overdue Reviews">
        <KPI label="Reviews overdue" value={data.overdue_reviews} color={data.overdue_reviews > 0 ? '#ef4444' : '#10b981'} />
      </Card>
      <Card title="Open Findings">
        <KPI label="Findings open" value={data.open_findings} color={data.open_findings > 0 ? '#ef4444' : '#10b981'} />
      </Card>
      <Card title="Total Audits">
        <KPI label="Audits performed" value={data.total_audits} color="#8b5cf6" />
      </Card>

      <Card title="SOP Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={statusPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, value }) => `${name} (${value})`}
              labelLine={false} fontSize={11}>
              {statusPie.map((entry, i) => (
                <Cell key={i} fill={STATUS_COLORS[entry.key] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Category Compliance" span={3}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={categoryData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="category" fontSize={11} angle={-20} textAnchor="end" height={60} />
            <YAxis domain={[0, 100]} fontSize={11} />
            <Tooltip formatter={(v) => [`${v}%`, 'Avg Score']} />
            <Bar dataKey="avg_score" radius={[4, 4, 0, 0]}>
              {categoryData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Standards Coverage" span={3}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={standardsData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} />
            <YAxis dataKey="standard" type="category" fontSize={11} width={140} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Finding Type Distribution" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={findingPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={80} label={({ name, value }) => `${value}`}
              labelLine={false} fontSize={11}>
              {findingPie.map((entry, i) => (
                <Cell key={i} fill={FINDING_COLORS[entry.key] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Distribution" span={1}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={sevPie} dataKey="value" nameKey="name" cx="50%" cy="50%"
              outerRadius={80} label={({ name, value }) => `${value}`}
              labelLine={false} fontSize={11}>
              {sevPie.map((entry, i) => (
                <Cell key={i} fill={SEVERITY_COLORS[entry.key] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

/* -- Procedures & Audits Tab ---------------------------------------------- */
function ProceduresTab({ data }) {
  const procedures = data.procedures || []
  const audits = data.audits || []

  const thStyle = {
    padding: '8px 10px', textAlign: 'left', fontSize: 11, color: '#64748b',
    fontWeight: 600, borderBottom: '2px solid #e2e8f0', whiteSpace: 'nowrap',
  }
  const tdStyle = { padding: '8px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <Card title="SOP Procedures">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>SOP ID</th>
                <th style={thStyle}>Title</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Category</th>
                <th style={thStyle}>Owner</th>
                <th style={{ ...thStyle, minWidth: 120 }}>Compliance</th>
                <th style={thStyle}>Last Reviewed</th>
                <th style={thStyle}>Next Review</th>
                <th style={thStyle}>Standards</th>
                <th style={thStyle}>Rev</th>
              </tr>
            </thead>
            <tbody>
              {procedures.map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#334155' }}>{p.sop_id}</td>
                  <td style={tdStyle}>{p.title}</td>
                  <td style={tdStyle}><StatusBadge status={p.status} /></td>
                  <td style={tdStyle}>{p.category}</td>
                  <td style={tdStyle}>{p.owner}</td>
                  <td style={tdStyle}><ComplianceBar score={p.compliance_score} /></td>
                  <td style={tdStyle}>{p.last_reviewed}</td>
                  <td style={{
                    ...tdStyle,
                    color: p.is_overdue ? '#ef4444' : undefined,
                    fontWeight: p.is_overdue ? 600 : undefined,
                  }}>{p.next_review_due}{p.is_overdue ? ' (overdue)' : ''}</td>
                  <td style={tdStyle}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {(p.applicable_standards || []).map((s, j) => (
                        <span key={j} style={{
                          display: 'inline-block', padding: '1px 6px', borderRadius: 6,
                          fontSize: 10, background: '#e0e7ff', color: '#3730a3',
                        }}>{s}</span>
                      ))}
                    </div>
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>{p.revision_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Audit Findings">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Audit ID</th>
                <th style={thStyle}>SOP ID</th>
                <th style={thStyle}>Date</th>
                <th style={thStyle}>Auditor</th>
                <th style={thStyle}>Finding Type</th>
                <th style={{ ...thStyle, minWidth: 160 }}>Description</th>
                <th style={{ ...thStyle, minWidth: 140 }}>Corrective Action</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>Severity</th>
              </tr>
            </thead>
            <tbody>
              {audits.map((a, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                  <td style={{ ...tdStyle, fontWeight: 600, color: '#334155' }}>{a.audit_id}</td>
                  <td style={tdStyle}>{a.sop_id}</td>
                  <td style={tdStyle}>{a.audit_date}</td>
                  <td style={tdStyle}>{a.auditor}</td>
                  <td style={tdStyle}><FindingBadge type={a.finding_type} /></td>
                  <td style={{ ...tdStyle, maxWidth: 220, whiteSpace: 'normal' }}>{a.finding_description}</td>
                  <td style={{ ...tdStyle, maxWidth: 200, whiteSpace: 'normal' }}>{a.corrective_action}</td>
                  <td style={tdStyle}><StatusBadge status={a.status} /></td>
                  <td style={tdStyle}><SeverityBadge severity={a.severity} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* -- Definitions Tab ------------------------------------------------------ */
function DefinitionsTab({ data }) {
  const terms = data.terms || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
      {terms.map((t, i) => (
        <Card key={i}>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>{t.term}</div>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{t.definition}</div>
        </Card>
      ))}
    </div>
  )
}
