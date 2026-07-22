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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function SeverityBadge({ severity }) {
  const s = String(severity || '')
  const color = s === 'critical' ? '#dc2626' : s === 'high' ? '#ef4444' : s === 'medium' ? '#f59e0b' : '#10b981'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s.replace(/_/g, ' ') || '--'}</span>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '')
  const color = s === 'closed' ? '#10b981' : s === 'in_progress' ? '#3b82f6' : s === 'open' ? '#ef4444'
    : s === 'published' ? '#10b981' : s === 'under_review' ? '#f59e0b' : s === 'draft' ? '#64748b'
    : s === 'retired' ? '#94a3b8' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'capitalize'
    }}>{s.replace(/_/g, ' ') || '--'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'procedures', label: 'Procedures' },
  { id: 'audits', label: 'Audit Findings' },
  { id: 'patients', label: 'By Patient' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ISSOPComplianceDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/is-sop-compliance/overview`),
      axios.get(`${API_URL}/api/is-sop-compliance/breakdown`),
      axios.get(`${API_URL}/api/is-sop-compliance/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading IS SOP compliance…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>IS SOP Compliance Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Standard operating procedure compliance — {fmt(overview?.total_sops)} SOPs, {fmt(overview?.total_audits)} audits, {fmt(overview?.open_findings)} open findings
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent', cursor: 'pointer'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'procedures' && <ProceduresTab procedures={breakdown?.procedures || []} />}
      {tab === 'audits' && <AuditsTab audits={breakdown?.audits || []} />}
      {tab === 'patients' && <PatientsTab patients={breakdown?.patients || []} />}
      {tab === 'definitions' && <DefinitionsTab data={defs} />}
    </div>
  )
}

function OverviewTab({ data }) {
  if (!data) return null
  const findingData = (data.finding_distribution || []).map(d => ({ name: d.type.replace(/_/g, ' '), value: d.count }))
  const categoryData = (data.category_distribution || []).map(d => ({ name: d.category, value: d.count }))
  const severityData = (data.severity_distribution || []).map(d => ({ name: d.severity, value: d.count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
          <KPI label="Total SOPs" value={data.total_sops} color="#3b82f6" />
          <KPI label="Total Audits" value={data.total_audits} color="#8b5cf6" />
          <KPI label="Avg Compliance" value={`${data.avg_compliance_score}%`} color="#10b981" />
          <KPI label="Audit Pass Rate" value={`${data.compliance_rate}%`} color="#06b6d4" />
          <KPI label="Open Findings" value={data.open_findings} color="#ef4444" />
          <KPI label="Overdue Reviews" value={data.overdue_reviews} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Finding Types">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={findingData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {findingData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="SOP Categories">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={categoryData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {categoryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Severity Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={severityData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Standards Coverage" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.standards_coverage || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" allowDecimals={false} />
            <YAxis dataKey="standard" type="category" width={100} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Monthly Audit Trend" span={3}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="audits" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="findings" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ProceduresTab({ procedures }) {
  const [sort, setSort] = useState({ key: 'sop_id', dir: 'asc' })
  const sorted = [...procedures].sort((a, b) => {
    const av = a[sort.key] ?? '', bv = b[sort.key] ?? ''
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggle = k => setSort(prev => ({ key: k, dir: prev.key === k && prev.dir === 'asc' ? 'desc' : 'asc' }))
  const th = (k, label) => (
    <th onClick={() => toggle(k)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #e2e8f0' }}>
      {label} {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )
  return (
    <Card title={`All Procedures (${procedures.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>{th('sop_id', 'SOP ID')}{th('title', 'Title')}{th('category', 'Category')}{th('status', 'Status')}{th('version', 'Version')}{th('compliance_score', 'Score')}{th('owner', 'Owner')}{th('last_reviewed', 'Last Reviewed')}{th('next_review_due', 'Next Review')}</tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.sop_id}</td>
                <td style={{ padding: '6px 10px' }}>{p.title}</td>
                <td style={{ padding: '6px 10px' }}>{p.category}</td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={p.status} /></td>
                <td style={{ padding: '6px 10px' }}>{p.version}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: (p.compliance_score || 0) >= 80 ? '#10b981' : (p.compliance_score || 0) >= 60 ? '#f59e0b' : '#ef4444' }}>{fmt(p.compliance_score)}</td>
                <td style={{ padding: '6px 10px' }}>{p.owner}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{p.last_reviewed}</td>
                <td style={{ padding: '6px 10px', fontSize: 12, color: (p.next_review_due || '') < '2026-07-22' ? '#ef4444' : '#10b981' }}>{p.next_review_due}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function AuditsTab({ audits }) {
  const [sort, setSort] = useState({ key: 'audit_id', dir: 'asc' })
  const sorted = [...audits].sort((a, b) => {
    const av = a[sort.key] ?? '', bv = b[sort.key] ?? ''
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggle = k => setSort(prev => ({ key: k, dir: prev.key === k && prev.dir === 'asc' ? 'desc' : 'asc' }))
  const th = (k, label) => (
    <th onClick={() => toggle(k)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #e2e8f0' }}>
      {label} {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )
  return (
    <Card title={`All Audit Findings (${audits.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>{th('audit_id', 'Audit ID')}{th('sop_id', 'SOP')}{th('audit_date', 'Date')}{th('auditor', 'Auditor')}{th('finding_type', 'Finding')}{th('severity', 'Severity')}{th('status', 'Status')}{th('finding_description', 'Description')}</tr>
          </thead>
          <tbody>
            {sorted.map((a, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.audit_id}</td>
                <td style={{ padding: '6px 10px' }}>{a.sop_id}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{a.audit_date}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{a.auditor}</td>
                <td style={{ padding: '6px 10px' }}>
                  <span style={{
                    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600,
                    background: a.finding_type === 'compliant' ? '#10b98122' : a.finding_type === 'observation' ? '#3b82f622' : '#ef444422',
                    color: a.finding_type === 'compliant' ? '#10b981' : a.finding_type === 'observation' ? '#3b82f6' : '#ef4444',
                  }}>{(a.finding_type || '').replace(/_/g, ' ')}</span>
                </td>
                <td style={{ padding: '6px 10px' }}><SeverityBadge severity={a.severity} /></td>
                <td style={{ padding: '6px 10px' }}><StatusBadge status={a.status} /></td>
                <td style={{ padding: '6px 10px', fontSize: 12, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.finding_description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PatientsTab({ patients }) {
  const [sort, setSort] = useState({ key: 'patient_id', dir: 'asc' })
  const sorted = [...patients].sort((a, b) => {
    const av = a[sort.key] ?? '', bv = b[sort.key] ?? ''
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))
    return sort.dir === 'asc' ? cmp : -cmp
  })
  const toggle = k => setSort(prev => ({ key: k, dir: prev.key === k && prev.dir === 'asc' ? 'desc' : 'asc' }))
  const th = (k, label) => (
    <th onClick={() => toggle(k)} style={{ padding: '8px 10px', cursor: 'pointer', fontSize: 12, color: '#475569', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #e2e8f0' }}>
      {label} {sort.key === k ? (sort.dir === 'asc' ? '▲' : '▼') : ''}
    </th>
  )
  return (
    <Card title={`Patient Compliance Summary (${patients.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr>{th('patient_id', 'Patient')}{th('sop_count', 'SOPs')}{th('audit_count', 'Audits')}{th('open_findings', 'Open Findings')}{th('avg_compliance_score', 'Avg Score')}</tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                <td style={{ padding: '6px 10px' }}>{p.sop_count}</td>
                <td style={{ padding: '6px 10px' }}>{p.audit_count}</td>
                <td style={{ padding: '6px 10px', color: p.open_findings > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{p.open_findings}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: (p.avg_compliance_score || 0) >= 80 ? '#10b981' : (p.avg_compliance_score || 0) >= 60 ? '#f59e0b' : '#ef4444' }}>{fmt(p.avg_compliance_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function DefinitionsTab({ data }) {
  if (!data) return null
  const Section = ({ title, items, renderItem }) => (
    <Card title={title}>
      <div style={{ display: 'grid', gap: 8 }}>
        {(items || []).map((item, i) => <div key={i}>{renderItem(item)}</div>)}
      </div>
    </Card>
  )
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Section title="Concepts" items={data.concepts} renderItem={c => (
        <div><strong style={{ color: '#1e293b' }}>{c.term}:</strong> <span style={{ color: '#475569' }}>{c.definition}</span></div>
      )} />
      <Section title="Severity Levels" items={data.severity_levels} renderItem={s => (
        <div><SeverityBadge severity={s.level} /> <span style={{ color: '#475569', marginLeft: 8 }}>{s.description}</span></div>
      )} />
      <Section title="SOP Categories" items={data.sop_categories} renderItem={c => (
        <div><strong style={{ color: '#1e293b' }}>{c.category}:</strong> <span style={{ color: '#475569' }}>{c.description}</span></div>
      )} />
      <Section title="Data Sources" items={data.data_sources} renderItem={d => (
        <div><code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 4, fontSize: 12 }}>{d.table}</code> <span style={{ color: '#475569', marginLeft: 4 }}>{d.description}</span></div>
      )} />
    </div>
  )
}
