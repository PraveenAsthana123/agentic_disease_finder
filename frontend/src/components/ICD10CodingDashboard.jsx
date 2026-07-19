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
  confirmed: '#22c55e',
  auto_coded: '#3b82f6',
  pending_review: '#f59e0b',
  rejected: '#ef4444'
}
const STATUS_LABELS = {
  confirmed: 'Confirmed',
  auto_coded: 'Auto-Coded',
  pending_review: 'Pending Review',
  rejected: 'Rejected'
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  const label = STATUS_LABELS[status] || status || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function ConfidenceBadge({ value }) {
  if (value == null) return <span style={{ color: '#94a3b8' }}>--</span>
  const pct = (value * 100).toFixed(0)
  const color = value >= 0.8 ? '#22c55e' : value >= 0.5 ? '#f59e0b' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{pct}%</span>
  )
}

export default function ICD10CodingDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [statusFilter, setStatusFilter] = useState('all')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/icd10-coding/overview`),
          axios.get(`${API_URL}/icd10-coding/breakdown`),
          axios.get(`${API_URL}/icd10-coding/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ICD-10 Coding data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No ICD-10 coding data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'records', label: 'Records' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const statusDistData = overview?.status_distribution
    ? Object.entries(overview.status_distribution).map(([k, v]) => ({
        name: STATUS_LABELS[k] || k, value: v, color: STATUS_COLORS[k] || '#94a3b8'
      }))
    : []

  const topCodesData = (overview?.top_codes || []).map(c => ({
    name: c.primary_code, count: c.cnt, desc: c.primary_desc
  }))

  const coderWorkloadData = overview?.coder_workload || []
  const timelineData = overview?.monthly_timeline || []

  /* Breakdown data prep */
  const rejectionDistData = breakdown?.rejection_distribution
    ? Object.entries(breakdown.rejection_distribution).map(([k, v]) => ({
        name: k.replace(/_/g, ' '), count: v
      }))
    : []

  const statusByCoderData = breakdown?.status_by_coder
    ? Object.entries(breakdown.status_by_coder).map(([coder, statuses]) => ({
        name: coder,
        ...(typeof statuses === 'object' ? statuses : {})
      }))
    : []

  /* Records tab data */
  const recentRecords = breakdown?.recent_records || []
  const filteredRecords = statusFilter === 'all'
    ? recentRecords
    : recentRecords.filter(r => r.status === statusFilter)

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>ICD-10 Coding Records Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Diagnostic coding analytics, AI auto-coding workflow, and accuracy tracking
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Cards */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 16 }}>
              <KPI label="Total Records" value={fmt(overview.total_records)} />
              <KPI label="Patients" value={fmt(overview.total_patients)} color="#3b82f6" />
              <KPI label="Coders" value={fmt(overview.total_coders)} color="#8b5cf6" />
              <KPI label="Auto-Coded %" value={fmt(overview.auto_coded_pct) + '%'} color="#06b6d4" />
              <KPI label="Confirmed %" value={fmt(overview.confirmed_pct) + '%'} color="#22c55e" />
              <KPI label="Rejected" value={fmt(overview.rejected_count)} color="#ef4444" />
              <KPI label="Avg Confidence" value={overview.avg_confidence != null ? (overview.avg_confidence * 100).toFixed(0) + '%' : '--'} color="#f59e0b" />
            </div>
          </Card>

          {/* Pie: Status Distribution */}
          <Card title="Status Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={statusDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {statusDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {statusDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Top ICD-10 Codes (horizontal) */}
          <Card title="Top ICD-10 Codes" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(220, topCodesData.length * 24)}>
              <BarChart data={topCodesData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={80} />
                <Tooltip formatter={(value, name, props) => [value, 'Count']}
                  labelFormatter={(label) => {
                    const item = topCodesData.find(c => c.name === label)
                    return item ? `${label}: ${item.desc}` : label
                  }} />
                <Bar dataKey="count" fill="#3b82f6" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar: Coder Workload */}
          <Card title="Coder Workload" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={coderWorkloadData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="coder" tick={{ fontSize: 9 }} angle={-15} textAnchor="end" height={55} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="confirmed" fill="#22c55e" name="Confirmed" stackId="a" />
                <Bar dataKey="pending" fill="#f59e0b" name="Pending" stackId="a" />
                <Bar dataKey="rejected" fill="#ef4444" name="Rejected" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: Monthly Volume Timeline */}
          <Card title="Monthly Volume Timeline" span={1}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="records" stroke="#3b82f6" name="Total Records" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="confirmed" stroke="#22c55e" name="Confirmed" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="auto_coded" stroke="#06b6d4" name="Auto-Coded" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Breakdown */}
      {tab === 'breakdown' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Bar: Rejection Reason Distribution */}
          <Card title="Rejection Reasons">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={rejectionDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={55} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Coder Summary Table */}
          <Card title="Per-Coder Summary">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Coder</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Records</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Confirmed Rate</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_coder_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.coder}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.total}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          padding: '2px 6px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: row.confirmed_rate_pct >= 80 ? '#22c55e22' : row.confirmed_rate_pct >= 50 ? '#f59e0b22' : '#ef444422',
                          color: row.confirmed_rate_pct >= 80 ? '#22c55e' : row.confirmed_rate_pct >= 50 ? '#f59e0b' : '#ef4444'
                        }}>{fmt(row.confirmed_rate_pct)}%</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <ConfidenceBadge value={row.avg_confidence} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Low Confidence Alerts (red header) */}
          <Card title="Low Confidence Alerts (< 60%)" span={2}>
            {(breakdown.low_confidence_records || []).length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No low confidence records.</div>
            ) : (
              <div style={{ maxHeight: 400, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#fef2f2', position: 'sticky', top: 0 }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#991b1b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#991b1b' }}>Date</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#991b1b' }}>Code</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#991b1b' }}>Description</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#991b1b' }}>Status</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#991b1b' }}>Confidence</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#991b1b' }}>Coder</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.low_confidence_records || []).map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{row.encounter_date || '--'}</td>
                        <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontWeight: 600 }}>{row.primary_code}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.primary_desc || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><ConfidenceBadge value={row.confidence} /></td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.coder || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Status-by-Coder Cross Table */}
          <Card title="Status by Coder" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Coder</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#22c55e' }}>Confirmed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#3b82f6' }}>Auto-Coded</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#f59e0b' }}>Pending</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#ef4444' }}>Rejected</th>
                  </tr>
                </thead>
                <tbody>
                  {statusByCoderData.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.confirmed || 0}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.auto_coded || 0}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.pending_review || 0}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.rejected || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Code Category Analysis Table */}
          <Card title="Code Category Analysis" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Confirmed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rejected</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Accuracy</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.code_category_analysis || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.category}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.total}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.confirmed}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.rejected}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <span style={{
                          padding: '2px 6px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                          background: row.accuracy_pct >= 80 ? '#22c55e22' : row.accuracy_pct >= 50 ? '#f59e0b22' : '#ef444422',
                          color: row.accuracy_pct >= 80 ? '#22c55e' : row.accuracy_pct >= 50 ? '#f59e0b' : '#ef4444'
                        }}>{fmt(row.accuracy_pct)}%</span>
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ConfidenceBadge value={row.avg_confidence} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Records */}
      {tab === 'records' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Status Filter */}
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>Filter by Status:</span>
              {['all', 'confirmed', 'auto_coded', 'pending_review', 'rejected'].map(s => (
                <button key={s} onClick={() => setStatusFilter(s)} style={{
                  padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 500,
                  background: statusFilter === s ? (s === 'all' ? '#1e293b' : (STATUS_COLORS[s] || '#1e293b')) : '#f1f5f9',
                  color: statusFilter === s ? '#fff' : '#475569'
                }}>{s === 'all' ? 'All' : (STATUS_LABELS[s] || s)}</button>
              ))}
              <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 8 }}>{filteredRecords.length} records</span>
            </div>
          </Card>

          {/* Recent Records Table */}
          <Card title="Recent Coding Records">
            <div style={{ maxHeight: 600, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Primary Code</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Secondary</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Coder</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Rejection Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>{row.encounter_date || '--'}</td>
                      <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontWeight: 600 }}>{row.primary_code}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.primary_desc || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.secondary_codes || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ConfidenceBadge value={row.confidence} /></td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.coder || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: row.rejection_reason ? '#ef4444' : '#94a3b8' }}>
                        {row.rejection_reason ? row.rejection_reason.replace(/_/g, ' ') : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* ICD-10 Code Categories */}
          {defs.code_categories && defs.code_categories.length > 0 && (
            <Card title="ICD-10 Code Categories">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.code_categories.map((cat, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 700, fontSize: 14, color: '#3b82f6', marginBottom: 4 }}>{cat.category}</div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{cat.name}</div>
                    <div style={{ fontSize: 12, color: '#475569' }}>{cat.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Status Descriptions */}
          {defs.status_descriptions && defs.status_descriptions.length > 0 && (
            <Card title="Status Descriptions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.status_descriptions.map((sd, i) => {
                  const color = STATUS_COLORS[sd.status] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: `4px solid ${color}` }}>
                      <div style={{ fontWeight: 700, fontSize: 13, color, marginBottom: 4 }}>{STATUS_LABELS[sd.status] || sd.status}</div>
                      <div style={{ fontSize: 12, color: '#475569' }}>{sd.description}</div>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Rejection Reasons */}
          {defs.rejection_reasons && defs.rejection_reasons.length > 0 && (
            <Card title="Rejection Reasons">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.rejection_reasons.map((rr, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#ef4444', marginBottom: 4 }}>{rr.reason.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#475569' }}>{rr.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Confidence Thresholds */}
          {defs.confidence_thresholds && defs.confidence_thresholds.length > 0 && (
            <Card title="Confidence Thresholds">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.confidence_thresholds.map((ct, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: `4px solid ${ct.color}` }}>
                    <div style={{ fontWeight: 700, fontSize: 14, color: ct.color, marginBottom: 2 }}>{ct.range}</div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{ct.label}</div>
                    <div style={{ fontSize: 12, color: '#475569' }}>{ct.description}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Clinical Notes */}
          {defs.clinical_notes && defs.clinical_notes.length > 0 && (
            <Card title="Clinical Notes">
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                  {defs.clinical_notes.map((note, i) => <li key={i} style={{ marginBottom: 6 }}>{note}</li>)}
                </ul>
              </div>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.glossary.map((g, i) => (
                  <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{g.term}</div>
                    <div style={{ fontSize: 12, color: '#475569' }}>{g.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
