import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const s = String(status || '')
  const color = s === 'confirmed' ? '#10b981'
    : s === 'auto_coded' ? '#3b82f6'
    : s === 'pending_review' ? '#f59e0b'
    : s === 'rejected' ? '#ef4444'
    : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{s.replace(/_/g, ' ') || '--'}</span>
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

export default function ICD10CodingDashboard() {
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
          axios.get(`${API_URL}/api/icd10-coding/overview`),
          axios.get(`${API_URL}/api/icd10-coding/breakdown`),
          axios.get(`${API_URL}/api/icd10-coding/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load ICD-10 coding data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ICD-10 coding data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>ICD-10 coding data not available</div>

  const tabs = ['overview', 'records', 'coders', 'review', 'definitions']
  const kpis = overview.kpis || overview

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>ICD-10 Coding Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Diagnostic coding analytics and quality assurance — {fmt(kpis.total_records)} records across {fmt(kpis.total_patients)} patients
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Records" value={kpis.total_records} />
              <KPI label="Total Patients" value={kpis.total_patients} />
              <KPI label="Unique Codes" value={kpis.unique_codes} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Confirmation Rate" value={kpis.confirmation_rate != null ? `${fmt(kpis.confirmation_rate)}%` : '--'} color="#10b981" />
              <KPI label="Avg Confidence" value={kpis.avg_confidence} color="#3b82f6" />
              <KPI label="Secondary Coverage" value={kpis.secondary_coverage != null ? `${fmt(kpis.secondary_coverage)}%` : '--'} color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={overview.status_distribution || []} dataKey="value" nameKey="name"
                  cx="50%" cy="50%" outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.status_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Confidence Tiers" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={overview.confidence_tiers || []} dataKey="value" nameKey="name"
                  cx="50%" cy="50%" outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.confidence_tiers || []).map((_, i) => {
                    const tierColors = ['#10b981', '#f59e0b', '#ef4444']
                    return <Cell key={i} fill={tierColors[i] || COLORS[i % COLORS.length]} />
                  })}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="AI vs Human Coding" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.ai_vs_human || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="confirmed" name="Confirmed" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="rejected" name="Rejected" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Top ICD-10 Codes" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.top_codes || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="code" type="category" tick={{ fontSize: 11 }} width={80} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {(overview.top_codes || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Trend" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={overview.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" name="Codes" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {(overview.rejection_reasons || []).length > 0 && (
            <Card title="Rejection Reasons" span={2}>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={overview.rejection_reasons}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="reason" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="count" name="Count" fill="#ef4444" radius={[4, 4, 0, 0]}>
                    {(overview.rejection_reasons || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* ── Records tab ── */}
      {tab === 'records' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Recent Coding Activity">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Code</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Description</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Status</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Coder</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_activity || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id || row.patient || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#3b82f6', fontWeight: 600 }}>{row.code || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.description || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: (row.confidence || 0) >= 0.9 ? '#10b981' : (row.confidence || 0) >= 0.7 ? '#f59e0b' : '#ef4444' }}>{row.confidence != null ? row.confidence.toFixed(2) : '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.coder || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Code Chapter Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={breakdown.chapter_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="chapter" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(breakdown.chapter_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Total Codes</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Unique</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Conf</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Confirmed</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Rejected</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Latest</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id || row.patient || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.total_codes)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.unique)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: (row.avg_confidence || 0) >= 0.9 ? '#10b981' : (row.avg_confidence || 0) >= 0.7 ? '#f59e0b' : '#ef4444' }}>{row.avg_confidence != null ? row.avg_confidence.toFixed(2) : '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10b981' }}>{fmt(row.confirmed)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#ef4444' }}>{fmt(row.rejected)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.latest || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Coders tab ── */}
      {tab === 'coders' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Coder Volume" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.coder_breakdown || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="coder" type="category" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="count" name="Records" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {(overview.coder_breakdown || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Avg Confidence by Coder" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.coder_breakdown || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} domain={[0, 1]} />
                <YAxis dataKey="coder" type="category" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="avg_confidence" name="Avg Confidence" fill="#10b981" radius={[0, 4, 4, 0]}>
                  {(overview.coder_breakdown || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {breakdown && (
            <Card title="Coder Performance Detail" span={4}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Coder</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Total</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Confirmed</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Rejected</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Pending</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Conf</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Min Conf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.coder_performance || []).map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.coder || '--'}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.total)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10b981' }}>{fmt(row.confirmed)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: '#ef4444' }}>{fmt(row.rejected)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: '#f59e0b' }}>{fmt(row.pending)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: (row.avg_confidence || 0) >= 0.9 ? '#10b981' : (row.avg_confidence || 0) >= 0.7 ? '#f59e0b' : '#ef4444' }}>{row.avg_confidence != null ? row.avg_confidence.toFixed(2) : '--'}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: (row.min_confidence || 0) < 0.7 ? '#ef4444' : '#64748b' }}>{row.min_confidence != null ? row.min_confidence.toFixed(2) : '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Review tab ── */}
      {tab === 'review' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Pending Review Queue">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fffbeb' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>ID</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Code</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Description</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#92400e' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#92400e' }}>Coder</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.pending_review_queue || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: (row.confidence || 0) < 0.7 ? '#fef2f2' : undefined }}>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.id || '--'}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id || row.patient || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#3b82f6', fontWeight: 600 }}>{row.code || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.description || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: (row.confidence || 0) >= 0.9 ? '#10b981' : (row.confidence || 0) >= 0.7 ? '#f59e0b' : '#ef4444' }}>{row.confidence != null ? row.confidence.toFixed(2) : '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.coder || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Low-Confidence Records">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fef2f2' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>ID</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Code</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Description</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Coder</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#991b1b' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.low_confidence_records || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: '#fef2f2' }}>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.id || '--'}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id || row.patient || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.date || '').slice(0, 10)}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#3b82f6', fontWeight: 600 }}>{row.code || '--'}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.description || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: '#ef4444' }}>{row.confidence != null ? row.confidence.toFixed(2) : '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.coder || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
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
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {defs.statuses && (
            <Card title="Status Definitions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.statuses).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <StatusBadge status={key} />
                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.confidence_thresholds && (
            <Card title="Confidence Thresholds">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.confidence_thresholds).map(([key, val]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>
                      {typeof val === 'object' ? `${val.range || ''} — ${val.action || ''}` : String(val)}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.coding_workflow && (
            <Card title="Coding Workflow">
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {(Array.isArray(defs.coding_workflow) ? defs.coding_workflow : Object.values(defs.coding_workflow)).map((step, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>
            </Card>
          )}

          {defs.icd10_chapters && (
            <Card title="ICD-10 Chapter Guide">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.icd10_chapters).map(([code, desc]) => (
                  <div key={code} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                    <span style={{ fontFamily: 'monospace', color: '#3b82f6', fontWeight: 600, fontSize: 13, whiteSpace: 'nowrap' }}>{code}</span>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {defs.metrics && (
            <Card title="Metrics Glossary">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.metrics).map(([term, definition]) => (
                  <div key={term} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{term.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{definition}</div>
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
