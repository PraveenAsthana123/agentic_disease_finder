import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const CODING_STATUS_COLORS = {
  auto_coded: '#3b82f6',
  confirmed: '#22c55e',
  pending_review: '#eab308',
  rejected: '#ef4444'
}
const CODING_STATUS_LABELS = {
  auto_coded: 'Auto-Coded',
  confirmed: 'Confirmed',
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

function CodingStatusBadge({ status }) {
  const color = CODING_STATUS_COLORS[status] || '#94a3b8'
  const label = CODING_STATUS_LABELS[status] || status || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
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
    { id: 'coding_detail', label: 'Coding Detail' },
    { id: 'accuracy', label: 'Accuracy' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const categoryDistData = overview?.category_distribution
    ? Object.entries(overview.category_distribution).map(([k, v]) => ({
        name: k, value: v, color: ['#3b82f6','#22c55e','#eab308','#ef4444','#8b5cf6','#06b6d4','#f97316','#ec4899','#14b8a6','#6366f1'][
          Object.keys(overview.category_distribution).indexOf(k) % 10
        ]
      }))
    : []

  const topCodesData = overview?.top_codes || []

  const timelineData = overview?.coding_timeline || []

  /* Breakdown data prep */
  const recentCodings = breakdown?.recent_codings || []

  /* Accuracy data prep */
  const accuracyByCategoryData = breakdown?.accuracy_by_category || []

  const rejectionReasonsData = breakdown?.rejection_reasons
    ? Object.entries(breakdown.rejection_reasons).map(([k, v]) => ({
        name: k, value: v, color: ['#ef4444','#f97316','#eab308','#8b5cf6','#3b82f6','#06b6d4'][
          Object.keys(breakdown.rejection_reasons).indexOf(k) % 6
        ]
      }))
    : []

  const coderWorkloadData = breakdown?.coder_workload || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>ICD-10 Medical Coding Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Automated ICD-10 coding, accuracy tracking, and review workflow management
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Encounters" value={fmt(overview.total_encounters)} />
              <KPI label="Coded" value={fmt(overview.coded_count)} />
              <KPI label="Auto-Coded" value={fmt(overview.auto_coded)} color="#3b82f6" />
              <KPI label="Confirmed" value={fmt(overview.confirmed)} color="#22c55e" />
              <KPI label="Pending Review" value={fmt(overview.pending_review)} color="#eab308" />
              <KPI label="Coding Accuracy" value={overview.coding_accuracy_pct != null ? fmt(overview.coding_accuracy_pct) + '%' : '--'} color="#22c55e" />
            </div>
          </Card>

          {/* Pie: ICD-10 Category Distribution */}
          <Card title="ICD-10 Category Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={categoryDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {categoryDistData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {categoryDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Top 10 Codes */}
          <Card title="Top 10 ICD-10 Codes" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topCodesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="code" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" name="Occurrences" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Line: Coding Timeline */}
          <Card title="Coding Timeline (30 days)" span={3}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="auto_coded" stroke="#3b82f6" name="Auto-Coded" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="confirmed" stroke="#22c55e" name="Confirmed" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Coding Detail */}
      {tab === 'coding_detail' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Recent Codings">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Date</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Primary Code</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Secondary Codes</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Confidence</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Coder</th>
                  </tr>
                </thead>
                <tbody>
                  {recentCodings.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.patient_id || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.date || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#3b82f6', fontWeight: 600 }}>{row.primary_code || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.description || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#64748b' }}>
                        {row.secondary_codes && row.secondary_codes.length > 0
                          ? row.secondary_codes.join(', ')
                          : '--'}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><CodingStatusBadge status={row.status} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569', fontWeight: 600 }}>
                        {row.confidence != null ? fmt(row.confidence) + '%' : '--'}
                      </td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.coder || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Accuracy */}
      {tab === 'accuracy' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Bar: Accuracy by Category */}
          <Card title="Accuracy by Category" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={accuracyByCategoryData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="accuracy" fill="#22c55e" name="Accuracy %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pie: Rejection Reasons */}
          <Card title="Rejection Reasons">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={rejectionReasonsData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {rejectionReasonsData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {rejectionReasonsData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Table: Coder Workload */}
          <Card title="Coder Workload">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Coder</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Reviewed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Confirmed</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rejected</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Accuracy %</th>
                  </tr>
                </thead>
                <tbody>
                  {coderWorkloadData.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.coder || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{fmt(row.reviewed)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#22c55e', fontWeight: 600 }}>{fmt(row.confirmed)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#ef4444', fontWeight: 600 }}>{fmt(row.rejected)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569', fontWeight: 600 }}>
                        {row.accuracy != null ? fmt(row.accuracy) + '%' : '--'}
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
          {/* ICD-10 Chapter Reference */}
          {defs.chapter_reference && defs.chapter_reference.length > 0 && (
            <Card title="ICD-10 Chapter Reference">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Chapter</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Code Range</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.chapter_reference.map((ch, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155' }}>{ch.chapter || ch.name}</td>
                      <td style={{ padding: '6px 8px', color: '#3b82f6', fontWeight: 600, fontSize: 12 }}>{ch.code_range || ch.range || '--'}</td>
                      <td style={{ padding: '6px 8px', color: '#475569' }}>{ch.description || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Status Definitions */}
          {defs.status_definitions && defs.status_definitions.length > 0 && (
            <Card title="Status Definitions">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                {defs.status_definitions.map((sd, i) => {
                  const color = CODING_STATUS_COLORS[sd.status] || '#94a3b8'
                  return (
                    <div key={i} style={{ padding: '12px 16px', background: color + '11', border: `1px solid ${color}33`, borderRadius: 8 }}>
                      <h4 style={{ margin: '0 0 8px', fontSize: 13, color }}>{CODING_STATUS_LABELS[sd.status] || sd.status}</h4>
                      <p style={{ margin: 0, fontSize: 12, color: '#475569' }}>{sd.description}</p>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}

          {/* Methodology */}
          {defs.methodology && (
            <Card title="Methodology">
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                {defs.methodology.description && (
                  <p style={{ margin: '0 0 12px', fontSize: 13, color: '#475569' }}>{defs.methodology.description}</p>
                )}
                {defs.methodology.steps && defs.methodology.steps.length > 0 && (
                  <ol style={{ margin: 0, padding: '0 0 0 20px', fontSize: 12, color: '#64748b' }}>
                    {defs.methodology.steps.map((s, i) => <li key={i} style={{ marginBottom: 4 }}>{s}</li>)}
                  </ol>
                )}
                {defs.methodology.factors && defs.methodology.factors.length > 0 && (
                  <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse', marginTop: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Factor</th>
                        <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Weight</th>
                        <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {defs.methodology.factors.map((f, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px 8px', fontWeight: 600, color: '#334155' }}>{f.factor || f.name}</td>
                          <td style={{ padding: '6px 8px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{f.weight || '--'}</td>
                          <td style={{ padding: '6px 8px', color: '#475569' }}>{f.description || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </Card>
          )}

          {/* Glossary */}
          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
                <tbody>
                  {defs.glossary.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 220 }}>{d.term}</td>
                      <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* References */}
          {defs.references && (
            <Card>
              <div style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 13, color: '#334155' }}>References</h4>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#64748b' }}>
                  {defs.references.map((ref, i) => <li key={i} style={{ marginBottom: 4 }}>{ref}</li>)}
                </ul>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
