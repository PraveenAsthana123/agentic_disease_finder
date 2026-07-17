import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line
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
      padding: '2px 8px', borderRadius: 9999, fontSize: 11, fontWeight: 500,
      background: color + '20', color: color
    }}>{text}</span>
  )
}

const STATUS_COLORS = {
  confirmed: '#10b981',
  auto_coded: '#3b82f6',
  pending_review: '#f59e0b',
  rejected: '#ef4444',
}

function statusColor(status) {
  return STATUS_COLORS[status] || '#64748b'
}

function accuracyColor(pct) {
  if (pct >= 90) return '#10b981'
  if (pct >= 75) return '#3b82f6'
  if (pct >= 60) return '#f59e0b'
  return '#ef4444'
}

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Breakdown' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ICD10CodingDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/icd10-coding/overview`),
      axios.get(`${API_URL}/api/icd10-coding/breakdown`),
      axios.get(`${API_URL}/api/icd10-coding/definitions`),
    ]).then(([o, b, d]) => {
      setOv(o.data); setBd(b.data); setDefs(d.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ICD-10 coding data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>ICD-10 Coding Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Diagnostic coding analytics — {ov?.total_encounters} encounters, {ov?.total_coded} coded, {ov?.coding_accuracy}% accuracy
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t.id ? '#1e293b' : '#f1f5f9', color: tab === t.id ? '#fff' : '#475569'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && ov && renderOverview(ov)}
      {tab === 'breakdown' && bd && renderBreakdown(bd)}
      {tab === 'definitions' && defs && renderDefinitions(defs)}
    </div>
  )
}

function renderOverview(ov) {
  const categoryData = ov.category_distribution
    ? Object.entries(ov.category_distribution).map(([name, value]) => ({ name, value }))
    : []

  const timelineData = ov.coding_timeline || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {/* KPI Row */}
      <Card title="Coding Summary" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12 }}>
          <KPI label="Total Encounters" value={ov.total_encounters} />
          <KPI label="Total Coded" value={ov.total_coded} />
          <KPI label="Auto-Coded" value={ov.auto_coded_count} color="#3b82f6" />
          <KPI label="Confirmed" value={ov.confirmed_count} color="#10b981" />
          <KPI label="Pending Review" value={ov.pending_review_count} color="#f59e0b" />
          <KPI label="Coding Accuracy" value={`${ov.coding_accuracy}%`} color={accuracyColor(ov.coding_accuracy)} />
        </div>
      </Card>

      {/* Category Distribution Pie */}
      <Card title="Category Distribution">
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={categoryData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name.length > 20 ? name.substring(0, 20) + '...' : name}: ${value}`}>
              {categoryData.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Top Codes Bar (Horizontal) */}
      <Card title="Top ICD-10 Codes" span={2}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={ov.top_codes || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="code" type="category" width={100} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value, name) => [value, name]} labelFormatter={(label) => {
              const item = (ov.top_codes || []).find(c => c.code === label)
              return item ? `${label}: ${item.description}` : label
            }} />
            <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Coding Timeline */}
      <Card title="Coding Timeline" span={3}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="auto_coded" stroke="#3b82f6" name="Auto-Coded" strokeWidth={2} />
            <Line type="monotone" dataKey="confirmed" stroke="#10b981" name="Confirmed" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderBreakdown(bd) {
  const rejectionData = bd.rejection_reasons
    ? Object.entries(bd.rejection_reasons).map(([reason, count]) => ({ reason: reason.replace(/_/g, ' '), count }))
    : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {/* Recent Codings Table */}
      <Card title="Recent Codings" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Name', 'Date', 'Primary Code', 'Description', 'Secondary', 'Status', 'Confidence', 'Coder'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(bd.recent_codings || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500, fontFamily: 'monospace' }}>{r.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{r.patient_name}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11 }}>{r.encounter_date}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontWeight: 600 }}>{r.primary_code}</td>
                  <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.primary_desc}</td>
                  <td style={{ padding: '6px 10px' }}>
                    {(r.secondary_codes || []).map((sc, j) => (
                      <Badge key={j} text={sc} color="#64748b" />
                    ))}
                    {(!r.secondary_codes || r.secondary_codes.length === 0) && <span style={{ color: '#94a3b8' }}>--</span>}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={r.status?.replace(/_/g, ' ')} color={statusColor(r.status)} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    {r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '--'}
                  </td>
                  <td style={{ padding: '6px 10px' }}>{r.coder}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Code Accuracy by Category */}
      <Card title="Code Accuracy by Category" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Category', 'Total', 'Correct', 'Accuracy', ''].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(bd.code_accuracy_by_category || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{c.category}</td>
                  <td style={{ padding: '6px 10px' }}>{c.total}</td>
                  <td style={{ padding: '6px 10px' }}>{c.correct}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={`${c.accuracy}%`} color={accuracyColor(c.accuracy)} /></td>
                  <td style={{ padding: '6px 10px' }}>
                    <div style={{ background: '#e2e8f0', borderRadius: 4, height: 8, width: 100 }}>
                      <div style={{ background: accuracyColor(c.accuracy), borderRadius: 4, height: 8, width: `${Math.min(c.accuracy, 100)}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Rejection Reasons Bar Chart */}
      <Card title="Rejection Reasons">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={rejectionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="reason" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" name="Count" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Coder Workload Table */}
      <Card title="Coder Workload" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Coder', 'Total', 'Confirmed', 'Pending'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(bd.coder_workload || []).map((w, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{w.coder}</td>
                  <td style={{ padding: '6px 10px' }}>{w.total}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={w.confirmed} color="#10b981" /></td>
                  <td style={{ padding: '6px 10px' }}><Badge text={w.pending} color="#f59e0b" /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderDefinitions(defs) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      {/* ICD-10 Chapters */}
      <Card title="ICD-10 Chapters (Relevant Ranges)" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Chapter', 'Title', 'Relevant Ranges'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(defs.icd10_chapters || []).map((ch, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{ch.chapter || ch.number}</td>
                  <td style={{ padding: '6px 10px', color: '#475569' }}>{ch.title || ch.name}</td>
                  <td style={{ padding: '6px 10px' }}>
                    {(ch.relevant_ranges || []).map((r, j) => (
                      <span key={j} style={{ marginRight: 4 }}><Badge text={r} color="#3b82f6" /></span>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Coding Statuses */}
      <Card title="Coding Status Definitions" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569', width: 160 }}>Status</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {defs.coding_statuses && Object.entries(defs.coding_statuses).map(([status, desc], i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px' }}>
                  <Badge text={status.replace(/_/g, ' ')} color={statusColor(status)} />
                </td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{desc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Accuracy Methodology */}
      <Card title="Accuracy Methodology">
        {defs.accuracy_methodology && (
          <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
            {typeof defs.accuracy_methodology === 'string' ? (
              <p>{defs.accuracy_methodology}</p>
            ) : (
              Object.entries(defs.accuracy_methodology).map(([key, val], i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < Object.keys(defs.accuracy_methodology).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                  <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{key.replace(/_/g, ' ')}</div>
                  <div>{typeof val === 'string' ? val : JSON.stringify(val)}</div>
                </div>
              ))
            )}
          </div>
        )}
      </Card>

      {/* Glossary */}
      <Card title="Clinical Glossary" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569', width: 160 }}>Term</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(defs.glossary || []).map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{g.term}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
