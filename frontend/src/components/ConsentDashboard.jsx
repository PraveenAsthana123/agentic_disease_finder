import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const STATUS_COLORS = {
  granted: '#10b981',
  pending: '#f59e0b',
  declined: '#ef4444',
  expired: '#64748b',
  withdrawn: '#8b5cf6'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(status || '').replace(/_/g, ' ')}</span>
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

export default function ConsentDashboard() {
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
          axios.get(`${API_URL}/api/consent-dashboard/overview`),
          axios.get(`${API_URL}/api/consent-dashboard/breakdown`),
          axios.get(`${API_URL}/api/consent-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load consent data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading consent data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Consent data not available</div>

  const tabs = ['overview', 'patients', 'activity', 'regulatory']
  const kpis = overview.kpis || []

  const statusData = Object.entries(overview.by_status || {}).map(([k, v]) => ({ name: k, value: v }))
  const typeData = Object.entries(overview.by_type || {}).map(([k, v]) => ({ name: k.replace(/_/g, ' '), value: v }))

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Consent Management Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Informed consent tracking — {fmt(overview.total_consents)} consent records across {fmt(overview.total_patients)} patients, {fmt(overview.consent_rate)}% granted
      </p>

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
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(kpis.length, 4)}, 1fr)`, gap: 16 }}>
              {kpis.slice(0, 4).map((k, i) => (
                <KPI key={i} label={k.label} value={k.value} sub={k.sub}
                  color={k.label === 'Declined' ? '#ef4444' : k.label === 'Withdrawn' ? '#8b5cf6' : undefined} />
              ))}
            </div>
            {kpis.length > 4 && (
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(kpis.length - 4, 4)}, 1fr)`, gap: 16, marginTop: 16 }}>
                {kpis.slice(4).map((k, i) => (
                  <KPI key={i} label={k.label} value={k.value} sub={k.sub}
                    color={k.label === 'Declined' ? '#ef4444' : k.label === 'Withdrawn' ? '#8b5cf6' : undefined} />
                ))}
              </div>
            )}
          </Card>

          <Card title="Consent Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90}
                  label={({ name, value }) => `${name} (${value})`}>
                  {statusData.map((entry, i) => (
                    <Cell key={i} fill={STATUS_COLORS[entry.name] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Consents by Type" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={typeData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="value" name="Consents" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Recent Activity" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Consent Type</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Granted</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Expiry</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Witness</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.recent_activity || []).slice(0, 10).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{(row.consent_type || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.granted_date || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.expiry_date || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.witness || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Consent Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Name</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Granted</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Pending</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Declined</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Expired</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Withdrawn</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Full Consent</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.name || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.total_consents)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#10b981' }}>{fmt(row.granted)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.pending > 0 ? '#f59e0b' : '#64748b' }}>{fmt(row.pending)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.declined > 0 ? '#ef4444' : '#64748b' }}>{fmt(row.declined)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.expired)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.withdrawn)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                          background: row.full_consent ? '#10b98122' : '#ef444422',
                          color: row.full_consent ? '#10b981' : '#ef4444',
                          fontWeight: 600, fontSize: 12
                        }}>{row.full_consent ? 'YES' : 'NO'}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Activity tab ── */}
      {tab === 'activity' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="All Consent Records">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>ID</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Granted</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Expiry</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Witness</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.recent_activity || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{row.id}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{(row.consent_type || '').replace(/_/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StatusBadge status={row.status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.granted_date || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.expiry_date || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.witness || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.notes || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Regulatory tab ── */}
      {tab === 'regulatory' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Consent Type Definitions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(defs.consent_types || {}).map(([key, desc]) => (
                <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                    {key.replace(/_/g, ' ')}
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Status Definitions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(defs.statuses || {}).map(([key, desc]) => (
                <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <StatusBadge status={key} />
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Regulatory Framework">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 12 }}>
              {Object.entries(defs.regulatory || {}).map(([key, reg]) => (
                <div key={key} style={{ padding: 14, background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{reg.title}</div>
                  <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2 }}>{reg.citation}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 6, lineHeight: 1.5 }}>{reg.summary}</div>
                </div>
              ))}
            </div>
          </Card>

          {defs.glossary && defs.glossary.length > 0 && (
            <Card title="Glossary">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {defs.glossary.map((item, i) => (
                  <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{item.definition}</div>
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
