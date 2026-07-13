import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const PRIORITY_COLORS = {
  urgent: '#ef4444',
  high: '#f59e0b',
  normal: '#3b82f6',
  low: '#64748b'
}

const READ_COLORS = {
  read: '#10b981',
  unread: '#ef4444'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function PriorityBadge({ priority }) {
  const color = PRIORITY_COLORS[priority] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{(priority || '').replace(/_/g, ' ')}</span>
  )
}

function ReadStatusBadge({ status }) {
  const color = READ_COLORS[status] || '#64748b'
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

export default function SecureMessagingDashboard() {
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
          axios.get(`${API_URL}/secure-messaging/overview`),
          axios.get(`${API_URL}/secure-messaging/breakdown`),
          axios.get(`${API_URL}/secure-messaging/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load secure messaging data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading secure messaging data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Secure messaging data not available</div>

  const tabs = ['overview', 'categories', 'patients', 'unread']
  const kpis = overview.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Secure Messaging Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Patient-provider communication analytics — {fmt(kpis.total_messages)} messages across {fmt(kpis.total_patients)} patients
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
              <KPI label="Total Messages" value={kpis.total_messages} />
              <KPI label="Inbound" value={kpis.inbound_count} sub="patient to provider" />
              <KPI label="Outbound" value={kpis.outbound_count} sub="provider to patient" />
              <KPI label="Unread" value={kpis.unread_count} sub={`${fmt(kpis.unread_pct)}%`} color="#ef4444" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Total Patients" value={kpis.total_patients} />
              <KPI label="Avg Response" value={kpis.avg_response_hours} sub="hours" color="#f59e0b" />
              <KPI label="Median Response" value={kpis.median_response_hours} sub="hours" color="#3b82f6" />
              <KPI label="Msgs/Patient" value={kpis.total_messages && kpis.total_patients ? (kpis.total_messages / kpis.total_patients).toFixed(1) : '--'} />
            </div>
          </Card>

          <Card title="Direction Split" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={overview.direction_split || []} dataKey="count" nameKey="direction" cx="50%" cy="50%" outerRadius={90} label={({ direction, pct }) => `${direction} ${pct}%`}>
                  {(overview.direction_split || []).map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Priority Breakdown" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.priority_breakdown || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="priority" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" name="Messages">
                  {(overview.priority_breakdown || []).map((entry, i) => (
                    <Cell key={i} fill={PRIORITY_COLORS[entry.priority] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Category Distribution" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={overview.category_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="category" type="category" tick={{ fontSize: 11 }} width={140} />
                <Tooltip />
                <Bar dataKey="count" name="Messages" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Message Volume" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={overview.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="messages" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Avg Response Time by Priority" span={4}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.response_by_priority || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="priority" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} label={{ value: 'Hours', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }} />
                <Tooltip formatter={(v) => `${v} hrs`} />
                <Bar dataKey="avg_hours" name="Avg Response (hrs)" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(overview.response_by_priority || []).map((entry, i) => (
                    <Cell key={i} fill={PRIORITY_COLORS[entry.priority] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            {defs?.response_time_note && (
              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>{defs.response_time_note}</p>
            )}
          </Card>
        </div>
      )}

      {/* ── Categories tab ── */}
      {tab === 'categories' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Category Detail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Total</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Inbound</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Unread</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Avg Response (hrs)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Unique Patients</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.category_detail || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{(row.category || '').replace(/-/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.total)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.inbound)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.unread > 0 ? '#ef4444' : '#64748b' }}>{fmt(row.unread)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.avg_response_hours)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.unique_patients)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Category Descriptions">
            {defs?.categories && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                {Object.entries(defs.categories).map(([key, desc]) => (
                  <div key={key} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>
                      {key.replace(/-/g, ' ')}
                    </div>
                    <div style={{ fontSize: 12, color: '#64748b' }}>{desc}</div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Messages</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Inbound</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Outbound</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Unread</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Avg Response (hrs)</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.messages)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.inbound)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.outbound)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.unread > 0 ? '#ef4444' : '#64748b' }}>{fmt(row.unread)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.avg_response_hours)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Messages">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Direction</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Subject</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Priority</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_messages || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.direction}</td>
                      <td style={{ padding: '8px 12px' }}>{(row.category || '').replace(/-/g, ' ')}</td>
                      <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.subject}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><PriorityBadge priority={row.priority} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><ReadStatusBadge status={row.read_status} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.created_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Unread tab ── */}
      {tab === 'unread' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Unread Message Queue (${(breakdown.unread_queue || []).length})`}>
            {(breakdown.unread_queue || []).length === 0 ? (
              <div style={{ padding: 24, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>
                All messages read - inbox clear
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Direction</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Category</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Subject</th>
                      <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Priority</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.unread_queue || []).map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: row.priority === 'urgent' ? '#fef2f2' : row.priority === 'high' ? '#fffbeb' : undefined }}>
                        <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                        <td style={{ padding: '8px 12px' }}>{row.direction}</td>
                        <td style={{ padding: '8px 12px' }}>{(row.category || '').replace(/-/g, ' ')}</td>
                        <td style={{ padding: '8px 12px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{row.subject}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}><PriorityBadge priority={row.priority} /></td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{(row.created_at || '').slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {defs?.unread_note && (
              <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 8 }}>{defs.unread_note}</p>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
