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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7']

function RoleBadge({ role }) {
  const styles = {
    operator: { bg: '#dbeafe', color: '#1d4ed8', label: 'Operator' },
    assistant: { bg: '#dcfce7', color: '#16a34a', label: 'Assistant' },
  }
  const s = styles[role] || { bg: '#f1f5f9', color: '#475569', label: role }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
      background: s.bg, color: s.color
    }}>{s.label}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'messages', label: 'Messages' },
  { id: 'daily', label: 'Daily Detail' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'definitions', label: 'Glossary' },
]

export default function ConversationLogDashboard() {
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
      axios.get(`${API_URL}/api/conversation-log/overview`),
      axios.get(`${API_URL}/api/conversation-log/breakdown`),
      axios.get(`${API_URL}/api/conversation-log/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Conversation Log data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Conversation Log Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          AI-human interaction analytics — message volume, role distribution, activity patterns, text complexity
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 1 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'messages' && breakdown && <MessagesTab data={breakdown} />}
      {tab === 'daily' && breakdown && <DailyTab data={breakdown} />}
      {tab === 'analysis' && <AnalysisTab overview={overview} breakdown={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

/* --- Overview Tab --- */
function OverviewTab({ data }) {
  const { total_messages, role_distribution, date_range, text_stats, total_characters,
          avg_messages_per_day, assistant_to_operator_ratio, daily_trend, hourly_pattern } = data

  const roleChart = (role_distribution || []).map(r => ({ name: r.role, value: r.cnt }))
  const volumeChart = (daily_trend || []).map(d => ({
    date: d.day?.slice(5),
    total: d.total,
    operator: d.operator_msgs,
    assistant: d.assistant_msgs,
  }))
  const hourlyChart = (hourly_pattern || []).map(h => ({
    hour: `${String(h.hour).padStart(2, '0')}:00`,
    count: h.cnt,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card>
        <KPI label="Total Messages" value={total_messages?.toLocaleString()} color="#3b82f6" />
      </Card>
      <Card>
        <KPI label="Active Days" value={date_range?.active_days} sub={`${date_range?.first_date} to ${date_range?.last_date}`} color="#10b981" />
      </Card>
      <Card>
        <KPI label="Avg/Day" value={avg_messages_per_day} sub="messages per active day" color="#8b5cf6" />
      </Card>
      <Card>
        <KPI label="AI:Human Ratio" value={`${assistant_to_operator_ratio}:1`} sub={`${(total_characters / 1000).toFixed(0)}K total chars`} color="#f59e0b" />
      </Card>

      {/* Daily volume trend */}
      <Card title="Daily Message Volume" span={4}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={volumeChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="assistant" stackId="a" fill="#10b981" name="Assistant" radius={[0, 0, 0, 0]} />
            <Bar dataKey="operator" stackId="a" fill="#3b82f6" name="Operator" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Role distribution pie */}
      <Card title="Role Distribution" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <PieChart>
            <Pie data={roleChart} dataKey="value" nameKey="name" cx="50%" cy="50%"
                 outerRadius={85} label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                 labelLine={{ stroke: '#94a3b8' }}>
              {roleChart.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Hourly activity */}
      <Card title="Hourly Activity Pattern (UTC)" span={2}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={hourlyChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="hour" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={45} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Text stats */}
      <Card title="Text Length Statistics by Role" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Role</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Avg Length</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Max Length</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Min Length</th>
                <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Total KB</th>
              </tr>
            </thead>
            <tbody>
              {(text_stats || []).map(s => (
                <tr key={s.role} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px' }}><RoleBadge role={s.role} /></td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{s.avg_len}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{s.max_len?.toLocaleString()}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{s.min_len}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'right' }}>{s.total_kb} KB</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* --- Messages Tab --- */
function MessagesTab({ data }) {
  const { recent_messages, operator_recent } = data
  const [filter, setFilter] = useState('all')

  const filtered = filter === 'all' ? recent_messages : (recent_messages || []).filter(m => m.role === filter)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Operator recent prompts */}
      <Card title="Recent Operator Prompts" span={1}>
        <div style={{ maxHeight: 250, overflowY: 'auto' }}>
          {(operator_recent || []).map((m, i) => (
            <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
              <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 2 }}>{m.ts_utc?.slice(0, 16)}</div>
              <div style={{ color: '#334155' }}>{m.preview}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Recent messages with filter */}
      <Card title={`Recent Messages (${filtered?.length || 0})`} span={1}>
        <div style={{ marginBottom: 12, display: 'flex', gap: 8 }}>
          {['all', 'operator', 'assistant'].map(f => (
            <button key={f} onClick={() => setFilter(f)} style={{
              padding: '4px 14px', border: '1px solid #e2e8f0', borderRadius: 6,
              fontSize: 12, cursor: 'pointer',
              background: filter === f ? '#3b82f6' : '#fff',
              color: filter === f ? '#fff' : '#475569',
            }}>{f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}</button>
          ))}
        </div>
        <div style={{ maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Role</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Preview</th>
                <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569' }}>Len</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(filtered || []).map(m => (
                <tr key={m.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>#{m.id}</td>
                  <td style={{ padding: '6px 8px' }}><RoleBadge role={m.role} /></td>
                  <td style={{ padding: '6px 8px', maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#334155' }}>
                    {m.text_preview}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#64748b' }}>{m.text_length}</td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8', whiteSpace: 'nowrap', fontSize: 11 }}>{m.ts_utc?.slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* --- Daily Tab --- */
function DailyTab({ data }) {
  const { daily_detail } = data

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Daily Breakdown (${(daily_detail || []).length} days)`}>
        <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Date</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Total</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Operator</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Assistant</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Avg Len</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569' }}>Total Chars</th>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569' }}>Volume</th>
              </tr>
            </thead>
            <tbody>
              {(daily_detail || []).map(d => {
                const maxTotal = Math.max(...(daily_detail || []).map(x => x.total || 0))
                const pct = maxTotal > 0 ? ((d.total / maxTotal) * 100) : 0
                return (
                  <tr key={d.day} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600, color: '#1e293b' }}>{d.day}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600 }}>{d.total}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#1d4ed8' }}>{d.operator_msgs}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#16a34a' }}>{d.assistant_msgs}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#64748b' }}>{d.avg_text_len}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#64748b' }}>{(d.total_chars / 1000).toFixed(1)}K</td>
                    <td style={{ padding: '8px 10px', width: 120 }}>
                      <div style={{ height: 14, borderRadius: 4, background: '#f1f5f9', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${pct}%`, background: '#3b82f6', borderRadius: 4 }} />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* --- Analysis Tab --- */
function AnalysisTab({ overview, breakdown }) {
  const { longest_messages, length_distribution } = breakdown || {}
  const lengthChart = (length_distribution || []).map(b => ({ name: b.bucket, count: b.cnt }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Length distribution */}
      <Card title="Message Length Distribution" span={1}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={lengthChart}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Summary stats */}
      <Card title="Interaction Summary" span={1}>
        <div style={{ padding: 12 }}>
          {overview && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div style={{ padding: 12, background: '#f0f9ff', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#1d4ed8' }}>
                  {overview.role_distribution?.find(r => r.role === 'operator')?.cnt || 0}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>Operator Messages</div>
              </div>
              <div style={{ padding: 12, background: '#f0fdf4', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#16a34a' }}>
                  {overview.role_distribution?.find(r => r.role === 'assistant')?.cnt || 0}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>Assistant Messages</div>
              </div>
              <div style={{ padding: 12, background: '#fefce8', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#ca8a04' }}>
                  {overview.text_stats?.find(s => s.role === 'operator')?.avg_len || 0}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>Avg Operator Msg Len</div>
              </div>
              <div style={{ padding: 12, background: '#faf5ff', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#7c3aed' }}>
                  {overview.text_stats?.find(s => s.role === 'assistant')?.avg_len || 0}
                </div>
                <div style={{ fontSize: 11, color: '#64748b' }}>Avg Assistant Msg Len</div>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Longest messages */}
      <Card title="Top 10 Longest Messages" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Role</th>
                <th style={{ textAlign: 'right', padding: '6px 8px', color: '#475569' }}>Length</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Preview</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#475569' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {(longest_messages || []).map(m => (
                <tr key={m.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>#{m.id}</td>
                  <td style={{ padding: '6px 8px' }}><RoleBadge role={m.role} /></td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', fontWeight: 600, color: '#ef4444' }}>{m.text_length?.toLocaleString()}</td>
                  <td style={{ padding: '6px 8px', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#334155' }}>
                    {m.text_preview}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8', fontSize: 11 }}>{m.ts_utc?.slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

/* --- Definitions Tab --- */
function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Field glossary */}
      <Card title="Field Glossary">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Field</th>
              <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Type</th>
              <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(data.fields || []).map(f => (
              <tr key={f.field} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontWeight: 600, color: '#1e293b' }}>{f.field}</td>
                <td style={{ padding: '8px 12px', color: '#8b5cf6', fontFamily: 'monospace', fontSize: 12 }}>{f.type}</td>
                <td style={{ padding: '8px 12px', color: '#475569' }}>{f.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Roles */}
      <Card title="Role Descriptions">
        {data.roles && Object.entries(data.roles).map(([role, desc]) => (
          <div key={role} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
            <RoleBadge role={role} />
            <span style={{ marginLeft: 12, fontSize: 13, color: '#475569' }}>{desc}</span>
          </div>
        ))}
      </Card>

      {/* Metrics */}
      <Card title="Metric Definitions">
        {data.metrics && Object.entries(data.metrics).map(([metric, desc]) => (
          <div key={metric} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
            <span style={{ fontFamily: 'monospace', fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{metric}</span>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{desc}</div>
          </div>
        ))}
      </Card>

      {/* Data sources */}
      <Card title="Data Sources">
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {(data.data_sources || []).map((s, i) => (
            <li key={i} style={{ fontSize: 13, color: '#475569', padding: '4px 0' }}>{s}</li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
