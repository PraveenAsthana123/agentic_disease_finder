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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4']
const STATUS_COLORS = { healthy: '#10b981', degraded: '#f59e0b', down: '#ef4444' }
const STATUS_ICONS = { healthy: '\u2705', degraded: '\u26a0\ufe0f', down: '\u274c' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Components & Incidents' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SystemHealthDashboard() {
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
      axios.get(`${API_URL}/api/system-health/overview`),
      axios.get(`${API_URL}/api/system-health/breakdown`),
      axios.get(`${API_URL}/api/system-health/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading System Health data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>System Health Monitoring Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Infrastructure health monitoring — component status, resource utilization, response times, incident tracking
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
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} overview={overview} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const { kpis, status_distribution, component_summary, resource_distribution, timeline } = data

  const statusData = Object.entries(status_distribution).map(([k, v]) => ({ name: k, value: v }))
  const componentChart = (component_summary || []).map(c => ({
    name: c.component,
    uptime: c.uptime_pct,
    avg_response: c.avg_response_ms,
    errors: c.error_count,
  }))

  const resourceData = ['cpu', 'memory', 'disk'].map(r => ({
    name: r.charAt(0).toUpperCase() + r.slice(1),
    Low: resource_distribution[r]?.low || 0,
    Moderate: resource_distribution[r]?.moderate || 0,
    High: resource_distribution[r]?.high || 0,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card>
        <KPI label="Overall Uptime" value={`${kpis.overall_uptime_pct}%`}
             color={kpis.overall_uptime_pct >= 90 ? '#10b981' : '#f59e0b'}
             sub={`${kpis.total_checks} checks`} />
      </Card>
      <Card>
        <KPI label="Avg Response Time" value={`${kpis.avg_response_ms}ms`}
             color={kpis.avg_response_ms < 500 ? '#10b981' : '#f59e0b'} />
      </Card>
      <Card>
        <KPI label="Total Errors" value={kpis.total_errors}
             color={kpis.total_errors === 0 ? '#10b981' : '#ef4444'}
             sub={`across ${kpis.components_monitored} components`} />
      </Card>
      <Card>
        <KPI label="Avg CPU / Mem / Disk"
             value={`${kpis.avg_cpu_pct}%`}
             sub={`Mem ${kpis.avg_memory_pct}% | Disk ${kpis.avg_disk_pct}%`} />
      </Card>

      {/* Status distribution pie */}
      <Card title="Health Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                 outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {statusData.map((e, i) => (
                <Cell key={i} fill={STATUS_COLORS[e.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Component uptime bar chart */}
      <Card title="Component Uptime (%)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={componentChart} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis type="category" dataKey="name" width={90} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Bar dataKey="uptime" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Component status matrix */}
      <Card title="Component Status Matrix" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Component', 'Current', 'Uptime', 'Avg RT', 'CPU', 'Mem', 'Disk', 'Errors'].map(h => (
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(component_summary || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{c.component}</td>
                  <td style={{ padding: '8px 6px' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 8px', borderRadius: 9999,
                      fontSize: 11, fontWeight: 600,
                      background: STATUS_COLORS[c.current_status] + '20',
                      color: STATUS_COLORS[c.current_status],
                    }}>{STATUS_ICONS[c.current_status]} {c.current_status}</span>
                  </td>
                  <td style={{ padding: '8px 6px', color: c.uptime_pct >= 90 ? '#10b981' : '#f59e0b' }}>{c.uptime_pct}%</td>
                  <td style={{ padding: '8px 6px' }}>{c.avg_response_ms}ms</td>
                  <td style={{ padding: '8px 6px' }}>{c.avg_cpu}%</td>
                  <td style={{ padding: '8px 6px' }}>{c.avg_mem}%</td>
                  <td style={{ padding: '8px 6px' }}>{c.avg_disk}%</td>
                  <td style={{ padding: '8px 6px', color: c.error_count > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                    {c.error_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Resource utilization stacked bar */}
      <Card title="Resource Utilization Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={resourceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Low" stackId="a" fill="#10b981" />
            <Bar dataKey="Moderate" stackId="a" fill="#f59e0b" />
            <Bar dataKey="High" stackId="a" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Response time timeline */}
      <Card title="Response Time Timeline" span={4}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={(timeline || []).map(t => ({
            ...t,
            date: t.timestamp?.slice(5, 10),
          }))}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(v) => `${v}ms`}
                     labelFormatter={(l) => `Date: ${l}`} />
            <Line type="monotone" dataKey="response_time_ms" stroke="#3b82f6"
                  strokeWidth={2} dot={{ r: 3 }} name="Response Time" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data, overview }) {
  const { all_checks, response_percentiles, error_events, incidents } = data

  const percData = Object.entries(response_percentiles || {}).map(([comp, p]) => ({
    name: comp, p50: p.p50, p90: p.p90, p99: p.p99,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Response time percentiles */}
      <Card title="Response Time Percentiles by Component" span={4}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={percData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Bar dataKey="p50" fill="#3b82f6" name="P50 (median)" />
            <Bar dataKey="p90" fill="#f59e0b" name="P90" />
            <Bar dataKey="p99" fill="#ef4444" name="P99" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Incidents */}
      <Card title={`Incidents (${incidents?.length || 0} degraded/down events)`} span={2}>
        {(!incidents || incidents.length === 0) ? (
          <div style={{ color: '#10b981', textAlign: 'center', padding: 20 }}>No incidents recorded</div>
        ) : (
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Time', 'Component', 'Status', 'RT (ms)', 'Errors'].map(h => (
                    <th key={h} style={{ padding: '6px 4px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {incidents.map((inc, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 4px' }}>{inc.timestamp?.slice(0, 16)}</td>
                    <td style={{ padding: '6px 4px', fontWeight: 600 }}>{inc.component}</td>
                    <td style={{ padding: '6px 4px' }}>
                      <span style={{
                        padding: '2px 6px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                        background: STATUS_COLORS[inc.status] + '20',
                        color: STATUS_COLORS[inc.status],
                      }}>{inc.status}</span>
                    </td>
                    <td style={{ padding: '6px 4px' }}>{inc.response_time_ms}</td>
                    <td style={{ padding: '6px 4px', color: (inc.error_count || 0) > 0 ? '#ef4444' : '#64748b' }}>
                      {inc.error_count || 0}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* Error events */}
      <Card title={`Error Events (${error_events?.length || 0})`} span={2}>
        {(!error_events || error_events.length === 0) ? (
          <div style={{ color: '#10b981', textAlign: 'center', padding: 20 }}>No errors recorded</div>
        ) : (
          <div style={{ maxHeight: 300, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Time', 'Component', 'Status', 'Error Count'].map(h => (
                    <th key={h} style={{ padding: '6px 4px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {error_events.map((ev, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 4px' }}>{ev.timestamp?.slice(0, 16)}</td>
                    <td style={{ padding: '6px 4px', fontWeight: 600 }}>{ev.component}</td>
                    <td style={{ padding: '6px 4px' }}>
                      <span style={{
                        padding: '2px 6px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                        background: STATUS_COLORS[ev.status] + '20',
                        color: STATUS_COLORS[ev.status],
                      }}>{ev.status}</span>
                    </td>
                    <td style={{ padding: '6px 4px', color: '#ef4444', fontWeight: 700 }}>{ev.error_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {/* All health checks log */}
      <Card title="Full Health Check Log" span={4}>
        <div style={{ maxHeight: 400, overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                {['ID', 'Timestamp', 'Component', 'Status', 'RT (ms)', 'CPU%', 'Mem%', 'Disk%', 'Errors'].map(h => (
                  <th key={h} style={{ padding: '6px 4px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(all_checks || []).map((chk, i) => (
                <tr key={i} style={{
                  borderBottom: '1px solid #f1f5f9',
                  background: chk.status !== 'healthy' ? '#fef2f2' : undefined,
                }}>
                  <td style={{ padding: '6px 4px', color: '#94a3b8' }}>{chk.check_id}</td>
                  <td style={{ padding: '6px 4px' }}>{chk.timestamp?.slice(0, 16)}</td>
                  <td style={{ padding: '6px 4px', fontWeight: 600 }}>{chk.component}</td>
                  <td style={{ padding: '6px 4px' }}>
                    <span style={{
                      padding: '2px 6px', borderRadius: 9999, fontSize: 11, fontWeight: 600,
                      background: STATUS_COLORS[chk.status] + '20',
                      color: STATUS_COLORS[chk.status],
                    }}>{chk.status}</span>
                  </td>
                  <td style={{ padding: '6px 4px' }}>{chk.response_time_ms}</td>
                  <td style={{ padding: '6px 4px', color: chk.cpu_pct > 70 ? '#ef4444' : undefined }}>{chk.cpu_pct}%</td>
                  <td style={{ padding: '6px 4px', color: chk.memory_pct > 75 ? '#ef4444' : undefined }}>{chk.memory_pct}%</td>
                  <td style={{ padding: '6px 4px', color: chk.disk_pct > 80 ? '#ef4444' : undefined }}>{chk.disk_pct}%</td>
                  <td style={{ padding: '6px 4px', color: (chk.error_count || 0) > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                    {chk.error_count || 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="System Health Monitoring Glossary">
        <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b', fontWeight: 600, width: 180 }}>Term</th>
              <th style={{ padding: '8px 6px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(data.glossary || []).map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600, color: '#334155' }}>{g.term}</td>
                <td style={{ padding: '8px 6px', color: '#475569' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
