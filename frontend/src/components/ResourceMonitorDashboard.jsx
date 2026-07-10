import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
]

const sevColor = (sev) => {
  if (sev === 'critical') return '#ef4444'
  if (sev === 'high') return '#f97316'
  if (sev === 'medium') return '#f59e0b'
  if (sev === 'low') return '#6366f1'
  return '#8b5cf6'
}

const statusColor = (status) => {
  if (status === 'critical') return '#ef4444'
  if (status === 'warning') return '#f59e0b'
  return '#10b981'
}

const healthColor = (score) => {
  if (score >= 80) return '#10b981'
  if (score >= 60) return '#f59e0b'
  return '#ef4444'
}

const card = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  marginBottom: 16,
}

const badge = (bg) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: bg,
})

export default function ResourceMonitorDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/resource-monitor/overview',
      processes: '/api/resource-monitor/breakdown',
      events: '/api/resource-monitor/breakdown',
      gpu: '/api/resource-monitor/breakdown',
      definitions: '/api/resource-monitor/definitions',
    }
    const url = endpoints[tab] || endpoints.overview
    axios.get(API + url)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'definitions') setDefinitions(r.data)
        else setBreakdown(r.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [tab])

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'processes', label: '⚙️ Processes' },
    { id: 'events', label: '🚨 OOM Events' },
    { id: 'gpu', label: '🖥️ GPU / Limits' },
    { id: 'definitions', label: '📖 Definitions' },
  ]

  /* ── Overview Tab ─────────────────────────────────────────────────── */
  const renderOverview = () => {
    if (!overview) return null
    const s = overview.summary
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, marginBottom: 20 }}>
          {[
            { label: 'Health Score', value: s.health_score, color: healthColor(s.health_score), suffix: '%' },
            { label: 'Memory', value: s.memory_pct, color: s.memory_pct > 80 ? '#ef4444' : '#6366f1', suffix: '%' },
            { label: 'CPU', value: s.cpu_pct, color: s.cpu_pct > 80 ? '#ef4444' : '#10b981', suffix: '%' },
            { label: 'GPU Util', value: s.gpu_util_pct, color: '#8b5cf6', suffix: '%' },
            { label: 'GPU VRAM', value: s.gpu_mem_pct, color: s.gpu_mem_pct > 80 ? '#ef4444' : '#14b8a6', suffix: '%' },
            { label: 'Disk', value: s.disk_pct, color: s.disk_pct > 80 ? '#f59e0b' : '#10b981', suffix: '%' },
            { label: 'OOM Events', value: s.total_oom_events, color: s.critical_ooms > 0 ? '#ef4444' : '#6366f1' },
            { label: 'Unresolved', value: s.unresolved_events, color: s.unresolved_events > 0 ? '#ef4444' : '#10b981' },
          ].map((m, i) => (
            <div key={i} style={{ ...card, textAlign: 'center' }}>
              <div style={{ fontSize: 12, color: '#6b7280' }}>{m.label}</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: m.color }}>{m.value}{m.suffix || ''}</div>
            </div>
          ))}
        </div>

        {/* Resource usage trend (24h) */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Resource Usage (24h)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={overview.usage_history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} interval={5} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Area type="monotone" dataKey="memory_pct" name="Memory %" stackId="0"
                    stroke="#6366f1" fill="#6366f180" />
              <Area type="monotone" dataKey="cpu_pct" name="CPU %" stackId="1"
                    stroke="#10b981" fill="#10b98140" />
              <Area type="monotone" dataKey="gpu_pct" name="GPU %" stackId="2"
                    stroke="#8b5cf6" fill="#8b5cf640" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Event type distribution */}
          <div style={card}>
            <h3 style={{ margin: '0 0 12px' }}>Event Type Distribution</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.event_type_distribution} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="type" type="category" width={130} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Severity distribution */}
          <div style={card}>
            <h3 style={{ margin: '0 0 12px' }}>Severity Distribution</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.severity_distribution} dataKey="count" nameKey="severity"
                     cx="50%" cy="50%" outerRadius={80} label={({ severity, count }) => `${severity}: ${count}`}>
                  {overview.severity_distribution.map((entry, i) => (
                    <Cell key={i} fill={sevColor(entry.severity)} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Limit status */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Resource Limit Status</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Resource</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Current</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Warning</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Critical</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {overview.limit_status.map((lim, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{lim.name}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{lim.current}{lim.unit}</td>
                  <td style={{ padding: 8, textAlign: 'right', color: '#f59e0b' }}>{lim.warning}{lim.unit}</td>
                  <td style={{ padding: 8, textAlign: 'right', color: '#ef4444' }}>{lim.critical}{lim.unit}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    <span style={badge(statusColor(lim.status))}>{lim.status.toUpperCase()}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Recent events */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Recent Resource Events</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Time</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Type</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Severity</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Resolved</th>
              </tr>
            </thead>
            <tbody>
              {overview.recent_events.map((ev, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{ev.id}</td>
                  <td style={{ padding: 8 }}>{new Date(ev.timestamp).toLocaleString()}</td>
                  <td style={{ padding: 8 }}>{ev.title}</td>
                  <td style={{ padding: 8, textAlign: 'center' }}>
                    <span style={badge(sevColor(ev.severity))}>{ev.severity}</span>
                  </td>
                  <td style={{ padding: 8, textAlign: 'center' }}>{ev.resolved ? '✅' : '⏳'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  /* ── Processes Tab ────────────────────────────────────────────────── */
  const renderProcesses = () => {
    if (!breakdown) return null
    return (
      <div>
        {/* Category resource usage */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Resource Usage by Category</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={breakdown.category_usage}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="rss_mb" name="RSS (MB)" fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="cpu_pct" name="CPU %" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Category detail table */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Process Categories</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'right', padding: 8 }}>RSS (MB)</th>
                <th style={{ textAlign: 'right', padding: 8 }}>CPU %</th>
                <th style={{ textAlign: 'right', padding: 8 }}>OOM Events</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {breakdown.category_usage.map((cat, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{cat.name}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{cat.rss_mb}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{cat.cpu_pct}%</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>
                    <span style={badge(cat.oom_events > 3 ? '#ef4444' : cat.oom_events > 0 ? '#f59e0b' : '#10b981')}>
                      {cat.oom_events}
                    </span>
                  </td>
                  <td style={{ padding: 8, fontSize: 12, color: '#6b7280' }}>{cat.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Top processes */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Top Processes by Memory</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>PID</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Name</th>
                <th style={{ textAlign: 'right', padding: 8 }}>RSS (MB)</th>
                <th style={{ textAlign: 'right', padding: 8 }}>CPU %</th>
              </tr>
            </thead>
            <tbody>
              {(breakdown.top_processes || []).map((proc, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{proc.pid}</td>
                  <td style={{ padding: 8 }}>{proc.name}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{proc.rss_mb}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{proc.cpu_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Recommendations */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Autoscaling Recommendations</h3>
          {breakdown.recommendations.map((rec, i) => (
            <div key={i} style={{ padding: 12, marginBottom: 8, borderRadius: 8,
                                   background: rec.priority === 'high' ? '#fef2f2' :
                                               rec.priority === 'medium' ? '#fffbeb' : '#f0fdf4' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={badge(rec.priority === 'high' ? '#ef4444' :
                                   rec.priority === 'medium' ? '#f59e0b' : '#10b981')}>
                  {rec.priority.toUpperCase()}
                </span>
                <strong style={{ fontSize: 13 }}>{rec.area}</strong>
              </div>
              <div style={{ fontSize: 13 }}>{rec.recommendation}</div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>Action: {rec.action}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  /* ── OOM Events Tab ──────────────────────────────────────────────── */
  const renderEvents = () => {
    if (!breakdown) return null
    return (
      <div>
        {/* Daily OOM trend */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>OOM Events (14-Day Trend)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={breakdown.daily_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="events" name="Events" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Full event log */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>OOM / Resource Exhaustion Event Log</h3>
          <div style={{ maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>ID</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Time</th>
                  <th style={{ textAlign: 'center', padding: 6 }}>Severity</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Process</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Mem (GB)</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>CPU %</th>
                  <th style={{ textAlign: 'center', padding: 6 }}>Status</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Mitigation</th>
                </tr>
              </thead>
              <tbody>
                {breakdown.oom_events.map((ev, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: 6, fontFamily: 'monospace' }}>{ev.id}</td>
                    <td style={{ padding: 6 }}>{new Date(ev.timestamp).toLocaleString()}</td>
                    <td style={{ padding: 6, textAlign: 'center' }}>
                      <span style={badge(sevColor(ev.severity))}>{ev.severity}</span>
                    </td>
                    <td style={{ padding: 6 }}>{ev.title}</td>
                    <td style={{ padding: 6 }}>{ev.process_name}</td>
                    <td style={{ padding: 6, textAlign: 'right', fontFamily: 'monospace' }}>{ev.memory_at_event_gb}</td>
                    <td style={{ padding: 6, textAlign: 'right', fontFamily: 'monospace' }}>{ev.cpu_at_event_pct}</td>
                    <td style={{ padding: 6, textAlign: 'center' }}>{ev.resolved ? '✅' : '⏳'}</td>
                    <td style={{ padding: 6, fontSize: 11, color: '#6b7280' }}>{ev.mitigation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  /* ── GPU / Limits Tab ────────────────────────────────────────────── */
  const renderGPU = () => {
    if (!breakdown) return null
    return (
      <div>
        {/* GPU devices */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>GPU Devices</h3>
          {(breakdown.gpu_detail || []).length === 0 ? (
            <div style={{ color: '#6b7280', fontSize: 13 }}>No GPU devices detected</div>
          ) : (
            breakdown.gpu_detail.map((gpu, i) => (
              <div key={i} style={{ padding: 16, background: '#f9fafb', borderRadius: 8, marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div>
                    <strong style={{ fontSize: 15 }}>{gpu.model}</strong>
                    <span style={{ marginLeft: 8, fontSize: 12, color: '#6b7280' }}>{gpu.vram_gb} GB VRAM</span>
                  </div>
                  <span style={badge(gpu.temp_c > 85 ? '#ef4444' : gpu.temp_c > 70 ? '#f59e0b' : '#10b981')}>
                    {gpu.temp_c}°C
                  </span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                  {[
                    { label: 'Compute', value: `${gpu.util_pct}%`, color: gpu.util_pct > 90 ? '#ef4444' : '#6366f1' },
                    { label: 'VRAM', value: `${gpu.mem_pct}%`, color: gpu.mem_pct > 90 ? '#ef4444' : '#8b5cf6' },
                    { label: 'Used', value: `${Math.round(gpu.mem_used_mb)} MB`, color: '#14b8a6' },
                    { label: 'Power', value: `${gpu.power_w} W`, color: '#f59e0b' },
                  ].map((m, j) => (
                    <div key={j} style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#6b7280' }}>{m.label}</div>
                      <div style={{ fontSize: 20, fontWeight: 700, color: m.color }}>{m.value}</div>
                    </div>
                  ))}
                </div>
                {/* Memory usage bar */}
                <div style={{ marginTop: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#6b7280', marginBottom: 4 }}>
                    <span>VRAM Usage</span>
                    <span>{Math.round(gpu.mem_used_mb)} / {Math.round(gpu.mem_total_mb)} MB</span>
                  </div>
                  <div style={{ background: '#e5e7eb', borderRadius: 4, height: 8 }}>
                    <div style={{
                      background: gpu.mem_pct > 90 ? '#ef4444' : gpu.mem_pct > 70 ? '#f59e0b' : '#10b981',
                      borderRadius: 4, height: 8, width: `${Math.min(gpu.mem_pct, 100)}%`,
                      transition: 'width 0.3s',
                    }} />
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Resource limits config */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Resource Limit Configuration</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Limit</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Metric</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Warning</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Critical</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(breakdown.limit_config || []).map((lim, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{lim.name}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{lim.metric}</td>
                  <td style={{ padding: 8, textAlign: 'right', color: '#f59e0b', fontWeight: 600 }}>{lim.warning}{lim.unit}</td>
                  <td style={{ padding: 8, textAlign: 'right', color: '#ef4444', fontWeight: 600 }}>{lim.critical}{lim.unit}</td>
                  <td style={{ padding: 8, fontSize: 12, color: '#6b7280' }}>{lim.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  /* ── Definitions Tab ─────────────────────────────────────────────── */
  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Resource Monitoring Concepts</h3>
          {definitions.concepts.map((c, i) => (
            <div key={i} style={{ padding: 12, marginBottom: 8, borderRadius: 8, background: '#f9fafb' }}>
              <strong style={{ color: '#6366f1' }}>{c.term}</strong>
              <div style={{ fontSize: 13, marginTop: 4, color: '#374151' }}>{c.definition}</div>
            </div>
          ))}
        </div>
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Severity Levels</h3>
          {definitions.severity_levels.map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 8 }}>
              <span style={{ ...badge(s.color), minWidth: 70, textAlign: 'center' }}>{s.level}</span>
              <span style={{ fontSize: 13, color: '#374151' }}>{s.description}</span>
            </div>
          ))}
        </div>
        <div style={card}>
          <h3 style={{ margin: '0 0 12px' }}>Monitored Process Categories</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Pattern</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {definitions.process_categories.map((cat, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{cat.name}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{cat.pattern}</td>
                  <td style={{ padding: 8, color: '#6b7280' }}>{cat.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ marginBottom: 4 }}>Resource Exhaustion Monitor</h2>
      <p style={{ color: '#6b7280', marginTop: 0, marginBottom: 16, fontSize: 14 }}>
        System memory, CPU, GPU, and OOM event tracking — resource limit monitoring and autoscaling recommendations
      </p>
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#6366f1' : '#f3f4f6',
            color: tab === t.id ? '#fff' : '#374151',
            fontWeight: tab === t.id ? 600 : 400, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>
      {loading ? <div style={{ textAlign: 'center', padding: 40, color: '#6b7280' }}>Loading...</div>
       : error ? <div style={{ textAlign: 'center', padding: 40, color: '#ef4444' }}>Error: {error}</div>
       : tab === 'overview' ? renderOverview()
       : tab === 'processes' ? renderProcesses()
       : tab === 'events' ? renderEvents()
       : tab === 'gpu' ? renderGPU()
       : renderDefinitions()}
    </div>
  )
}
