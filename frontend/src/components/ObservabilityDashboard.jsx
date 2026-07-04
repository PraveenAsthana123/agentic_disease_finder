import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const LEVEL_COLORS = { DEBUG: '#94a3b8', INFO: '#3b82f6', WARN: '#f59e0b', ERROR: '#ef4444', FATAL: '#dc2626' }
const STATUS_COLORS = { healthy: '#10b981', warning: '#f59e0b', critical: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 16, fontWeight: 600 }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color }) {
  return (
    <div style={{ background: '#f8fafc', borderRadius: 8, padding: '14px 18px', textAlign: 'center', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{label}</div>
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 12, fontWeight: 600,
      background: (color || '#64748b') + '22', color: color || '#64748b'
    }}>{text}</span>
  )
}

export default function ObservabilityDashboard() {
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
          axios.get(`${API_URL}/api/observability/overview`),
          axios.get(`${API_URL}/api/observability/breakdown`),
          axios.get(`${API_URL}/api/observability/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load observability data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Observability data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  const k = overview?.kpis || {}
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Components' },
    { id: 'logs', label: 'Logs & Traces' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'definitions', label: 'Definitions' }
  ]

  // Log level distribution for pie chart
  const logLevelData = Object.entries(overview?.log_level_distribution || {}).map(([name, value]) => ({ name, value }))

  // Daily volume for area chart
  const dailyVolume = overview?.daily_volume || []

  // Component health
  const componentHealth = overview?.component_health || []

  // Active alerts
  const activeAlerts = overview?.active_alerts || []

  // Breakdown data
  const bd = breakdown || {}
  const recentLogs = bd.recent_logs || []
  const sampleTraces = bd.sample_traces || []
  const actionDist = Object.entries(bd.action_distribution || {}).map(([name, value]) => ({ name, value }))
  const actorDist = Object.entries(bd.actor_distribution || {}).map(([name, value]) => ({ name, value }))
  const latPerc = bd.latency_percentiles || {}

  const renderOverview = () => (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 18 }}>
        <KPI label="Total Events" value={k.total_events} color="#3b82f6" />
        <KPI label="Error Count" value={k.error_count} color="#ef4444" />
        <KPI label="Error Rate %" value={k.error_rate_pct != null ? `${k.error_rate_pct}%` : '--'} color={k.error_rate_pct > 5 ? '#ef4444' : k.error_rate_pct > 1 ? '#f59e0b' : '#10b981'} />
        <KPI label="Warnings" value={k.warn_count} color="#f59e0b" />
        <KPI label="P50 Latency" value={k.p50_latency_ms != null ? `${k.p50_latency_ms}ms` : '--'} />
        <KPI label="P95 Latency" value={k.p95_latency_ms != null ? `${k.p95_latency_ms}ms` : '--'} color={k.p95_latency_ms > 3000 ? '#ef4444' : '#f59e0b'} />
        <KPI label="P99 Latency" value={k.p99_latency_ms != null ? `${k.p99_latency_ms}ms` : '--'} />
        <KPI label="Mean Latency" value={k.mean_latency_ms != null ? `${k.mean_latency_ms}ms` : '--'} />
        <KPI label="Components" value={k.active_components} color="#8b5cf6" />
        <KPI label="Active Alerts" value={k.active_alerts} color={k.active_alerts > 0 ? '#ef4444' : '#10b981'} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        <Card title="Log Level Distribution">
          {logLevelData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={logLevelData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75}
                  label={({ name, value }) => `${name}: ${value}`}>
                  {logLevelData.map((entry, i) => (
                    <Cell key={i} fill={LEVEL_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        <Card title="Daily Event Volume">
          {dailyVolume.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={dailyVolume}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="total" stroke="#3b82f6" fill="#3b82f680" name="Total" />
                <Area type="monotone" dataKey="errors" stroke="#ef4444" fill="#ef444440" name="Errors" />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        <Card title="Top Actions">
          {actionDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={actionDist.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Count" />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        <Card title="Actors">
          {actorDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={actorDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75}
                  label={({ name, value }) => `${name}: ${value}`}>
                  {actorDist.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>
      </div>
    </>
  )

  const renderComponents = () => (
    <Card title={`Component Health (${componentHealth.length} components)`}>
      {componentHealth.length > 0 ? (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Component</th>
              <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Status</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Events</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Errors</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Error %</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>P50 (ms)</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>P95 (ms)</th>
              <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>P99 (ms)</th>
            </tr>
          </thead>
          <tbody>
            {componentHealth.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 6px', fontWeight: 600 }}>{c.component}</td>
                <td style={{ textAlign: 'center', padding: '8px 6px' }}>
                  <Badge text={c.status} color={STATUS_COLORS[c.status] || '#64748b'} />
                </td>
                <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(c.total_events)}</td>
                <td style={{ textAlign: 'right', padding: '8px 6px', color: c.error_count > 0 ? '#ef4444' : '#64748b' }}>{c.error_count}</td>
                <td style={{ textAlign: 'right', padding: '8px 6px', color: c.error_rate_pct > 5 ? '#ef4444' : c.error_rate_pct > 1 ? '#f59e0b' : '#64748b' }}>{c.error_rate_pct}%</td>
                <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(c.p50_ms)}</td>
                <td style={{ textAlign: 'right', padding: '8px 6px', color: c.p95_ms > 3000 ? '#ef4444' : '#64748b' }}>{fmt(c.p95_ms)}</td>
                <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(c.p99_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No component data</div>}
    </Card>
  )

  const renderLogs = () => (
    <>
      <Card title="Latency Percentiles">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 12 }}>
          <KPI label="P50" value={`${latPerc.p50 || 0}ms`} />
          <KPI label="P95" value={`${latPerc.p95 || 0}ms`} color={latPerc.p95 > 3000 ? '#ef4444' : '#f59e0b'} />
          <KPI label="P99" value={`${latPerc.p99 || 0}ms`} color={latPerc.p99 > 8000 ? '#ef4444' : '#f59e0b'} />
          <KPI label="Mean" value={`${latPerc.mean || 0}ms`} />
        </div>
      </Card>

      <Card title={`Sample Traces (${sampleTraces.length})`}>
        {sampleTraces.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Trace ID</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Patient</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Spans</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Duration (ms)</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Has Error</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {sampleTraces.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 11 }}>{t.trace_id.slice(0, 18)}...</td>
                  <td style={{ padding: '8px 6px' }}>{t.patient_id}</td>
                  <td style={{ textAlign: 'right', padding: '8px 6px' }}>{t.span_count}</td>
                  <td style={{ textAlign: 'right', padding: '8px 6px', fontWeight: 600 }}>{fmt(t.total_duration_ms)}</td>
                  <td style={{ textAlign: 'center', padding: '8px 6px' }}>
                    {t.has_error ? <Badge text="ERROR" color="#ef4444" /> : <Badge text="OK" color="#10b981" />}
                  </td>
                  <td style={{ padding: '8px 6px', fontSize: 11, color: '#64748b' }}>{t.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No traces</div>}
      </Card>

      <Card title={`Recent Logs (last ${recentLogs.length})`}>
        {recentLogs.length > 0 ? (
          <div style={{ maxHeight: 400, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Time</th>
                  <th style={{ textAlign: 'center', padding: '6px', color: '#64748b' }}>Level</th>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Component</th>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Actor</th>
                  <th style={{ textAlign: 'right', padding: '6px', color: '#64748b' }}>Latency</th>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Patient</th>
                </tr>
              </thead>
              <tbody>
                {recentLogs.map((l, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: l.level === 'ERROR' ? '#fef2f2' : l.level === 'WARN' ? '#fffbeb' : 'transparent' }}>
                    <td style={{ padding: '6px', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{l.timestamp.slice(11, 19) || l.timestamp.slice(0, 16)}</td>
                    <td style={{ textAlign: 'center', padding: '6px' }}>
                      <Badge text={l.level} color={LEVEL_COLORS[l.level] || '#64748b'} />
                    </td>
                    <td style={{ padding: '6px', fontWeight: 500 }}>{l.component}</td>
                    <td style={{ padding: '6px' }}>{l.action}</td>
                    <td style={{ padding: '6px', color: '#64748b' }}>{l.actor}</td>
                    <td style={{ textAlign: 'right', padding: '6px', color: l.latency_ms > 3000 ? '#ef4444' : '#64748b' }}>{l.latency_ms}ms</td>
                    <td style={{ padding: '6px', fontSize: 11 }}>{l.patient_id || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No logs</div>}
      </Card>
    </>
  )

  const renderAlerts = () => (
    <>
      <Card title={`Active Alerts (${activeAlerts.length})`}>
        {activeAlerts.length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Rule</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Severity</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Value</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Component</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {activeAlerts.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: a.severity === 'critical' ? '#fef2f2' : '#fffbeb' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{a.rule}</td>
                  <td style={{ textAlign: 'center', padding: '8px 6px' }}>
                    <Badge text={a.severity} color={a.severity === 'critical' ? '#ef4444' : '#f59e0b'} />
                  </td>
                  <td style={{ padding: '8px 6px' }}>{a.value}</td>
                  <td style={{ padding: '8px 6px' }}>{a.component || '--'}</td>
                  <td style={{ textAlign: 'center', padding: '8px 6px' }}>
                    <Badge text={a.status || 'firing'} color="#ef4444" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ padding: 20, textAlign: 'center', color: '#10b981', fontSize: 14 }}>
            No active alerts — all systems nominal
          </div>
        )}
      </Card>

      <Card title="Alert Rules">
        {(defs?.alert_rules || []).length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Name</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Condition</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Severity</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {defs.alert_rules.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{r.name}</td>
                  <td style={{ padding: '8px 6px', fontFamily: 'monospace', fontSize: 11 }}>{r.condition}</td>
                  <td style={{ textAlign: 'center', padding: '8px 6px' }}>
                    <Badge text={r.severity} color={r.severity === 'critical' ? '#ef4444' : '#f59e0b'} />
                  </td>
                  <td style={{ padding: '8px 6px' }}>{r.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No rules defined</div>}
      </Card>

      <Card title="Metric Thresholds">
        {defs?.metric_thresholds ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Metric</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#f59e0b' }}>Warning</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#ef4444' }}>Critical</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(defs.metric_thresholds).map(([metric, vals], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{metric}</td>
                  <td style={{ textAlign: 'right', padding: '8px 6px', color: '#f59e0b' }}>{vals.warning}</td>
                  <td style={{ textAlign: 'right', padding: '8px 6px', color: '#ef4444' }}>{vals.critical}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No thresholds</div>}
      </Card>
    </>
  )

  const renderDefinitions = () => (
    <>
      <Card title="Log Levels">
        {defs?.log_levels ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 100 }}>Level</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(defs.log_levels).map(([level, desc], i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px' }}><Badge text={level} color={LEVEL_COLORS[level]} /></td>
                  <td style={{ padding: '8px 6px', color: '#475569' }}>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Trace Span Types">
        {(defs?.trace_span_types || []).length > 0 ? (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 180 }}>Span</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {defs.trace_span_types.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>{s.name}</td>
                  <td style={{ padding: '8px 6px', color: '#475569' }}>{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Instrumentation">
        {defs?.instrumentation ? (
          <div style={{ fontSize: 13, color: '#475569' }}>
            <p><strong>Standard:</strong> {defs.instrumentation.standard}</p>
            <p><strong>Trace Propagation:</strong> {defs.instrumentation.trace_propagation}</p>
            <p><strong>Canonical Fields:</strong> {(defs.instrumentation.canonical_fields || []).join(', ')}</p>
            <div style={{ marginTop: 8 }}>
              <strong>Backends:</strong>
              <ul style={{ margin: '4px 0 0 16px', padding: 0 }}>
                {Object.entries(defs.instrumentation.backends || {}).map(([k, v], i) => (
                  <li key={i}><strong>{k}:</strong> {v}</li>
                ))}
              </ul>
            </div>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Clinical Relevance">
        <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>
          {defs?.clinical_relevance || 'No data'}
        </p>
      </Card>

      <Card title="Data Source">
        {defs?.data_source ? (
          <div style={{ fontSize: 13, color: '#475569' }}>
            <p><strong>Table:</strong> {defs.data_source.table}</p>
            <p><strong>Database:</strong> {defs.data_source.database}</p>
            <p>{defs.data_source.description}</p>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>
    </>
  )

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ margin: '0 0 6px', fontSize: 22, color: '#1e293b' }}>Observability Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Logs, traces, metrics, component health, and alert monitoring from transaction_log
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'components' && renderComponents()}
      {tab === 'logs' && renderLogs()}
      {tab === 'alerts' && renderAlerts()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
