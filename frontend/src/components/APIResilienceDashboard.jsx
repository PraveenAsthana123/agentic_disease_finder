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

const cbColor = (state) => {
  if (state === 'closed') return '#10b981'
  if (state === 'half_open') return '#f59e0b'
  return '#ef4444'
}

const cbLabel = (state) => {
  if (state === 'closed') return 'Closed'
  if (state === 'half_open') return 'Half-Open'
  return 'Open'
}

const healthColor = (score) => {
  if (score >= 90) return '#10b981'
  if (score >= 70) return '#f59e0b'
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

export default function APIResilienceDashboard() {
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
      overview: '/api/api-resilience/overview',
      services: '/api/api-resilience/breakdown',
      hourly: '/api/api-resilience/breakdown',
      incidents: '/api/api-resilience/breakdown',
      definitions: '/api/api-resilience/definitions',
    }
    const url = endpoints[tab]
    if (!url) return
    axios.get(`${API}${url}`)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'services' || tab === 'hourly' || tab === 'incidents') setBreakdown(r.data)
        else setDefinitions(r.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'services', label: 'Services' },
    { id: 'hourly', label: 'Traffic Pattern' },
    { id: 'incidents', label: 'Incidents' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading API Resilience...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px' }}>API Resilience Monitor</h2>
      <p style={{ color: '#64748b', margin: '0 0 20px' }}>Circuit breaker, rate limiting & retry observability</p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px',
              borderRadius: 8,
              border: tab === t.id ? '2px solid #6366f1' : '1px solid #e2e8f0',
              background: tab === t.id ? '#eef2ff' : '#fff',
              fontWeight: tab === t.id ? 600 : 400,
              cursor: 'pointer',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && overview && renderOverview(overview)}
      {tab === 'services' && breakdown && renderServices(breakdown)}
      {tab === 'hourly' && breakdown && renderHourly(breakdown)}
      {tab === 'incidents' && breakdown && renderIncidents(breakdown)}
      {tab === 'definitions' && definitions && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(data) {
  const agg = data.aggregate
  const cbData = [
    { name: 'Closed', value: agg.circuit_breakers_closed },
    { name: 'Half-Open', value: agg.circuit_breakers_half_open },
    { name: 'Open', value: agg.circuit_breakers_open },
  ].filter(d => d.value > 0)
  const cbColors = ['#10b981', '#f59e0b', '#ef4444']

  return (
    <div>
      {/* Health Score */}
      <div style={{ ...card, textAlign: 'center', borderLeft: `4px solid ${healthColor(data.health_score)}` }}>
        <div style={{ fontSize: 48, fontWeight: 700, color: healthColor(data.health_score) }}>
          {data.health_score}
        </div>
        <div style={{ color: '#64748b', fontSize: 14 }}>Health Score (0-100)</div>
      </div>

      {/* Aggregate metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
        {[
          { label: 'API Calls (24h)', value: agg.total_api_calls_24h.toLocaleString() },
          { label: 'Success Rate', value: `${agg.overall_success_rate_pct}%` },
          { label: 'Failures', value: agg.total_failures_24h.toLocaleString() },
          { label: 'Timeouts', value: agg.total_timeouts_24h.toLocaleString() },
          { label: 'Retries', value: agg.total_retries_24h.toLocaleString() },
          { label: 'Rate Limited', value: agg.total_rate_limited_24h.toLocaleString() },
        ].map((m, i) => (
          <div key={i} style={card}>
            <div style={{ fontSize: 22, fontWeight: 600 }}>{m.value}</div>
            <div style={{ color: '#64748b', fontSize: 13 }}>{m.label}</div>
          </div>
        ))}
      </div>

      {/* Circuit Breaker Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Circuit Breaker States</h4>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={cbData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                {cbData.map((_, i) => <Cell key={i} fill={cbColors[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Service Health</h4>
          <div style={{ maxHeight: 200, overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 4 }}>Service</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>Success</th>
                  <th style={{ textAlign: 'right', padding: 4 }}>p95</th>
                  <th style={{ textAlign: 'center', padding: 4 }}>CB</th>
                </tr>
              </thead>
              <tbody>
                {(data.services || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 4 }}>{s.service}</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{s.success_rate_pct}%</td>
                    <td style={{ textAlign: 'right', padding: 4 }}>{s.p95_ms}ms</td>
                    <td style={{ textAlign: 'center', padding: 4 }}>
                      <span style={badge(cbColor(s.cb_state))}>{cbLabel(s.cb_state)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

function renderServices(data) {
  const services = data.services || []
  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>Per-Service Resilience Details</h3>
      {services.map((svc, i) => (
        <div key={i} style={{ ...card, borderLeft: `4px solid ${cbColor(svc.circuit_breaker.state)}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <strong>{svc.service_name}</strong>
              <span style={{ marginLeft: 8, color: '#64748b', fontSize: 12 }}>[{svc.service_type}]</span>
            </div>
            <span style={badge(cbColor(svc.circuit_breaker.state))}>
              {cbLabel(svc.circuit_breaker.state)}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, fontSize: 13 }}>
            <div><span style={{ color: '#64748b' }}>Calls 24h:</span> {svc.total_calls_24h.toLocaleString()}</div>
            <div><span style={{ color: '#64748b' }}>Failures:</span> {svc.failures} ({svc.failure_rate_pct}%)</div>
            <div><span style={{ color: '#64748b' }}>Timeouts:</span> {svc.timeouts}</div>
            <div><span style={{ color: '#64748b' }}>p50:</span> {svc.latency_p50_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>p95:</span> {svc.latency_p95_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>p99:</span> {svc.latency_p99_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>Retries:</span> {svc.retries_24h} ({svc.retry_success_rate_pct}% success)</div>
            <div><span style={{ color: '#64748b' }}>Rate Limit:</span> {svc.rate_limit.current_rpm}/{svc.rate_limit.max_rpm} RPM ({svc.rate_limit.utilization_pct}%)</div>
            <div><span style={{ color: '#64748b' }}>CB Trips:</span> {svc.circuit_breaker.trips_24h}</div>
            {svc.backoff.active && (
              <div><span style={{ color: '#ef4444' }}>Backoff:</span> x{svc.backoff.current_multiplier}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function renderHourly(data) {
  const hourly = data.hourly_pattern || []
  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>24-Hour Traffic Pattern</h3>
      <div style={card}>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={hourly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" label={{ value: 'Hour', position: 'bottom' }} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="calls" stroke="#6366f1" fill="#eef2ff" name="Calls" />
            <Area type="monotone" dataKey="failures" stroke="#ef4444" fill="#fef2f2" name="Failures" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Retry Configuration</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 13 }}>
          <div><strong>Strategy:</strong> {data.retry_config?.strategy}</div>
          <div><strong>Max Retries:</strong> {data.retry_config?.max_retries}</div>
          <div><strong>Base Delay:</strong> {data.retry_config?.base_delay_ms}ms</div>
          <div><strong>Max Delay:</strong> {data.retry_config?.max_delay_ms}ms</div>
          <div><strong>Jitter:</strong> {data.retry_config?.jitter ? 'Yes' : 'No'}</div>
          <div><strong>Retryable Codes:</strong> {(data.retry_config?.retryable_codes || []).join(', ')}</div>
        </div>
      </div>
    </div>
  )
}

function renderIncidents(data) {
  const incidents = data.recent_incidents || []
  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>Recent Incidents</h3>
      {incidents.length === 0 ? (
        <div style={card}><p style={{ color: '#64748b' }}>No recent incidents recorded.</p></div>
      ) : (
        incidents.map((inc, i) => (
          <div key={i} style={{ ...card, borderLeft: `4px solid ${inc.severity === 'critical' ? '#ef4444' : '#f59e0b'}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong>{inc.service} - {inc.type.replace(/_/g, ' ')}</strong>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={badge(inc.severity === 'critical' ? '#ef4444' : '#f59e0b')}>
                  {inc.severity}
                </span>
                <span style={badge(inc.resolved ? '#10b981' : '#64748b')}>
                  {inc.resolved ? 'Resolved' : 'Active'}
                </span>
              </div>
            </div>
            <div style={{ marginTop: 8, fontSize: 13, color: '#64748b' }}>
              Occurred: {inc.occurred_at} | Retries triggered: {inc.retries_triggered}
              {inc.cb_tripped && ' | Circuit breaker tripped'}
            </div>
          </div>
        ))
      )}
    </div>
  )
}

function renderDefinitions(data) {
  const concepts = data.concepts || []
  const practices = data.best_practices || []
  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>API Resilience Patterns</h3>
      {concepts.map((c, i) => (
        <div key={i} style={card}>
          <h4 style={{ margin: '0 0 8px', color: '#6366f1' }}>{c.term}</h4>
          <p style={{ margin: '0 0 8px', fontSize: 14 }}>{c.definition}</p>
          {c.states && (
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse', marginTop: 8 }}>
              <tbody>
                {c.states.map((s, j) => (
                  <tr key={j} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 4, fontWeight: 600 }}>{s.state}</td>
                    <td style={{ padding: 4 }}>{s.meaning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {c.formula && <div style={{ marginTop: 8, fontSize: 12, fontFamily: 'monospace', background: '#f8fafc', padding: 8, borderRadius: 4 }}>{c.formula}</div>}
        </div>
      ))}

      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Best Practices</h4>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
          {practices.map((p, i) => <li key={i} style={{ marginBottom: 4 }}>{p}</li>)}
        </ul>
      </div>
    </div>
  )
}
