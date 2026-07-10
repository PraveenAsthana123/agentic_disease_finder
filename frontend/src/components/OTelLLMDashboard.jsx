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

const healthColor = (score) => {
  if (score >= 90) return '#10b981'
  if (score >= 70) return '#f59e0b'
  return '#ef4444'
}

const severityColor = (sev) => {
  if (sev === 'critical') return '#ef4444'
  if (sev === 'warning') return '#f59e0b'
  return '#6366f1'
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

export default function OTelLLMDashboard() {
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
      overview: '/api/otel-llm/overview',
      models: '/api/otel-llm/breakdown',
      traces: '/api/otel-llm/breakdown',
      alerts: '/api/otel-llm/breakdown',
      definitions: '/api/otel-llm/definitions',
    }
    const url = endpoints[tab]
    if (!url) return
    axios.get(`${API}${url}`)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'models' || tab === 'traces' || tab === 'alerts') setBreakdown(r.data)
        else setDefinitions(r.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'models', label: 'Models' },
    { id: 'traces', label: 'Trace Flow' },
    { id: 'alerts', label: 'Alerts' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading OTel LLM Observability...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px' }}>OpenTelemetry LLM Observability</h2>
      <p style={{ color: '#64748b', margin: '0 0 20px' }}>OTel + OpenLIT trace monitoring for local LLM inference</p>

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
      {tab === 'models' && breakdown && renderModels(breakdown)}
      {tab === 'traces' && breakdown && renderTraceFlow(breakdown)}
      {tab === 'alerts' && breakdown && renderAlerts(breakdown)}
      {tab === 'definitions' && definitions && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(data) {
  const agg = data.aggregate
  const modelUsage = (data.model_usage || []).map((m, i) => ({ ...m, fill: COLORS[i % COLORS.length] }))
  const spanDist = (data.span_distribution || []).map((s, i) => ({ ...s, name: s.type.replace(/_/g, ' '), fill: COLORS[i % COLORS.length] }))

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
          { label: 'Total Traces (24h)', value: agg.total_traces_24h.toLocaleString() },
          { label: 'Avg Latency', value: `${agg.avg_latency_ms}ms` },
          { label: 'Token Throughput', value: `${agg.token_throughput_tok_per_sec} tok/s` },
          { label: 'Error Rate', value: `${agg.error_rate_pct}%` },
          { label: 'Est. Cost (24h)', value: `$${agg.estimated_cost_24h_usd}` },
          { label: 'Active Models', value: agg.active_models },
          { label: 'Total Spans (24h)', value: agg.total_spans_24h.toLocaleString() },
          { label: 'Completion Rate', value: `${agg.trace_completion_rate_pct}%` },
        ].map((m, i) => (
          <div key={i} style={card}>
            <div style={{ fontSize: 22, fontWeight: 600 }}>{m.value}</div>
            <div style={{ color: '#64748b', fontSize: 13 }}>{m.label}</div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Model Usage Distribution</h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={modelUsage} dataKey="traces" nameKey="model" cx="50%" cy="50%" outerRadius={80} label={({ model, pct }) => `${model} (${pct}%)`}>
                {modelUsage.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div style={card}>
          <h4 style={{ margin: '0 0 12px' }}>Span Type Distribution</h4>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={spanDist} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, pct }) => `${name} (${pct}%)`}>
                {spanDist.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model summary table */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Model Summary</h4>
        <div style={{ overflow: 'auto' }}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6 }}>Model</th>
                <th style={{ textAlign: 'left', padding: 6 }}>Type</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Traces</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Avg Latency</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Error Rate</th>
                <th style={{ textAlign: 'right', padding: 6 }}>Throughput</th>
              </tr>
            </thead>
            <tbody>
              {(data.models || []).map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6 }}>{m.model}</td>
                  <td style={{ padding: 6 }}><span style={badge(m.type === 'embedding' ? '#14b8a6' : m.type === 'code' ? '#8b5cf6' : '#6366f1')}>{m.type}</span></td>
                  <td style={{ textAlign: 'right', padding: 6 }}>{m.traces_24h.toLocaleString()}</td>
                  <td style={{ textAlign: 'right', padding: 6 }}>{m.avg_latency_ms}ms</td>
                  <td style={{ textAlign: 'right', padding: 6 }}>{m.error_rate_pct}%</td>
                  <td style={{ textAlign: 'right', padding: 6 }}>{m.throughput > 0 ? `${m.throughput} tok/s` : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function renderModels(data) {
  const models = data.models || []

  // Prepare comparison chart data
  const comparisonData = models.filter(m => m.model_type !== 'embedding').map(m => ({
    name: m.model_name,
    p50: m.latency.p50_ms,
    p95: m.latency.p95_ms,
    p99: m.latency.p99_ms,
  }))

  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>Per-Model LLM Observability</h3>

      {/* Latency comparison chart */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Latency Comparison (Chat/Code Models)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Bar dataKey="p50" fill="#6366f1" name="P50" />
            <Bar dataKey="p95" fill="#f59e0b" name="P95" />
            <Bar dataKey="p99" fill="#ef4444" name="P99" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-model detail cards */}
      {models.map((m, i) => (
        <div key={i} style={{ ...card, borderLeft: `4px solid ${COLORS[i % COLORS.length]}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <strong>{m.model_name}</strong>
              <span style={{ marginLeft: 8 }}>
                <span style={badge(m.model_type === 'embedding' ? '#14b8a6' : m.model_type === 'code' ? '#8b5cf6' : '#6366f1')}>
                  {m.model_type}
                </span>
              </span>
              <span style={{ marginLeft: 8, color: '#64748b', fontSize: 12 }}>[{m.provider}]</span>
            </div>
            <span style={badge(m.error_rate_pct > 3 ? '#ef4444' : m.error_rate_pct > 1 ? '#f59e0b' : '#10b981')}>
              {m.error_rate_pct}% errors
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8, fontSize: 13 }}>
            <div><span style={{ color: '#64748b' }}>Traces 24h:</span> {m.total_traces_24h.toLocaleString()}</div>
            <div><span style={{ color: '#64748b' }}>Spans 24h:</span> {m.total_spans_24h.toLocaleString()}</div>
            <div><span style={{ color: '#64748b' }}>Completion:</span> {m.completion_rate_pct}%</div>
            <div><span style={{ color: '#64748b' }}>p50:</span> {m.latency.p50_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>p95:</span> {m.latency.p95_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>p99:</span> {m.latency.p99_ms}ms</div>
            <div><span style={{ color: '#64748b' }}>Input Tokens:</span> {m.tokens.total_input_24h.toLocaleString()}</div>
            <div><span style={{ color: '#64748b' }}>Output Tokens:</span> {m.tokens.total_output_24h.toLocaleString()}</div>
            <div><span style={{ color: '#64748b' }}>Throughput:</span> {m.tokens.throughput_tok_per_sec > 0 ? `${m.tokens.throughput_tok_per_sec} tok/s` : 'N/A'}</div>
            <div><span style={{ color: '#64748b' }}>GPU Seconds:</span> {m.cost.gpu_seconds_24h}s</div>
            <div><span style={{ color: '#64748b' }}>Est. Cost:</span> ${m.cost.estimated_24h_usd}</div>
            <div><span style={{ color: '#64748b' }}>Ctx Window:</span> {m.ctx_window.toLocaleString()}</div>
          </div>

          {/* Error breakdown */}
          {m.failed_traces > 0 && (
            <div style={{ marginTop: 10, fontSize: 12, color: '#64748b' }}>
              Errors: timeout={m.errors.timeout}, context_overflow={m.errors.context_overflow}, model_error={m.errors.model_error}, other={m.errors.other}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function renderTraceFlow(data) {
  const hourly = data.hourly_volume || []
  const spanBreakdown = data.span_breakdown || []

  // Prepare span breakdown as stacked bar chart data
  const spanBarData = spanBreakdown.map(sb => ({
    model: sb.model,
    llm_call: sb.spans.llm_call || 0,
    embedding: sb.spans.embedding || 0,
    retrieval: sb.spans.retrieval || 0,
    tool_use: sb.spans.tool_use || 0,
    agent_step: sb.spans.agent_step || 0,
  }))

  // Trace completion stats
  const models = data.models || []
  const totalTraces = models.reduce((s, m) => s + m.total_traces_24h, 0)
  const completedTraces = models.reduce((s, m) => s + m.successful_traces, 0)
  const failedTraces = models.reduce((s, m) => s + m.failed_traces, 0)

  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>Trace Flow Analysis</h3>

      {/* Trace completion stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
        {[
          { label: 'Total Traces', value: totalTraces.toLocaleString() },
          { label: 'Completed', value: completedTraces.toLocaleString() },
          { label: 'Failed', value: failedTraces.toLocaleString() },
          { label: 'Completion Rate', value: `${totalTraces > 0 ? ((completedTraces / totalTraces) * 100).toFixed(1) : 0}%` },
        ].map((m, i) => (
          <div key={i} style={card}>
            <div style={{ fontSize: 22, fontWeight: 600 }}>{m.value}</div>
            <div style={{ color: '#64748b', fontSize: 13 }}>{m.label}</div>
          </div>
        ))}
      </div>

      {/* Hourly trace volume */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>24-Hour Trace Volume</h4>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={hourly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" label={{ value: 'Hour', position: 'bottom' }} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="traces" stroke="#6366f1" fill="#eef2ff" name="Traces" />
            <Area type="monotone" dataKey="errors" stroke="#ef4444" fill="#fef2f2" name="Errors" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Latency trend */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Hourly Avg Latency</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={hourly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis label={{ value: 'ms', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Line type="monotone" dataKey="avg_latency_ms" stroke="#f59e0b" name="Avg Latency (ms)" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Span type breakdown per model */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>Span Type Breakdown by Model</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={spanBarData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="llm_call" stackId="a" fill="#6366f1" name="LLM Call" />
            <Bar dataKey="embedding" stackId="a" fill="#14b8a6" name="Embedding" />
            <Bar dataKey="retrieval" stackId="a" fill="#f59e0b" name="Retrieval" />
            <Bar dataKey="tool_use" stackId="a" fill="#8b5cf6" name="Tool Use" />
            <Bar dataKey="agent_step" stackId="a" fill="#ec4899" name="Agent Step" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* OTel Config */}
      <div style={card}>
        <h4 style={{ margin: '0 0 12px' }}>OTel Exporter Configuration</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, fontSize: 13 }}>
          <div><strong>Exporter:</strong> {data.otel_config?.exporter}</div>
          <div><strong>Endpoint:</strong> {data.otel_config?.endpoint}</div>
          <div><strong>Service:</strong> {data.otel_config?.service_name}</div>
          <div><strong>Sampling Rate:</strong> {data.otel_config?.sampling_rate}</div>
          <div><strong>Batch Size:</strong> {data.otel_config?.batch_size}</div>
          <div><strong>Export Interval:</strong> {data.otel_config?.export_interval_ms}ms</div>
          <div><strong>Conventions:</strong> {data.otel_config?.semantic_conventions}</div>
          <div><strong>OpenLIT:</strong> {data.otel_config?.openlit_enabled ? 'Enabled' : 'Disabled'}</div>
        </div>
      </div>
    </div>
  )
}

function renderAlerts(data) {
  const alerts = data.alerts || []
  return (
    <div>
      <h3 style={{ margin: '0 0 16px' }}>Anomaly Alerts</h3>
      {alerts.length === 0 ? (
        <div style={card}><p style={{ color: '#64748b' }}>No recent alerts recorded.</p></div>
      ) : (
        alerts.map((alert, i) => (
          <div key={i} style={{ ...card, borderLeft: `4px solid ${severityColor(alert.severity)}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong>{alert.type.replace(/_/g, ' ')} — {alert.model}</strong>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={badge(severityColor(alert.severity))}>
                  {alert.severity}
                </span>
                <span style={badge(alert.resolved ? '#10b981' : '#64748b')}>
                  {alert.resolved ? 'Resolved' : 'Active'}
                </span>
              </div>
            </div>
            <div style={{ marginTop: 8, fontSize: 14 }}>{alert.message}</div>
            <div style={{ marginTop: 4, fontSize: 12, color: '#64748b' }}>
              Occurred: {alert.occurred_at} | Trace: <code style={{ fontSize: 11 }}>{alert.trace_id}</code>
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
      <h3 style={{ margin: '0 0 16px' }}>OpenTelemetry LLM Concepts</h3>
      {concepts.map((c, i) => (
        <div key={i} style={card}>
          <h4 style={{ margin: '0 0 8px', color: '#6366f1' }}>{c.term}</h4>
          <p style={{ margin: '0 0 8px', fontSize: 14 }}>{c.definition}</p>
          {c.components && (
            <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 13 }}>
              {c.components.map((comp, j) => <li key={j} style={{ marginBottom: 2 }}>{comp}</li>)}
            </ul>
          )}
          {c.attributes && (
            <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 13 }}>
              {c.attributes.map((attr, j) => <li key={j} style={{ marginBottom: 2 }}>{attr}</li>)}
            </ul>
          )}
          {c.key_attributes && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#64748b', marginBottom: 4 }}>Key Attributes ({c.namespace}):</div>
              <ul style={{ margin: '0', paddingLeft: 20, fontSize: 13 }}>
                {c.key_attributes.map((attr, j) => <li key={j} style={{ marginBottom: 2 }}>{attr}</li>)}
              </ul>
            </div>
          )}
          {c.features && (
            <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 13 }}>
              {c.features.map((f, j) => <li key={j} style={{ marginBottom: 2 }}>{f}</li>)}
            </ul>
          )}
          {c.metrics && (
            <ul style={{ margin: '4px 0', paddingLeft: 20, fontSize: 13 }}>
              {c.metrics.map((m, j) => <li key={j} style={{ marginBottom: 2 }}>{m}</li>)}
            </ul>
          )}
          {c.thresholds && typeof c.thresholds === 'object' && !Array.isArray(c.thresholds) && (
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 4, fontSize: 12 }}>
              {Object.entries(c.thresholds).map(([k, v], j) => (
                <div key={j}><span style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}:</span> {v}</div>
              ))}
            </div>
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
