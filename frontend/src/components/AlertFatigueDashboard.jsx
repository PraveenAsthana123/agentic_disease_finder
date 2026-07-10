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

export default function AlertFatigueDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/alert-fatigue/overview`).then(r => setOverview(r.data)),
      axios.get(`${API}/api/alert-fatigue/breakdown`).then(r => setBreakdown(r.data)),
      axios.get(`${API}/api/alert-fatigue/definitions`).then(r => setDefs(r.data)),
    ])
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'alerts', label: 'Alert Feed' },
    { id: 'sources', label: 'Source Analytics' },
    { id: 'routing', label: 'Severity Routing' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Alert Fatigue data…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px' }}>Alert Fatigue Monitor</h2>
      <p style={{ color: '#6b7280', margin: '0 0 20px', fontSize: 14 }}>
        Alert volume analytics, deduplication, and severity routing
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#6366f1' : '#f3f4f6',
            color: tab === t.id ? '#fff' : '#374151',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'alerts' && breakdown && <AlertFeedTab data={breakdown} />}
      {tab === 'sources' && breakdown && <SourcesTab data={breakdown} />}
      {tab === 'routing' && breakdown && <RoutingTab data={breakdown} overview={overview} />}
      {tab === 'definitions' && defs && <DefinitionsTab data={defs} />}
    </div>
  )
}

/* ── Overview Tab ─────────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  return (
    <div>
      {/* KPI cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
        <div style={card}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Fatigue Score</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: healthColor(data.fatigue_score) }}>
            {data.fatigue_score}
          </div>
          <span style={badge(healthColor(data.fatigue_score))}>{data.fatigue_status}</span>
        </div>
        <div style={card}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Total Alerts</div>
          <div style={{ fontSize: 32, fontWeight: 700 }}>{data.total_alerts}</div>
          <span style={{ fontSize: 12, color: '#6b7280' }}>
            Monitoring: {data.monitoring_alerts} · IoT: {data.iot_alerts} · SOS: {data.sos_events}
          </span>
        </div>
        <div style={card}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Suppression Rate</div>
          <div style={{ fontSize: 32, fontWeight: 700, color: data.suppression_rate_pct >= 30 ? '#10b981' : '#ef4444' }}>
            {data.suppression_rate_pct}%
          </div>
          <span style={{ fontSize: 12, color: '#6b7280' }}>
            {data.duplicates_suppressed} duplicates suppressed
          </span>
        </div>
        <div style={card}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Unique Alerts</div>
          <div style={{ fontSize: 32, fontWeight: 700 }}>{data.unique_alerts}</div>
          <span style={{ fontSize: 12, color: '#6b7280' }}>after dedup</span>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 24 }}>
        {/* Volume trend */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Alert Volume Trend (14 days)</h3>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={data.volume_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Area type="monotone" dataKey="alerts" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} name="Total" />
              <Area type="monotone" dataKey="suppressed" stroke="#10b981" fill="#10b981" fillOpacity={0.1} name="Suppressed" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Severity pie */}
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Severity Distribution</h3>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={data.severity_distribution} dataKey="count" nameKey="severity"
                cx="50%" cy="50%" outerRadius={90} label={({ severity, count }) => `${severity}: ${count}`}>
                {data.severity_distribution.map((s, i) => (
                  <Cell key={i} fill={sevColor(s.severity)} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Routing summary */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Routing Summary</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f9fafb' }}>
              <th style={{ padding: 10, textAlign: 'left' }}>Severity</th>
              <th style={{ padding: 10, textAlign: 'left' }}>Channel</th>
              <th style={{ padding: 10, textAlign: 'left' }}>Response SLA</th>
              <th style={{ padding: 10, textAlign: 'right' }}>Alerts</th>
            </tr>
          </thead>
          <tbody>
            {data.routing_summary.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid #e5e7eb' }}>
                <td style={{ padding: 10 }}><span style={badge(sevColor(r.severity))}>{r.severity}</span></td>
                <td style={{ padding: 10, color: '#374151' }}>{r.channel}</td>
                <td style={{ padding: 10, color: '#6b7280' }}>{r.response_sla}</td>
                <td style={{ padding: 10, textAlign: 'right', fontWeight: 600 }}>{r.alert_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Noisiest sources */}
      {data.noisiest_sources && data.noisiest_sources.length > 0 && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Noisiest Sources</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={data.noisiest_sources} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="source" type="category" tick={{ fontSize: 11 }} width={120} />
              <Tooltip />
              <Bar dataKey="alert_count" fill="#f59e0b" name="Alerts" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

/* ── Alert Feed Tab ───────────────────────────────────────────────────────── */
function AlertFeedTab({ data }) {
  const alerts = data.alerts || []
  return (
    <div>
      <div style={card}>
        <h3 style={{ margin: '0 0 4px', fontSize: 15 }}>Live Alert Feed</h3>
        <p style={{ color: '#6b7280', fontSize: 12, margin: '0 0 16px' }}>
          {alerts.length} alerts extracted from monitoring reports
        </p>
        {alerts.length === 0 ? (
          <div style={{ padding: 24, textAlign: 'center', color: '#10b981', fontWeight: 600 }}>
            No active alerts — all monitoring systems quiet
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f9fafb' }}>
                <th style={{ padding: 10, textAlign: 'left' }}>Severity</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Source</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Message</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Routed To</th>
                <th style={{ padding: 10, textAlign: 'left' }}>Hash</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a, i) => (
                <tr key={i} style={{ borderTop: '1px solid #e5e7eb' }}>
                  <td style={{ padding: 10 }}><span style={badge(sevColor(a.severity))}>{a.severity}</span></td>
                  <td style={{ padding: 10, fontWeight: 500 }}>{a.source_name}</td>
                  <td style={{ padding: 10, color: '#374151', maxWidth: 400, wordBreak: 'break-word' }}>{a.message}</td>
                  <td style={{ padding: 10, color: '#6b7280', fontSize: 12 }}>{a.routed_to}</td>
                  <td style={{ padding: 10, fontFamily: 'monospace', fontSize: 11, color: '#9ca3af' }}>{a.content_hash}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Dedup stats */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Deduplication Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Total Generated</div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>{data.dedup_stats?.total_alerts || 0}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Unique After Dedup</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981' }}>{data.dedup_stats?.unique_alerts || 0}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Duplicates Suppressed</div>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#f59e0b' }}>{data.dedup_stats?.duplicates_suppressed || 0}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Suppression Rate</div>
            <div style={{ fontSize: 24, fontWeight: 700 }}>{data.dedup_stats?.suppression_rate_pct || 0}%</div>
          </div>
        </div>
      </div>

      {/* Dedup strategies */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Dedup Strategies</h3>
        {(data.dedup_strategies || []).map((s, i) => (
          <div key={i} style={{ padding: '10px 0', borderTop: i > 0 ? '1px solid #f3f4f6' : 'none' }}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>{s.name}</div>
            <div style={{ color: '#6b7280', fontSize: 13 }}>{s.description}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Sources Tab ──────────────────────────────────────────────────────────── */
function SourcesTab({ data }) {
  const sources = data.source_health || []
  const iot = data.iot_breakdown || {}
  return (
    <div>
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Source Health</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f9fafb' }}>
              <th style={{ padding: 10, textAlign: 'left' }}>Source</th>
              <th style={{ padding: 10, textAlign: 'left' }}>Description</th>
              <th style={{ padding: 10, textAlign: 'right' }}>Alerts</th>
              <th style={{ padding: 10, textAlign: 'right' }}>Critical</th>
              <th style={{ padding: 10, textAlign: 'right' }}>High</th>
              <th style={{ padding: 10, textAlign: 'right' }}>Health</th>
            </tr>
          </thead>
          <tbody>
            {sources.map((s, i) => (
              <tr key={i} style={{ borderTop: '1px solid #e5e7eb' }}>
                <td style={{ padding: 10, fontWeight: 600 }}>{s.source_name}</td>
                <td style={{ padding: 10, color: '#6b7280', fontSize: 12 }}>{s.description}</td>
                <td style={{ padding: 10, textAlign: 'right' }}>{s.total_alerts}</td>
                <td style={{ padding: 10, textAlign: 'right', color: s.critical > 0 ? '#ef4444' : '#6b7280' }}>{s.critical}</td>
                <td style={{ padding: 10, textAlign: 'right', color: s.high > 0 ? '#f97316' : '#6b7280' }}>{s.high}</td>
                <td style={{ padding: 10, textAlign: 'right' }}>
                  <span style={badge(healthColor(s.health_score))}>{s.health_score}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Source alert chart */}
      {sources.length > 0 && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Alerts by Source</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={sources}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="source_name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="total_alerts" fill="#6366f1" name="Total" radius={[4, 4, 0, 0]} />
              <Bar dataKey="critical" fill="#ef4444" name="Critical" radius={[4, 4, 0, 0]} />
              <Bar dataKey="high" fill="#f97316" name="High" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* IoT / SOS breakdown */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>IoT / SOS Alert Volume</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 16 }}>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>IoT Alerts</div>
            <div style={{ fontSize: 28, fontWeight: 700 }}>{iot.iot_alerts || 0}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>SOS Events</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{iot.sos_events || 0}</div>
          </div>
        </div>
        {(iot.recent_iot || []).length > 0 && (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f9fafb' }}>
                <th style={{ padding: 8, textAlign: 'left' }}>Alert Type</th>
                <th style={{ padding: 8, textAlign: 'left' }}>Severity</th>
                <th style={{ padding: 8, textAlign: 'right' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {iot.recent_iot.map((r, i) => (
                <tr key={i} style={{ borderTop: '1px solid #e5e7eb' }}>
                  <td style={{ padding: 8 }}>{r.alert_type}</td>
                  <td style={{ padding: 8 }}><span style={badge(sevColor(r.severity))}>{r.severity}</span></td>
                  <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{r.cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Volume trend */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Alert Volume Trend</h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.volume_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="alerts" stroke="#6366f1" strokeWidth={2} dot={false} name="Alerts" />
            <Line type="monotone" dataKey="suppressed" stroke="#10b981" strokeWidth={2} dot={false} name="Suppressed" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

/* ── Routing Tab ──────────────────────────────────────────────────────────── */
function RoutingTab({ data, overview: ov }) {
  const rules = data.routing_rules || []
  return (
    <div>
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Severity Routing Rules</h3>
        {rules.map((r, i) => (
          <div key={i} style={{
            padding: 16, marginBottom: 12, borderRadius: 8,
            border: `2px solid ${sevColor(r.severity)}20`,
            background: `${sevColor(r.severity)}08`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span style={badge(sevColor(r.severity))}>{r.severity.toUpperCase()}</span>
              <span style={{ fontWeight: 600 }}>{r.channel}</span>
              <span style={{ color: '#6b7280', fontSize: 12, marginLeft: 'auto' }}>
                SLA: {r.response_sla}
              </span>
            </div>
            <div style={{ fontSize: 13, color: '#374151', marginBottom: 6 }}>
              <strong>Escalation:</strong> {r.escalation}
            </div>
            <div style={{ fontSize: 12, color: '#6b7280' }}>
              <strong>Examples:</strong> {r.examples.join(' · ')}
            </div>
          </div>
        ))}
      </div>

      {/* Routing distribution chart */}
      {ov && ov.routing_summary && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Current Routing Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={ov.routing_summary}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="alert_count" name="Alerts Routed" radius={[4, 4, 0, 0]}>
                {ov.routing_summary.map((r, i) => (
                  <Cell key={i} fill={sevColor(r.severity)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

/* ── Definitions Tab ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  return (
    <div>
      <div style={card}>
        <h3 style={{ margin: '0 0 16px', fontSize: 15 }}>{data.title}</h3>
        {(data.terms || []).map((t, i) => (
          <div key={i} style={{ padding: '10px 0', borderTop: i > 0 ? '1px solid #f3f4f6' : 'none' }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 2 }}>{t.term}</div>
            <div style={{ color: '#6b7280', fontSize: 13, lineHeight: 1.5 }}>{t.definition}</div>
          </div>
        ))}
      </div>

      {/* Dedup strategies reference */}
      {data.dedup_strategies && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Deduplication Strategies</h3>
          {data.dedup_strategies.map((s, i) => (
            <div key={i} style={{ padding: '10px 0', borderTop: i > 0 ? '1px solid #f3f4f6' : 'none' }}>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{s.name}</div>
              <div style={{ color: '#6b7280', fontSize: 13 }}>{s.description}</div>
            </div>
          ))}
        </div>
      )}

      {/* Alert sources reference */}
      {data.alert_sources && (
        <div style={card}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15 }}>Monitored Alert Sources</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f9fafb' }}>
                <th style={{ padding: 8, textAlign: 'left' }}>Source</th>
                <th style={{ padding: 8, textAlign: 'left' }}>Description</th>
                <th style={{ padding: 8, textAlign: 'left' }}>Report</th>
              </tr>
            </thead>
            <tbody>
              {data.alert_sources.map((s, i) => (
                <tr key={i} style={{ borderTop: '1px solid #e5e7eb' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{s.name}</td>
                  <td style={{ padding: 8, color: '#6b7280' }}>{s.description}</td>
                  <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 11 }}>{s.report}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
