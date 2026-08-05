import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#22c55e', '#eab308', '#ef4444', '#94a3b8', '#3b82f6', '#8b5cf6', '#f97316', '#06b6d4']

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155', fontWeight: 600 }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const map = {
    ok: ['#22c55e', '✓ OK'],
    warning: ['#eab308', '⚠ Warning'],
    critical: ['#ef4444', '✖ Critical'],
    error: ['#94a3b8', '? Error'],
  }
  const [color, label] = map[status] || ['#94a3b8', status]
  return (
    <span style={{
      display: 'inline-block', padding: '3px 10px', borderRadius: 12,
      background: color + '22', color, fontSize: 12, fontWeight: 600, border: `1px solid ${color}44`
    }}>{label}</span>
  )
}

function SeverityBadge({ severity }) {
  const color = severity === 'P1' ? '#ef4444' : '#f97316'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 8,
      background: color + '22', color, fontSize: 11, fontWeight: 700
    }}>{severity}</span>
  )
}

export default function ProductionMonitoringDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastRefresh, setLastRefresh] = useState(null)

  const load = async () => {
    setLoading(true)
    try {
      const [o, b, d] = await Promise.all([
        axios.get(`${API_URL}/api/production-monitoring/overview`),
        axios.get(`${API_URL}/api/production-monitoring/breakdown`),
        axios.get(`${API_URL}/api/production-monitoring/definitions`)
      ])
      setOverview(o.data)
      setBreakdown(b.data)
      setDefinitions(d.data)
      setLastRefresh(new Date().toLocaleTimeString())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Running live production checks…</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Live Checks' },
    { id: 'breakdown', label: 'Full Breakdown' },
    { id: 'trend', label: 'Health Trend' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
      fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9',
      color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  const summary = overview?.summary || {}
  const checksSummary = overview?.checks_summary || []
  const healthTrend = overview?.health_trend || []
  const allChecks = breakdown?.checks || []
  const overallStatus = overview?.overall_status || 'unknown'

  const overallColor = overallStatus === 'ok' ? '#22c55e'
    : overallStatus === 'warning' ? '#eab308'
    : overallStatus === 'critical' ? '#ef4444' : '#94a3b8'

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Production Issue Monitoring</h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
            Live detection for 6 enterprise Agentic-AI watchpoints — Token/Cost · MCP · Vector DB · RAG Freshness · Planner · Version
          </p>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
          <StatusBadge status={overallStatus} />
          {lastRefresh && <span style={{ fontSize: 11, color: '#94a3b8' }}>Refreshed {lastRefresh}</span>}
          <button onClick={load} style={{
            padding: '4px 12px', borderRadius: 6, border: '1px solid #e2e8f0',
            background: '#f8fafc', cursor: 'pointer', fontSize: 12, color: '#475569'
          }}>↻ Re-run</button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ===== LIVE CHECKS TAB ===== */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Checks" value={summary.total_checks || 0} color="#3b82f6" />
              <KPI label="Passing" value={summary.ok || 0} color="#22c55e" />
              <KPI label="Warning" value={summary.warning || 0} color="#eab308" />
              <KPI label="Critical" value={summary.critical || 0} color="#ef4444" />
              <KPI label="Overall" value={overallStatus?.toUpperCase()} color={overallColor} />
            </div>
          </Card>

          {/* Check status cards */}
          {checksSummary.map((c, i) => (
            <Card key={i} title={c.check}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <StatusBadge status={c.status} />
                  <SeverityBadge severity={c.severity} />
                </div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  <span style={{ fontWeight: 600 }}>Layer:</span> {c.layer}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>
                  {c.detected ? '✓ Detection active' : '○ Detection pending'}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ===== FULL BREAKDOWN TAB ===== */}
      {tab === 'breakdown' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {allChecks.map((c, i) => (
            <Card key={i} title={`${c.check} — ${c.layer} (${c.severity})`}>
              <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <div style={{ minWidth: 120 }}>
                  <StatusBadge status={c.status} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#475569', marginBottom: 6 }}>Details</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 6 }}>
                    {Object.entries(c.details || {}).map(([k, v]) => {
                      if (typeof v === 'object') return null
                      return (
                        <div key={k} style={{ fontSize: 12, background: '#f8fafc', padding: '6px 10px', borderRadius: 6 }}>
                          <span style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}: </span>
                          <span style={{ fontWeight: 600, color: '#334155', fontFamily: 'monospace', fontSize: 11 }}>
                            {String(v)}
                          </span>
                        </div>
                      )
                    })}
                  </div>

                  {/* Nested array/object details */}
                  {c.details?.results && (
                    <div style={{ marginTop: 8 }}>
                      {c.details.results.map((r, j) => (
                        <div key={j} style={{ fontSize: 11, background: '#f1f5f9', padding: '6px 10px', borderRadius: 6, marginBottom: 4 }}>
                          <strong>{r.db || r.label}</strong>: {r.status} — {r.total_docs != null ? `${r.total_docs} docs` : r.error || ''}
                          {r.collections && ` (${r.collections.join(', ')})`}
                        </div>
                      ))}
                    </div>
                  )}

                  <div style={{ marginTop: 10, fontSize: 12 }}>
                    <span style={{ fontWeight: 600, color: '#64748b' }}>Threshold: </span>
                    <span style={{ color: '#475569' }}>{c.threshold}</span>
                  </div>
                  <div style={{ marginTop: 4, fontSize: 12 }}>
                    <span style={{ fontWeight: 600, color: '#64748b' }}>Remediation: </span>
                    <span style={{ color: '#3b82f6' }}>{c.remediation}</span>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ===== HEALTH TREND TAB ===== */}
      {tab === 'trend' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="System Health Over Time (from health log)">
            {healthTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={healthTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="healthy" name="Healthy Checks" stroke="#22c55e" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="issues" name="Issue Events" stroke="#ef4444" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
                No historical trend data available yet.
              </div>
            )}
          </Card>

          {/* Check status bar chart */}
          <Card title="Current Check Status Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'Passing', count: summary.ok || 0, fill: '#22c55e' },
                { name: 'Warning', count: summary.warning || 0, fill: '#eab308' },
                { name: 'Critical', count: summary.critical || 0, fill: '#ef4444' },
                { name: 'Error', count: summary.error || 0, fill: '#94a3b8' },
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" name="Checks">
                  {[
                    { fill: '#22c55e' }, { fill: '#eab308' }, { fill: '#ef4444' }, { fill: '#94a3b8' }
                  ].map((entry, i) => (
                    <rect key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ===== DEFINITIONS TAB ===== */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card title="Status Definitions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              {(definitions.statuses || []).map((s, i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                  <StatusBadge status={s.status} />
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 8, lineHeight: 1.4 }}>{s.meaning}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Check Thresholds">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Check</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Warning</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Critical</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.thresholds || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{t.check}</td>
                    <td style={{ padding: '8px 12px', color: '#eab308' }}>{t.warn}</td>
                    <td style={{ padding: '8px 12px', color: '#ef4444' }}>{t.critical}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Metrics Reference">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 200 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 80 }}>Unit</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{m.name}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{m.description}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{m.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {definitions.clinical_relevance && (
            <Card title="Clinical & Governance Relevance">
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.7 }}>
                {definitions.clinical_relevance}
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
