import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  const colorMap = {
    ok: '#22c55e', healthy: '#22c55e', complete: '#22c55e', registered: '#22c55e',
    low: '#22c55e', safe: '#22c55e',
    warning: '#eab308', partial: '#eab308', medium: '#eab308',
    critical: '#ef4444', failed: '#ef4444', high: '#ef4444', unregistered: '#ef4444',
    pending: '#94a3b8', unknown: '#94a3b8'
  }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function ShadowAIDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/api/shadow-ai/overview`),
          axios.get(`${API_URL}/api/shadow-ai/breakdown`),
          axios.get(`${API_URL}/api/shadow-ai/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Shadow AI data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'timeline', label: 'Detection Timeline' },
    { id: 'sources', label: 'Source Analysis' },
    { id: 'events', label: 'Recent Events' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  const riskLevel = overview?.risk_level || 'unknown'
  const riskColor = riskLevel === 'low' ? '#22c55e' : riskLevel === 'medium' ? '#eab308' : riskLevel === 'high' ? '#ef4444' : '#94a3b8'
  const timeline = overview?.detection_timeline || []
  const riskDist = overview?.risk_distribution || []
  const topSources = overview?.top_shadow_sources || []
  const temporalPattern = breakdown?.temporal_pattern || []
  const sourceAnalysis = breakdown?.source_analysis || []
  const recentEvents = breakdown?.recent_shadow_events || []
  const levelDist = breakdown?.level_distribution || []
  const heatmap = breakdown?.hourly_heatmap || {}
  const defs = definitions?.metrics || []

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Shadow AI Detection Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Detect unauthorized AI tool usage across the platform — {fmt(overview?.total_events_scanned)} events scanned
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Events Scanned" value={fmt(overview?.total_events_scanned)} />
              <KPI label="Registered Tools" value={fmt(overview?.registered_tools_count)} color="#3b82f6" />
              <KPI label="Shadow Detections" value={fmt(overview?.shadow_detections)} color={overview?.shadow_detections > 0 ? '#f97316' : '#22c55e'} />
              <KPI label="Shadow Rate" value={fmt(overview?.shadow_rate) + '%'} color={riskColor} />
              <KPI label="Risk Level" value={<StatusBadge status={riskLevel} />} />
            </div>
          </Card>

          <Card title="Last 24h / 7d">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <KPI label="Detections (24h)" value={fmt(overview?.detections_last_24h)} color="#3b82f6" />
              <KPI label="Detections (7d)" value={fmt(overview?.detections_last_7d)} color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Risk Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={riskDist} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={80} label={({ category, count }) => `${category}: ${count}`}>
                  {riskDist.map((_, i) => <Cell key={i} fill={['#22c55e','#eab308','#ef4444'][i] || COLORS[i]} />)}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Top Shadow Sources">
            {topSources.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={topSources} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="source" width={180} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f97316" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No shadow sources detected</div>
            )}
          </Card>

          <Card title="Temporal Pattern (6h Blocks)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={temporalPattern}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="block" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'timeline' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Detection Timeline (14 Days)">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={timeline}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#ef4444" fill="#ef444422" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Hourly Heatmap (Last 7 Days)">
            {heatmap.matrix && heatmap.matrix.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
                  <thead>
                    <tr>
                      <th style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b' }}>Day</th>
                      {(heatmap.hour_labels || []).map(h => (
                        <th key={h} style={{ padding: '4px 4px', textAlign: 'center', color: '#64748b', minWidth: 28 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(heatmap.day_labels || []).map((day, di) => (
                      <tr key={di}>
                        <td style={{ padding: '4px 8px', fontWeight: 600, whiteSpace: 'nowrap' }}>{day}</td>
                        {(heatmap.matrix[di] || []).map((val, hi) => {
                          const maxVal = Math.max(...heatmap.matrix.flat(), 1)
                          const intensity = val / maxVal
                          const bg = val === 0 ? '#f8fafc' : `rgba(239, 68, 68, ${0.15 + intensity * 0.7})`
                          const textColor = intensity > 0.5 ? '#fff' : '#334155'
                          return (
                            <td key={hi} style={{ padding: '4px 4px', textAlign: 'center', background: bg, color: textColor, borderRadius: 2 }}>
                              {val > 0 ? val : ''}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No heatmap data available</div>
            )}
          </Card>

          <Card title="Level Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={levelDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'sources' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Source Event Distribution" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sourceAnalysis}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="total_events" fill="#3b82f6" name="Total Events" radius={[4, 4, 0, 0]} />
                <Bar dataKey="shadow_events" fill="#ef4444" name="Shadow Events" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Source Details" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Source</th>
                    <th style={{ padding: '8px 12px', textAlign: 'center' }}>Registered</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Total Events</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Shadow Events</th>
                    <th style={{ padding: '8px 12px', textAlign: 'right' }}>Shadow Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {sourceAnalysis.map((s, i) => {
                    const rate = s.total_events > 0 ? ((s.shadow_events / s.total_events) * 100).toFixed(1) : '0.0'
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.source}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                          <StatusBadge status={s.registered ? 'registered' : 'unregistered'} />
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(s.total_events)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: s.shadow_events > 0 ? '#ef4444' : '#22c55e' }}>
                          {fmt(s.shadow_events)}
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{rate}%</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'events' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Recent Shadow Events (${recentEvents.length})`}>
            <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Timestamp</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Level</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Shadow Source</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>Event</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left' }}>User</th>
                  </tr>
                </thead>
                <tbody>
                  {recentEvents.map((ev, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', whiteSpace: 'nowrap', color: '#64748b' }}>{ev.ts}</td>
                      <td style={{ padding: '8px 10px' }}><StatusBadge status={ev.level} /></td>
                      <td style={{ padding: '8px 10px', fontWeight: 600, color: '#f97316' }}>{ev.shadow_source}</td>
                      <td style={{ padding: '8px 10px', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {ev.event}
                      </td>
                      <td style={{ padding: '8px 10px' }}>{ev.user || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Shadow AI Metric Definitions">
            <div style={{ display: 'grid', gap: 12 }}>
              {defs.map((d, i) => (
                <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
