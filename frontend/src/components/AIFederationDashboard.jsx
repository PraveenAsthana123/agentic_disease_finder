import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'

const STATUS_COLORS = {
  active: '#22c55e',
  inactive: '#ef4444',
  onboarding: '#f59e0b',
  completed: '#22c55e',
  failed: '#ef4444',
  in_progress: '#3b82f6',
}

const PIE_COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4', '#ec4899', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function Card({ title, children, span, accent }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined,
      borderLeft: accent ? `4px solid ${accent}` : undefined,
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '22', color,
      textTransform: 'capitalize'
    }}>{status ? status.replace('_', ' ') : 'Unknown'}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'site_detail', label: 'Site Detail' },
  { id: 'round_history', label: 'Round History' },
  { id: 'drift_monitor', label: 'Drift Monitor' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AIFederationDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedDef, setExpandedDef] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/ai-federation/overview`),
      axios.get(`${API_URL}/api/ai-federation/breakdown`),
      axios.get(`${API_URL}/api/ai-federation/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefs(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI Federation data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const kpis = ov.kpis || {}
  const sitesByStatus = ov.sites_by_status || []
  const sitesByRegion = ov.sites_by_region || []
  const accuracyTrend = ov.accuracy_trend || []
  const aggregationMethods = ov.aggregation_methods || []
  const topDrifting = ov.top_drifting_sites || []
  const sites = bd.sites || []
  const rounds = bd.rounds || []
  const perSiteAccuracy = bd.per_site_accuracy || []

  const thStyle = {
    textAlign: 'left', padding: '8px 10px', fontSize: 12,
    color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600
  }
  const tdStyle = {
    padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9'
  }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        AI Federation Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Federated learning across hospital sites: global accuracy tracking, drift monitoring, aggregation rounds, and multi-site data quality
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13,
            fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b',
            background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>

          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Sites" value={fmt(kpis.total_sites)} sub="Hospital sites in federation" />
              <KPI label="Active Sites" value={fmt(kpis.active_sites)} color="#22c55e" sub="Currently participating" />
              <KPI label="Total Rounds" value={fmt(kpis.total_rounds)} sub="Aggregation rounds run" />
              <KPI label="Completed Rounds" value={fmt(kpis.completed_rounds)} color="#22c55e" sub="Successfully aggregated" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Global Accuracy" value={`${fmt(kpis.global_accuracy)}%`}
                color={kpis.global_accuracy >= 85 ? '#22c55e' : kpis.global_accuracy >= 75 ? '#f59e0b' : '#ef4444'}
                sub="Latest round" />
              <KPI label="Avg Site Accuracy" value={`${fmt(kpis.avg_site_accuracy)}%`}
                color={kpis.avg_site_accuracy >= 80 ? '#22c55e' : '#f59e0b'}
                sub="Across all sites" />
              <KPI label="Avg Data Quality" value={fmt(kpis.avg_data_quality)}
                color={kpis.avg_data_quality >= 0.8 ? '#22c55e' : '#f59e0b'}
                sub="Score 0-1" />
              <KPI label="Avg Drift Score" value={fmt(kpis.avg_drift_score)}
                color={kpis.avg_drift_score <= 0.2 ? '#22c55e' : kpis.avg_drift_score <= 0.35 ? '#f59e0b' : '#ef4444'}
                sub="Lower = less drift" />
            </div>
          </Card>

          {/* Accuracy Trend Line Chart */}
          <Card title="Global Accuracy Trend (by Round)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={accuracyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -5 }} />
                <YAxis domain={['auto', 'auto']} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={2} dot={{ fill: '#2563eb' }} name="Global Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Sites by Status Pie */}
          <Card title="Sites by Status">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={sitesByStatus} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={90} label={({ status, count }) => `${status}: ${count}`}>
                  {sitesByStatus.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Sites by Region Pie */}
          <Card title="Sites by Region">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={sitesByRegion} dataKey="count" nameKey="region" cx="50%" cy="50%" outerRadius={90} label={({ region, count }) => `${region}: ${count}`}>
                  {sitesByRegion.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Aggregation Methods Bar */}
          <Card title="Aggregation Methods Used" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={aggregationMethods}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="method" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} name="Rounds" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Site Detail Tab */}
      {tab === 'site_detail' && (
        <Card title="Federation Sites">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Site ID</th>
                  <th style={thStyle}>Site Name</th>
                  <th style={thStyle}>Region</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Patients</th>
                  <th style={thStyle}>Accuracy</th>
                  <th style={thStyle}>Drift</th>
                  <th style={thStyle}>Data Quality</th>
                  <th style={thStyle}>Model</th>
                  <th style={thStyle}>Last Sync</th>
                </tr>
              </thead>
              <tbody>
                {sites.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={tdStyle}><strong>{s.site_id}</strong></td>
                    <td style={tdStyle}>{s.site_name}</td>
                    <td style={tdStyle}>{s.region}</td>
                    <td style={tdStyle}><StatusBadge status={s.status} /></td>
                    <td style={tdStyle}>{s.patients_contributed}</td>
                    <td style={tdStyle}>
                      <span style={{ color: s.local_accuracy >= 85 ? '#22c55e' : s.local_accuracy >= 75 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                        {fmt(s.local_accuracy)}%
                      </span>
                    </td>
                    <td style={tdStyle}>
                      <span style={{ color: s.drift_score <= 0.2 ? '#22c55e' : s.drift_score <= 0.35 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                        {s.drift_score.toFixed(3)}
                      </span>
                    </td>
                    <td style={tdStyle}>
                      <span style={{ color: s.data_quality_score >= 0.8 ? '#22c55e' : '#f59e0b', fontWeight: 600 }}>
                        {s.data_quality_score.toFixed(2)}
                      </span>
                    </td>
                    <td style={tdStyle}><Badge text={s.model_version} color="#6366f1" /></td>
                    <td style={tdStyle} title={s.last_sync}>{s.last_sync ? s.last_sync.slice(0, 16).replace('T', ' ') : '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Round History Tab */}
      {tab === 'round_history' && (
        <Card title="Federation Rounds">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Round</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Sites</th>
                  <th style={thStyle}>Acc Before</th>
                  <th style={thStyle}>Acc After</th>
                  <th style={thStyle}>Delta</th>
                  <th style={thStyle}>Method</th>
                  <th style={thStyle}>Conv. Delta</th>
                  <th style={thStyle}>Started</th>
                  <th style={thStyle}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {rounds.map((r, i) => {
                  const delta = r.global_accuracy_after - r.global_accuracy_before
                  return (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}><strong>R{r.round_id}</strong></td>
                      <td style={tdStyle}><StatusBadge status={r.status} /></td>
                      <td style={tdStyle}>{r.participating_sites}</td>
                      <td style={tdStyle}>{fmt(r.global_accuracy_before)}%</td>
                      <td style={tdStyle}>
                        <span style={{ fontWeight: 600, color: r.status === 'completed' ? '#22c55e' : '#64748b' }}>
                          {fmt(r.global_accuracy_after)}%
                        </span>
                      </td>
                      <td style={tdStyle}>
                        <span style={{ color: delta >= 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                          {delta >= 0 ? '+' : ''}{delta.toFixed(1)}%
                        </span>
                      </td>
                      <td style={tdStyle}><Badge text={r.aggregation_method} color="#8b5cf6" /></td>
                      <td style={tdStyle}>{r.convergence_delta.toFixed(4)}</td>
                      <td style={tdStyle}>{r.started_at ? r.started_at.slice(0, 16).replace('T', ' ') : '--'}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#64748b', maxWidth: 220 }}>{r.notes}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Drift Monitor Tab */}
      {tab === 'drift_monitor' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* Per-Site Drift Bar Chart */}
          <Card title="Drift Score by Site" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sites.sort((a, b) => b.drift_score - a.drift_score)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="site_name" angle={-25} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 0.5]} label={{ value: 'Drift Score', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="drift_score" name="Drift Score" radius={[6, 6, 0, 0]}>
                  {sites.map((s, i) => (
                    <Cell key={i} fill={s.drift_score <= 0.2 ? '#22c55e' : s.drift_score <= 0.35 ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Per-Site Accuracy Bar Chart */}
          <Card title="Local Accuracy by Site">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={perSiteAccuracy} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[60, 100]} />
                <YAxis dataKey="site" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="accuracy" fill="#2563eb" radius={[0, 6, 6, 0]} name="Accuracy %" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Top Drifting Sites Table */}
          <Card title="Top Drifting Sites (Alerts)" span={3}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Site ID</th>
                  <th style={thStyle}>Site Name</th>
                  <th style={thStyle}>Drift Score</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Alert Level</th>
                </tr>
              </thead>
              <tbody>
                {topDrifting.map((s, i) => {
                  const alertLevel = s.drift_score > 0.35 ? 'High' : s.drift_score > 0.2 ? 'Medium' : 'Low'
                  const alertColor = s.drift_score > 0.35 ? '#ef4444' : s.drift_score > 0.2 ? '#f59e0b' : '#22c55e'
                  return (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}><strong>{s.site_id}</strong></td>
                      <td style={tdStyle}>{s.site_name}</td>
                      <td style={tdStyle}>
                        <span style={{ fontWeight: 600, color: alertColor }}>{s.drift_score.toFixed(3)}</span>
                      </td>
                      <td style={tdStyle}><StatusBadge status={s.status} /></td>
                      <td style={tdStyle}><Badge text={alertLevel} color={alertColor} /></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          <Card>
            <h3 style={{ margin: '0 0 16px', fontSize: 16, color: '#0f172a' }}>{defs.title}</h3>
            {(defs.sections || []).map((sec, si) => (
              <div key={si} style={{ marginBottom: 20 }}>
                <h4 style={{ fontSize: 14, color: '#334155', margin: '0 0 8px', borderBottom: '1px solid #e2e8f0', paddingBottom: 6 }}>{sec.heading}</h4>
                {(sec.items || []).map((item, ii) => {
                  const defKey = `${si}-${ii}`
                  const isExpanded = expandedDef === defKey
                  return (
                    <div key={ii} style={{ marginBottom: 6 }}>
                      <button
                        onClick={() => setExpandedDef(isExpanded ? null : defKey)}
                        style={{
                          display: 'block', width: '100%', textAlign: 'left',
                          padding: '8px 12px', background: isExpanded ? '#eff6ff' : '#f8fafc',
                          border: '1px solid #e2e8f0', borderRadius: 6, cursor: 'pointer',
                          fontSize: 13, fontWeight: 600, color: '#1e293b'
                        }}
                      >
                        {isExpanded ? '\u25BC' : '\u25B6'} {item.term}
                      </button>
                      {isExpanded && (
                        <div style={{ padding: '10px 14px', fontSize: 13, color: '#475569', lineHeight: 1.6, background: '#f8fafc', borderRadius: '0 0 6px 6px', borderLeft: '3px solid #2563eb' }}>
                          {item.detail}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
