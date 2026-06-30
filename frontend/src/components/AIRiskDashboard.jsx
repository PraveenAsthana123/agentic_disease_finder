import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const SEV_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#3b82f6' }
const STATUS_COLORS = { open: '#ef4444', monitoring: '#f59e0b', mitigated: '#10b981' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AIRiskDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/ai-risk/overview`),
          axios.get(`${API_URL}/ai-risk/breakdown`),
          axios.get(`${API_URL}/ai-risk/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI risk data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI risk data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'AI risk data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const risks = overview.risks || []
  const daily = overview.daily_trend || []
  const alertInstr = Object.entries(overview.alert_by_instrument || {}).map(([name, value]) => ({ name, value }))
  const categories = Array.isArray(breakdown?.categories) ? breakdown.categories : []

  const sevData = [
    { name: 'High', value: s.high_severity, fill: SEV_COLORS.high },
    { name: 'Medium', value: s.medium_severity, fill: SEV_COLORS.medium },
    { name: 'Low', value: s.low_severity, fill: SEV_COLORS.low }
  ].filter(d => d.value > 0)

  const statusData = [
    { name: 'Open', value: s.open, fill: STATUS_COLORS.open },
    { name: 'Monitoring', value: s.monitoring, fill: STATUS_COLORS.monitoring },
    { name: 'Mitigated', value: s.mitigated, fill: STATUS_COLORS.mitigated }
  ].filter(d => d.value > 0)

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }

  const kpiItems = [
    { label: 'Total Risks', value: s.total_risks, color: '#64748b' },
    { label: 'High Severity', value: s.high_severity, color: SEV_COLORS.high },
    { label: 'Medium Severity', value: s.medium_severity, color: SEV_COLORS.medium },
    { label: 'Mitigated %', value: s.mitigated_pct != null ? `${s.mitigated_pct}%` : '--', color: '#10b981' },
    { label: 'Clinical Alerts', value: s.clinical_alerts, color: '#f97316' },
    { label: 'Guardrail Blocks', value: s.guardrail_blocks, color: '#8b5cf6' },
    { label: 'Seizure Episodes', value: s.seizure_episodes, color: '#ec4899' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  const sevBadge = (sev) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${SEV_COLORS[sev] || '#94a3b8'}22`, color: SEV_COLORS[sev] || '#64748b'
  })

  const statusBadge = (st) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${STATUS_COLORS[st] || '#94a3b8'}22`, color: STATUS_COLORS[st] || '#64748b'
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Risk Dashboard</h2>
        <button
          onClick={() => setShowDefs(!showDefs)}
          style={{ background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
        >
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {showDefs && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#fef2f2', border: '1px solid #fecaca' }}>
          <h4 style={{ margin: '0 0 10px', color: '#991b1b' }}>Risk Metric Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <strong style={{ color: '#7f1d1d' }}>{d.term}:</strong>{' '}
              <span style={{ color: '#334155', fontSize: 13 }}>{d.definition}</span>
            </div>
          ))}
        </div>
      )}

      {/* KPI row */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {kpiItems.map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
              {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
            </div>
          </div>
        ))}
      </div>

      {/* Severity + Status donut charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risks by Severity</h4>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                label={({ name, value }) => `${name}: ${value}`}>
                {sevData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risks by Status</h4>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={statusData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                label={({ name, value }) => `${name}: ${value}`}>
                {statusData.map((d, i) => <Cell key={i} fill={d.fill} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Daily risk trend */}
      {daily.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Risk Activity (alerts + guardrail blocks)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={daily}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="alerts" stroke={SEV_COLORS.high} strokeWidth={2} dot={{ r: 3 }} name="Clinical Alerts" />
              <Line type="monotone" dataKey="guardrail_blocks" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="Guardrail Blocks" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Alerts by instrument */}
      {alertInstr.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Clinical Alerts by Instrument</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={alertInstr}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill="#f97316" radius={[4, 4, 0, 0]} name="Alerts" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Risk register table */}
      {risks.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Register</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>ID</th>
                  <th style={{ padding: '8px 10px' }}>Category</th>
                  <th style={{ padding: '8px 10px' }}>Risk</th>
                  <th style={{ padding: '8px 10px' }}>Severity</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                  <th style={{ padding: '8px 10px' }}>Count</th>
                  <th style={{ padding: '8px 10px' }}>Mitigation</th>
                </tr>
              </thead>
              <tbody>
                {risks.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{r.id}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.category}</td>
                    <td style={{ padding: '8px 10px', color: '#1e293b' }}>{r.title}</td>
                    <td style={{ padding: '8px 10px' }}><span style={sevBadge(r.severity)}>{r.severity}</span></td>
                    <td style={{ padding: '8px 10px' }}><span style={statusBadge(r.status)}>{r.status}</span></td>
                    <td style={{ padding: '8px 10px' }}>{r.count != null ? fmt(r.count) : '--'}</td>
                    <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{r.mitigation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Category breakdown */}
      {categories.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Risk Categories</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
            {categories.map((cat, i) => (
              <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 6 }}>{cat.category}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>
                  Total: {cat.total} &nbsp;|&nbsp;
                  {cat.high > 0 && <span style={{ color: SEV_COLORS.high }}>High: {cat.high} </span>}
                  {cat.medium > 0 && <span style={{ color: SEV_COLORS.medium }}>Med: {cat.medium} </span>}
                  {cat.low > 0 && <span style={{ color: SEV_COLORS.low }}>Low: {cat.low}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: clinical.db (assessments, seizure_diary, transaction_log, medications, hitl_reviews)
      </div>
    </div>
  )
}
