import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#8bc34a']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function ToolExecutionDashboard() {
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
          axios.get(`${API_URL}/api/tool-execution/overview`),
          axios.get(`${API_URL}/api/tool-execution/breakdown`),
          axios.get(`${API_URL}/api/tool-execution/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load tool execution data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading tool execution data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Tool execution data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const topTools = overview.top_tools || []
  const topActions = overview.top_actions || []
  const actors = overview.actors || []
  const trend = overview.daily_trend || []
  const tools = breakdown?.tools || []
  const defsList = defs?.metrics || []

  const kpiItems = [
    { label: 'Total Executions', value: s.total_executions, color: COLORS[0] },
    { label: 'Unique Tools', value: s.unique_tools, color: COLORS[1] },
    { label: 'Success Rate', value: s.success_rate_pct, color: COLORS[2], unit: '%', decimals: 1 },
    { label: 'Avg Daily', value: s.avg_daily_executions, color: COLORS[3], decimals: 1 },
  ]

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
    border: '1px solid #e5e7eb',
  }

  const kpiCardStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    flex: 1,
    minWidth: 150,
    padding: 16,
  })

  const sectionHeading = { fontSize: 16, fontWeight: 700, margin: '20px 0 12px', color: '#334155' }

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          Tool Execution Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          {s.total_executions != null ? `${s.total_executions} total executions across ${s.unique_tools} tools` : 'Tracking tool executions'}
          {s.success_rate_pct != null ? ` — ${fmt(s.success_rate_pct, 1)}% success rate` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiCardStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: kpi.color }}>
              {fmt(kpi.value, kpi.decimals || 0)}{kpi.unit || ''}
            </div>
          </div>
        ))}
      </div>

      {/* Top Tools Bar Chart */}
      {topTools.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Top Tools by Execution Count</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={topTools}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} />
              <YAxis type="category" dataKey="tool" tick={{ fontSize: 12, fill: '#475569' }} width={115} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val) => [val, 'Executions']}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {topTools.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Trend + Actions side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* 14-day Trend */}
        {trend.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>14-Day Execution Trend</h3>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={trend} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#475569' }} tickFormatter={d => d.slice(5)} />
                <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
                <Legend />
                <Line type="monotone" dataKey="executions" stroke={COLORS[0]} strokeWidth={2} dot={{ r: 3 }} name="Executions" />
                <Line type="monotone" dataKey="unique_tools" stroke={COLORS[1]} strokeWidth={2} dot={{ r: 3 }} name="Unique Tools" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Actor Pie Chart */}
        {actors.length > 0 && (
          <div style={cardStyle}>
            <h3 style={{ ...sectionHeading, margin: '0 0 12px' }}>Executions by Actor</h3>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={actors}
                  dataKey="count"
                  nameKey="actor"
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  label={({ actor, count }) => `${actor}: ${count}`}
                >
                  {actors.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Per-Tool Breakdown Table */}
      {tools.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Per-Tool Breakdown</h3>
          <div style={{ maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Tool</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Executions</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>% of Total</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Top Actions</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Last Seen</th>
                </tr>
              </thead>
              <tbody>
                {tools.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>{t.tool}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#1e88e5', fontWeight: 600 }}>{t.executions}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#475569' }}>{fmt(t.pct_of_total, 1)}%</td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>
                      {Object.entries(t.top_actions || {}).slice(0, 3).map(([a, c]) => `${a}(${c})`).join(', ')}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 12 }}>
                      {t.last_seen ? t.last_seen.slice(0, 10) : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} Metric Definitions
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Unit</th>
                </tr>
              </thead>
              <tbody>
                {defsList.map((def, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {def.name || `Metric ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '--'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.unit || '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {defs.clinical_relevance && (
              <p style={{ marginTop: 16, padding: '12px 14px', background: '#f0fdf4', borderRadius: 8, fontSize: 13, color: '#166534', lineHeight: 1.5 }}>
                <strong>Clinical Relevance:</strong> {defs.clinical_relevance}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
