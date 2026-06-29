import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function fmt(v, decimals = 2) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function AICostDashboard() {
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
          axios.get(`${API_URL}/ai-cost/overview`),
          axios.get(`${API_URL}/ai-cost/breakdown`),
          axios.get(`${API_URL}/ai-cost/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI cost data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI cost and resource data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'AI cost data not available. Ensure the cost tracking service is running.'}
    </div>
  )

  const summary = overview.summary || {}
  const resources = overview.resources || {}
  const daily = overview.daily_trend || []
  const components = Array.isArray(breakdown) ? breakdown : (breakdown?.components || [])
  const defsList = Array.isArray(defs?.cost_categories) ? defs.cost_categories : (Array.isArray(defs) ? defs : [])

  const kpiItems = [
    { label: 'Total Operations', value: summary.total_operations, color: COLORS[0], unit: '' },
    { label: 'Est. Monthly Cost', value: summary.estimated_monthly_cost, color: COLORS[2], unit: '$', decimals: 4 },
    { label: 'Carbon Footprint', value: summary.carbon_footprint_kg, color: COLORS[3], unit: 'kg CO₂', decimals: 4 },
    { label: 'Active Models', value: summary.active_models, color: COLORS[1], unit: '' },
  ]

  const topComponents = [...components]
    .sort((a, b) => (b.estimated_cost || 0) - (a.estimated_cost || 0))
    .slice(0, 8)
    .map(c => ({ name: c.component || c.name || 'Unknown', cost: c.estimated_cost || 0 }))

  const cpuPct = resources.cpu_utilization_pct != null ? resources.cpu_utilization_pct : null
  const memUsed = resources.memory_used_gb != null ? resources.memory_used_gb : null
  const memTotal = resources.memory_total_gb != null ? resources.memory_total_gb : null

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
          AI Cost &amp; Resource Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          {summary.total_operations != null ? `${summary.total_operations} total operations` : 'Tracking AI operations'}
          {summary.estimated_monthly_cost != null ? ` — Est. $${fmt(summary.estimated_monthly_cost, 4)} monthly` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiCardStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: kpi.color }}>
              {kpi.unit && kpi.unit !== '$' && kpi.unit !== 'kg CO₂'
                ? fmt(kpi.value, kpi.decimals || 0)
                : kpi.unit === '$'
                  ? `$${fmt(kpi.value, kpi.decimals || 4)}`
                  : kpi.unit === 'kg CO₂'
                    ? `${fmt(kpi.value, kpi.decimals || 4)} kg`
                    : fmt(kpi.value, kpi.decimals || 0)}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{kpi.unit || 'count'}</div>
          </div>
        ))}
      </div>

      {/* Cost by Component */}
      {topComponents.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Cost by Component</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={topComponents}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" tick={{ fontSize: 12, fill: '#475569' }} tickFormatter={v => `$${v.toFixed(4)}`} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12, fill: '#475569' }} width={95} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val) => [`$${val.toFixed(6)}`, 'Est. Cost']}
              />
              <Bar dataKey="cost" radius={[0, 4, 4, 0]}>
                {topComponents.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Resource Usage */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* CPU */}
        <div style={cardStyle}>
          <h3 style={{ ...sectionHeading, margin: '0 0 14px' }}>CPU Utilization</h3>
          {cpuPct != null ? (
            <>
              <div style={{ fontSize: 36, fontWeight: 700, color: cpuPct > 80 ? '#f44336' : cpuPct > 60 ? '#ff9800' : '#4caf50' }}>
                {fmt(cpuPct, 1)}%
              </div>
              <div style={{ marginTop: 10, background: '#e2e8f0', borderRadius: 6, height: 10 }}>
                <div style={{
                  width: `${Math.min(cpuPct, 100)}%`, height: '100%', borderRadius: 6,
                  background: cpuPct > 80 ? '#f44336' : cpuPct > 60 ? '#ff9800' : '#4caf50',
                  transition: 'width 0.4s ease',
                }} />
              </div>
              <div style={{ marginTop: 6, fontSize: 12, color: '#94a3b8' }}>
                {cpuPct > 80 ? 'High load' : cpuPct > 60 ? 'Moderate load' : 'Normal'}
              </div>
            </>
          ) : (
            <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No CPU data available</div>
          )}
        </div>

        {/* Memory */}
        <div style={cardStyle}>
          <h3 style={{ ...sectionHeading, margin: '0 0 14px' }}>Memory Usage</h3>
          {memUsed != null ? (
            <>
              <div style={{ fontSize: 36, fontWeight: 700, color: '#1e88e5' }}>
                {fmt(memUsed, 1)} <span style={{ fontSize: 18, color: '#64748b' }}>GB</span>
              </div>
              {memTotal != null && (
                <>
                  <div style={{ marginTop: 10, background: '#e2e8f0', borderRadius: 6, height: 10 }}>
                    <div style={{
                      width: `${Math.min((memUsed / memTotal) * 100, 100)}%`, height: '100%',
                      borderRadius: 6, background: '#1e88e5', transition: 'width 0.4s ease',
                    }} />
                  </div>
                  <div style={{ marginTop: 6, fontSize: 12, color: '#94a3b8' }}>
                    {fmt(memUsed, 1)} / {fmt(memTotal, 1)} GB ({fmt((memUsed / memTotal) * 100, 1)}% used)
                  </div>
                </>
              )}
            </>
          ) : (
            <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No memory data available</div>
          )}
        </div>
      </div>

      {/* Daily Cost Trend */}
      {daily.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={sectionHeading}>Cost Trend (Recent Days)</h3>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={daily} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#475569' }} />
              <YAxis yAxisId="ops" orientation="left" tick={{ fontSize: 12, fill: '#475569' }} />
              <YAxis yAxisId="cost" orientation="right" tick={{ fontSize: 12, fill: '#475569' }} tickFormatter={v => `$${v.toFixed(3)}`} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val, name) => name === 'cost' ? [`$${val.toFixed(4)}`, 'Est. Cost'] : [val, 'Operations']}
              />
              <Legend />
              <Bar yAxisId="ops" dataKey="operations" fill={COLORS[0]} radius={[4, 4, 0, 0]} name="Operations" />
              <Bar yAxisId="cost" dataKey="cost" fill={COLORS[2]} radius={[4, 4, 0, 0]} name="cost" />
            </BarChart>
          </ResponsiveContainer>
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
                      {def.name || def.metric_name || `Metric ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '--'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.unit || '--'}
                    </td>
                  </tr>
                ))}
                {defsList.length === 0 && (
                  <tr>
                    <td colSpan={3} style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No definitions available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
