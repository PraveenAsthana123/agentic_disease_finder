import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#607d8b']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

export default function AIUsageDashboard() {
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
          axios.get(`${API_URL}/ai-usage/overview`),
          axios.get(`${API_URL}/ai-usage/breakdown`),
          axios.get(`${API_URL}/ai-usage/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI usage data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI usage data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'AI usage data not available.'}
    </div>
  )

  const summary = overview.summary || {}
  const daily = overview.daily_trend || []
  const byComponent = Object.entries(overview.by_component || {}).map(([name, value]) => ({ name, value }))
  const byAction = Object.entries(overview.by_action || {}).map(([name, value]) => ({ name, value }))
  const byActor = Object.entries(overview.by_actor || {}).map(([name, value]) => ({ name, value }))
  const components = Array.isArray(breakdown?.components) ? breakdown.components : []

  const kpiItems = [
    { label: 'Total Operations', value: summary.total_operations, color: COLORS[0] },
    { label: "Today's Ops", value: summary.today_operations, color: COLORS[2] },
    { label: 'Unique Patients', value: summary.unique_patients, color: COLORS[1] },
    { label: 'Components', value: summary.unique_components, color: COLORS[3] },
    { label: 'Actors', value: summary.unique_actors, color: COLORS[4] },
    { label: 'Peak Hour (UTC)', value: summary.peak_hour_utc != null ? `${summary.peak_hour_utc}:00` : '--', color: COLORS[5] },
  ]

  const topComponents = byComponent.slice(0, 10)
  const topActions = byAction.slice(0, 10)

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 130
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Usage Dashboard</h2>
        <button
          onClick={() => setShowDefs(!showDefs)}
          style={{ background: '#f1f5f9', border: '1px solid #cbd5e1', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontSize: 13 }}
        >
          {showDefs ? 'Hide' : 'Show'} Definitions
        </button>
      </div>

      {showDefs && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#f0f9ff', border: '1px solid #bae6fd' }}>
          <h4 style={{ margin: '0 0 10px', color: '#0369a1' }}>Metric Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 8 }}>
              <strong style={{ color: '#0c4a6e' }}>{d.term}:</strong>{' '}
              <span style={{ color: '#334155', fontSize: 13 }}>{d.definition}</span>
            </div>
          ))}
        </div>
      )}

      {/* KPI cards */}
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

      {/* Daily trend */}
      {daily.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Operations (14-day trend)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={daily}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="operations" stroke={COLORS[0]} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Component + Action charts side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
        {topComponents.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Operations by Component</h4>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={topComponents} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="value" fill={COLORS[0]} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {topActions.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Operations by Action</h4>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={topActions} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {topActions.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Actor breakdown */}
      {byActor.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Operations by Actor</h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={byActor}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" fill={COLORS[1]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Component detail table */}
      {components.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Component Detail</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Component</th>
                  <th style={{ padding: '8px 10px' }}>Operations</th>
                  <th style={{ padding: '8px 10px' }}>Top Actions</th>
                  <th style={{ padding: '8px 10px' }}>Actors</th>
                  <th style={{ padding: '8px 10px' }}>Last Activity</th>
                </tr>
              </thead>
              <tbody>
                {components.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.component}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(c.operations)}</td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>
                      {Object.entries(c.actions || {}).slice(0, 3).map(([a, n]) => `${a}(${n})`).join(', ')}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>
                      {Object.entries(c.actors || {}).slice(0, 3).map(([a, n]) => `${a}(${n})`).join(', ')}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12 }}>
                      {c.last_activity ? c.last_activity.slice(0, 16).replace('T', ' ') : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Data range: {overview.date_range?.first?.slice(0, 10) || '—'} to {overview.date_range?.last?.slice(0, 10) || '—'}
      </div>
    </div>
  )
}
