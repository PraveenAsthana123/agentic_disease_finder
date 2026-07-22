import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

function fmtUsd(v) {
  if (v == null) return '--'
  return '$' + (typeof v === 'number' ? v.toFixed(2) : v)
}

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

function CostBadge({ value }) {
  const raw = typeof value === 'string' ? parseFloat(value.replace('$', '')) : value
  const color = raw > 0.2 ? '#ef4444' : raw > 0.1 ? '#f59e0b' : '#10b981'
  return <span style={{ background: color + '18', color, padding: '2px 8px', borderRadius: 8, fontSize: 12, fontWeight: 600 }}>{typeof value === 'string' ? value : fmtUsd(value)}</span>
}

export default function AIFinOpsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/ai-finops/overview`),
          axios.get(`${API_URL}/api/ai-finops/breakdown`),
          axios.get(`${API_URL}/api/ai-finops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AI FinOps data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI FinOps data...</div>
  )
  if (error) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview?.available) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>AI FinOps data unavailable</div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Breakdown' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const successRate = overview.total_builds > 0
    ? ((overview.successful_builds / overview.total_builds) * 100).toFixed(1)
    : '0'

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Cost KPIs" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
          <KPI label="Total Cost" value={fmtUsd(overview.total_cost)} color="#3b82f6" />
          <KPI label="Last 24h" value={fmtUsd(overview.cost_last_24h)} color="#8b5cf6" />
          <KPI label="Last 7d" value={fmtUsd(overview.cost_last_7d)} color="#10b981" />
          <KPI label="Cost / Build" value={fmtUsd(overview.cost_per_build)} sub={`${fmt(overview.total_builds)} builds`} />
          <KPI label="Total Events" value={fmt(overview.total_events)} sub={`${successRate}% success`} />
        </div>
      </Card>

      <Card title="Build Stats">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          <KPI label="Total Builds" value={fmt(overview.total_builds)} color="#3b82f6" />
          <KPI label="Successful" value={fmt(overview.successful_builds)} color="#10b981" />
          <KPI label="Build Minutes" value={fmt(overview.total_build_minutes)} color="#f59e0b" />
          <KPI label="Storage (MB)" value={fmt(overview.total_storage_mb)} color="#64748b" />
        </div>
      </Card>

      <Card title="Cost by Category" span={2}>
        {(overview.cost_breakdown || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={overview.cost_breakdown.map(c => ({ name: c.category, value: c.cost }))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}>
                {overview.cost_breakdown.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={v => fmtUsd(v)} />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Category Breakdown">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Category</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Cost</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Events</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>%</th>
            </tr>
          </thead>
          <tbody>
            {(overview.cost_breakdown || []).map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px' }}>{c.category}</td>
                <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{fmtUsd(c.cost)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(c.events)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(c.percent)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Daily Cost Trend" span={2}>
        {(overview.daily_costs || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={overview.daily_costs}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
              <YAxis tickFormatter={v => `$${v}`} />
              <Tooltip formatter={(v, name) => [name === 'cost' ? fmtUsd(v) : fmt(v), name]} />
              <Legend />
              <Line type="monotone" dataKey="cost" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>

      <Card title="Daily Builds & Health Checks">
        {(overview.daily_costs || []).length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={overview.daily_costs}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="builds" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="health_checks" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Model Storage" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Model</th>
              <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>File</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Size (MB)</th>
              <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Monthly Cost</th>
            </tr>
          </thead>
          <tbody>
            {(overview.model_storage || []).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.name}</td>
                <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12, color: '#64748b' }}>{m.file}</td>
                <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(m.size_mb)}</td>
                <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmtUsd(m.monthly_cost)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )

  const renderBreakdown = () => {
    const bd = breakdown || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Cost Velocity (Cumulative)" span={2}>
          {(bd.cost_velocity || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={bd.cost_velocity}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
                <YAxis tickFormatter={v => `$${v}`} />
                <Tooltip formatter={(v, name) => [fmtUsd(v), name === 'cumulative' ? 'Cumulative' : 'Daily']} />
                <Legend />
                <Area type="monotone" dataKey="cumulative" stroke="#3b82f6" fill="#3b82f620" strokeWidth={2} />
                <Area type="monotone" dataKey="daily" stroke="#10b981" fill="#10b98120" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        <Card title="Cost Efficiency by Category">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Level</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Events</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Cost/Event</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {(bd.efficiency || []).map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600, textTransform: 'capitalize' }}>{e.level}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(e.events)}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}><CostBadge value={e.cost_per_event} /></td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{e.total_cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Model Storage Distribution" span={2}>
          {(bd.storage_breakdown || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bd.storage_breakdown} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={v => `${v} MB`} />
                <YAxis type="category" dataKey="model" width={100} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v, name) => [name === 'size_mb' ? `${v} MB` : `${v}%`, name === 'size_mb' ? 'Size' : 'Percent']} />
                <Bar dataKey="size_mb" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
        </Card>

        <Card title="Hourly Cost Heatmap (7d)" span={3}>
          {bd.hourly_heatmap ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '4px 6px', color: '#475569', fontSize: 10 }}>Day</th>
                    {(bd.hourly_heatmap.hour_labels || []).map((h, i) => (
                      <th key={i} style={{ padding: '4px 3px', color: '#94a3b8', fontSize: 9, textAlign: 'center' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.hourly_heatmap.matrix || []).map((row, ri) => (
                    <tr key={ri}>
                      <td style={{ padding: '4px 6px', fontSize: 10, color: '#475569', whiteSpace: 'nowrap' }}>{(bd.hourly_heatmap.day_labels || [])[ri] || ''}</td>
                      {row.map((val, ci) => {
                        const intensity = Math.min(val / 0.7, 1)
                        const bg = val === 0 ? '#f8fafc' : `rgba(59, 130, 246, ${0.15 + intensity * 0.7})`
                        const textColor = intensity > 0.5 ? '#fff' : '#334155'
                        return (
                          <td key={ci} style={{
                            padding: '4px 2px', textAlign: 'center', background: bg,
                            color: textColor, fontSize: 9, borderRadius: 2, border: '1px solid #f1f5f9'
                          }}>
                            {val > 0 ? `$${val.toFixed(2)}` : ''}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No heatmap data</div>}
        </Card>

        <Card title="Top Expensive Builds" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#fef2f2' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#991b1b' }}>Start Time</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#991b1b' }}>Duration (min)</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#991b1b' }}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {(bd.top_expensive_builds || []).map((b, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{b.start}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(b.duration_min)}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}><CostBadge value={b.cost} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Recent Build Log">
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Start</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Min</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {(bd.build_log || []).slice(0, 20).map((b, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '5px 10px', fontFamily: 'monospace', fontSize: 11 }}>{b.start}</td>
                    <td style={{ padding: '5px 10px', textAlign: 'right' }}>{fmt(b.duration_min)}</td>
                    <td style={{ padding: '5px 10px', textAlign: 'right' }}>{b.cost}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Downtime Impact">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <KPI label="Downtime Events" value={fmt(bd.downtime_events)} color={bd.downtime_events > 0 ? '#ef4444' : '#10b981'} />
            <KPI label="Downtime Cost" value={fmtUsd(bd.downtime_cost)} color={bd.downtime_cost > 0 ? '#ef4444' : '#10b981'} />
          </div>
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    const df = defs || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Cost Model" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Resource</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Rate</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(df.cost_model || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.resource}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: '#3b82f6' }}>{c.rate}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{c.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Metrics Glossary" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Term</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
              </tr>
            </thead>
            <tbody>
              {(df.metrics || []).map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.term}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Optimization Strategies" span={2}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f0fdf4' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#166534' }}>Strategy</th>
                <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '2px solid #e2e8f0', color: '#166534' }}>Savings</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#166534' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(df.optimization_strategies || []).map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.strategy}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                    <span style={{ background: '#10b98118', color: '#10b981', padding: '2px 8px', borderRadius: 8, fontSize: 12, fontWeight: 600 }}>{s.savings}</span>
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>AI FinOps Dashboard</h2>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9', color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>
      {tab === 'overview' && renderOverview()}
      {tab === 'breakdown' && renderBreakdown()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
