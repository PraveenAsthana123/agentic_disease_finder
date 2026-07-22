import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const CAT_COLORS = {
  llm_inference: '#3b82f6', gpu_compute: '#f59e0b', cloud_infra: '#10b981'
}
const CAT_LABELS = {
  llm_inference: 'LLM Inference', gpu_compute: 'GPU Compute', cloud_infra: 'Cloud Infra'
}

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

function BudgetBar({ used, total }) {
  const pct = Math.min(used, 100)
  const barColor = pct > 90 ? '#ef4444' : pct > 70 ? '#f59e0b' : '#10b981'
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: '#64748b' }}>Budget: {fmtUsd(total)}/mo</span>
        <span style={{ color: barColor, fontWeight: 600 }}>{pct.toFixed(1)}% used</span>
      </div>
      <div style={{ background: '#f1f5f9', borderRadius: 6, height: 10, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: barColor, borderRadius: 6, transition: 'width 0.5s' }} />
      </div>
    </div>
  )
}

export default function FinOpsDashboard() {
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
          axios.get(`${API_URL}/api/finops/overview`),
          axios.get(`${API_URL}/api/finops/breakdown`),
          axios.get(`${API_URL}/api/finops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load FinOps data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading FinOps data...</div>
  )
  if (error) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview?.available) return (
    <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>FinOps data unavailable</div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'models', label: 'Models & GPU' },
    { id: 'trends', label: 'Trends & Costs' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpi = overview.kpi

  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Cost KPIs" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 16 }}>
          <KPI label="30-Day Spend" value={fmtUsd(kpi.total_30d_usd)} color="#3b82f6" />
          <KPI label="7-Day Spend" value={fmtUsd(kpi.total_7d_usd)} color="#8b5cf6" />
          <KPI label="Today" value={fmtUsd(kpi.total_today_usd)} color="#10b981" />
          <KPI label="Cost / Request" value={fmtUsd(kpi.cost_per_request)} sub={`${fmt(kpi.total_requests_30d)} reqs`} />
          <KPI label="Cost / Patient" value={fmtUsd(kpi.cost_per_patient)} sub={`${kpi.unique_patients_30d} patients`} />
        </div>
        <BudgetBar used={kpi.budget_used_pct} total={kpi.monthly_budget_usd} />
      </Card>

      <Card title="Spend by Category">
        {overview.category_breakdown.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={overview.category_breakdown.map(c => ({
                name: CAT_LABELS[c.category] || c.category, value: c.cost_usd
              }))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: $${value}`}>
                {overview.category_breakdown.map((c, i) => (
                  <Cell key={i} fill={CAT_COLORS[c.category] || COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={v => fmtUsd(v)} />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Top Services by Cost" span={2}>
        {overview.top_services.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={overview.top_services} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tickFormatter={v => `$${v}`} />
              <YAxis type="category" dataKey="service" width={140} tick={{ fontSize: 12 }} />
              <Tooltip formatter={v => fmtUsd(v)} />
              <Bar dataKey="cost_usd" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      <Card title="Daily Cost Trend (30d)" span={3}>
        {overview.daily_trend.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={overview.daily_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
              <YAxis tickFormatter={v => `$${v}`} />
              <Tooltip formatter={v => fmtUsd(v)} />
              <Line type="monotone" dataKey="cost_usd" stroke="#3b82f6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>
    </div>
  )

  const renderModels = () => {
    const bd = breakdown || {}
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="LLM Model Costs (30d)" span={2}>
          {(bd.model_costs || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Model</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Requests</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Tokens In</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Tokens Out</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Avg Tok/Req</th>
                  <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Cost</th>
                </tr>
              </thead>
              <tbody>
                {bd.model_costs.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.model}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(m.requests)}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(m.tokens_in)}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(m.tokens_out)}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px' }}>{fmt(m.avg_tokens_per_request)}</td>
                    <td style={{ textAlign: 'right', padding: '8px 6px', fontWeight: 600, color: '#3b82f6' }}>{fmtUsd(m.cost_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model data</div>}
        </Card>

        <Card title="GPU Utilization (30d)">
          {(bd.gpu_usage || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bd.gpu_usage}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="task" tick={{ fontSize: 10 }} />
                <YAxis tickFormatter={v => `${v} min`} />
                <Tooltip formatter={(v, name) => name === 'cost_usd' ? fmtUsd(v) : `${v} min`} />
                <Bar dataKey="gpu_minutes" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No GPU data</div>}
        </Card>

        <Card title="Component Spend (30d)" span={2}>
          {(bd.component_spend || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={bd.component_spend} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={v => `$${v}`} />
                <YAxis type="category" dataKey="component" width={120} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => fmtUsd(v)} />
                <Bar dataKey="cost_usd" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No component data</div>}
        </Card>

        <Card title="Cloud Services (30d)">
          {(bd.cloud_costs || []).length > 0 ? (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Service</th>
                  <th style={{ textAlign: 'left', padding: '6px', color: '#64748b' }}>Type</th>
                  <th style={{ textAlign: 'right', padding: '6px', color: '#64748b' }}>30d Cost</th>
                </tr>
              </thead>
              <tbody>
                {bd.cloud_costs.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{c.service}</td>
                    <td style={{ padding: '6px', color: '#64748b' }}>{c.type}</td>
                    <td style={{ textAlign: 'right', padding: '6px', color: '#10b981', fontWeight: 600 }}>{fmtUsd(c.cost_30d_usd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No cloud data</div>}
        </Card>
      </div>
    )
  }

  const renderTrends = () => {
    const bd = breakdown || {}
    const wc = bd.weekly_comparison || {}
    const changeColor = wc.change_pct > 0 ? '#ef4444' : '#10b981'
    const changeArrow = wc.change_pct > 0 ? '\u2191' : '\u2193'
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Week-over-Week Comparison">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <KPI label="This Week" value={fmtUsd(wc.this_week_usd)} color="#3b82f6" />
            <KPI label="Last Week" value={fmtUsd(wc.last_week_usd)} color="#64748b" />
            <KPI label="Change" value={`${changeArrow} ${Math.abs(wc.change_pct || 0).toFixed(1)}%`} color={changeColor} />
          </div>
        </Card>

        <Card title="Token Volume (30d)">
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <KPI label="Total Tokens" value={fmt(kpi.total_tokens_30d)} color="#8b5cf6" />
            <KPI label="GPU Minutes" value={fmt(kpi.total_gpu_minutes_30d)} color="#f59e0b" />
          </div>
        </Card>

        <Card title="Per-Patient Cost (Top 15)" span={3}>
          {(bd.patient_costs || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.patient_costs}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tickFormatter={v => `$${v}`} />
                <Tooltip formatter={(v, name) => name === 'cost_usd' ? fmtUsd(v) : fmt(v)} />
                <Bar dataKey="cost_usd" fill="#ec4899" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient cost data</div>}
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    const metrics = defs?.metrics || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Metric Definitions">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b', width: 200 }}>Metric</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.name}</td>
                  <td style={{ padding: '8px 6px', color: '#475569' }}>{m.desc}</td>
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
      <h2 style={{ margin: '0 0 6px', fontSize: 22, color: '#1e293b' }}>FinOps Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Token, GPU, and cloud cost analytics — budget tracking and per-patient cost attribution
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'models' && renderModels()}
      {tab === 'trends' && renderTrends()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
