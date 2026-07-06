import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = '/api'
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
    ok: '#22c55e', healthy: '#22c55e', complete: '#22c55e', computed: '#22c55e',
    warning: '#eab308', partial: '#eab308', 'in-progress': '#eab308',
    critical: '#ef4444', failed: '#ef4444', 'over-budget': '#ef4444',
    pending: '#94a3b8', low: '#94a3b8',
    high: '#22c55e', medium: '#eab308',
    left: '#3b82f6', right: '#f97316', bilateral: '#8b5cf6'
  }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function TokenCostDashboard() {
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
          axios.get(`${API_URL}/token-cost/overview`),
          axios.get(`${API_URL}/token-cost/breakdown`),
          axios.get(`${API_URL}/token-cost/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Token Cost data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'token-breakdown', label: 'Token Breakdown' },
    { id: 'cost-analysis', label: 'Cost Analysis' },
    { id: 'operations', label: 'Operations' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const summary = overview?.summary || {}
  const llmUsage = overview?.llm_usage || {}
  const operations = overview?.operations || {}
  const inferences = overview?.inferences || {}
  const budget = overview?.budget || {}
  const dailyTrend = overview?.daily_trend || []

  /* --- Breakdown data --- */
  const tokenBreakdown = breakdown?.token_breakdown || {}
  const byRole = tokenBreakdown.by_role || []
  const componentBreakdown = breakdown?.component_breakdown || []
  const modelBreakdown = breakdown?.model_breakdown || []

  /* --- Operations breakdown data --- */
  const byComponent = operations.by_component || {}
  const byAction = operations.by_action || {}
  const byComponentArr = Object.entries(byComponent).map(([name, count]) => ({ name, count }))
  const byActionArr = Object.entries(byAction).map(([name, count]) => ({ name, count }))

  /* --- Budget gauge helpers --- */
  const budgetSpent = budget.monthly_spent_usd || 0
  const budgetLimit = budget.monthly_limit_usd || 1
  const budgetPct = budget.utilization_pct != null ? budget.utilization_pct : (budgetSpent / budgetLimit) * 100
  const budgetColor = budgetPct >= 100 ? '#ef4444' : budgetPct >= 80 ? '#eab308' : '#22c55e'

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Token Cost Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          LLM token usage, inference cost tracking, and budget utilization across the clinical AI pipeline
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Tokens" value={fmt(summary.total_tokens)} color="#3b82f6" />
              <KPI label="Total Cost (USD)" value={summary.total_cost_usd != null ? `$${summary.total_cost_usd.toFixed(4)}` : '--'} color="#ef4444" />
              <KPI label="Monthly Cost (USD)" value={summary.monthly_cost_usd != null ? `$${summary.monthly_cost_usd.toFixed(4)}` : '--'} color="#f97316" />
              <KPI label="Budget Utilization" value={summary.budget_utilization_pct != null ? summary.budget_utilization_pct.toFixed(1) + '%' : '--'} color={budgetColor} />
              <KPI label="Total Operations" value={fmt(summary.total_operations)} color="#8b5cf6" />
              <KPI label="Model Inferences" value={fmt(summary.model_inferences)} color="#22c55e" />
            </div>
          </Card>

          {/* Daily Trend AreaChart */}
          <Card title="Daily Token & Cost Trend" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={dailyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Area yAxisId="left" type="monotone" dataKey="input_tokens" name="Input Tokens" stroke={COLORS[0]} fill={COLORS[0] + '33'} strokeWidth={2} />
                <Area yAxisId="left" type="monotone" dataKey="output_tokens" name="Output Tokens" stroke={COLORS[1]} fill={COLORS[1] + '33'} strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="estimated_cost" name="Est. Cost ($)" stroke={COLORS[2]} strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          {/* Budget Gauge */}
          <Card title="Monthly Budget" span={1}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 240, gap: 16 }}>
              <div style={{ fontSize: 14, color: '#64748b', fontWeight: 600 }}>Budget Status</div>
              <div style={{ position: 'relative', width: 160, height: 16, background: '#e2e8f0', borderRadius: 8, overflow: 'hidden' }}>
                <div style={{
                  position: 'absolute', left: 0, top: 0, height: '100%',
                  width: Math.min(budgetPct, 100) + '%',
                  background: budgetColor, borderRadius: 8, transition: 'width .3s'
                }} />
              </div>
              <div style={{ fontSize: 32, fontWeight: 700, color: budgetColor }}>{budgetPct.toFixed(1)}%</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>
                ${budgetSpent.toFixed(4)} of ${budgetLimit.toFixed(2)} limit
              </div>
              <StatusBadge status={budget.status || 'ok'} />
            </div>
          </Card>

          {/* LLM Usage Card */}
          <Card title="LLM Usage" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12 }}>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Model</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#334155' }}>{llmUsage.model_label || llmUsage.model || '--'}</div>
              </div>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Conversations</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#3b82f6' }}>{fmt(llmUsage.conversations)}</div>
              </div>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Monthly Input Tokens</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#22c55e' }}>{fmt(llmUsage.monthly_input_tokens)}</div>
              </div>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Monthly Output Tokens</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#f97316' }}>{fmt(llmUsage.monthly_output_tokens)}</div>
              </div>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Hypothetical Cloud Cost</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: '#ef4444' }}>
                  {llmUsage.hypothetical_cloud_cost != null ? `$${llmUsage.hypothetical_cloud_cost.toFixed(4)}` : '--'}
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== TOKEN BREAKDOWN TAB ======================== */}
      {tab === 'token-breakdown' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* PieChart by role */}
          <Card title="Tokens by Role" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={byRole}
                  dataKey="tokens"
                  nameKey="role"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label={({ role, percent }) => `${role} (${(percent * 100).toFixed(1)}%)`}
                >
                  {byRole.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(v) => [v.toLocaleString(), 'Tokens']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Token distribution BarChart */}
          <Card title="Token Distribution by Role" span={1}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={byRole} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="role" type="category" width={80} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [v.toLocaleString(), 'Tokens']} />
                <Bar dataKey="tokens" name="Tokens" radius={[0, 4, 4, 0]}>
                  {byRole.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Messages per role */}
          <Card title="Messages per Role" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              {byRole.map((r, i) => (
                <div key={i} style={{ padding: 16, background: '#f8fafc', borderRadius: 8, textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, textTransform: 'capitalize' }}>{r.role}</div>
                  <div style={{ fontSize: 24, fontWeight: 700, color: COLORS[i % COLORS.length] }}>{fmt(r.messages)}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>messages</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{r.tokens != null ? r.tokens.toLocaleString() : '--'} tokens</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ======================== COST ANALYSIS TAB ======================== */}
      {tab === 'cost-analysis' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Component cost horizontal BarChart */}
          <Card title="Cost by Component" span={2}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={componentBreakdown.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `$${v.toFixed(4)}`} />
                <YAxis dataKey="component" type="category" width={160} tick={{ fontSize: 10 }} />
                <Tooltip formatter={(v) => [`$${v.toFixed(4)}`, 'Cost (USD)']} />
                <Bar dataKey="cost" name="Cost (USD)" radius={[0, 4, 4, 0]}>
                  {componentBreakdown.slice(0, 10).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Top 10 components by operations */}
          <Card title="Top Components by Operations" span={1}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Component</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Operations</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Cost (USD)</th>
                  </tr>
                </thead>
                <tbody>
                  {[...componentBreakdown]
                    .sort((a, b) => (b.operations || 0) - (a.operations || 0))
                    .slice(0, 10)
                    .map((c, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600 }}>{c.component}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(c.operations)}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>
                          {c.cost != null ? `$${c.cost.toFixed(4)}` : '--'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Model breakdown table */}
          <Card title="Model Inference Breakdown" span={1}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Disease</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Inferences</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg Confidence</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Cost (USD)</th>
                  </tr>
                </thead>
                <tbody>
                  {modelBreakdown.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{m.disease}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(m.inferences)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>
                        {m.avg_confidence != null ? (m.avg_confidence * 100).toFixed(1) + '%' : '--'}
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>
                        {m.cost != null ? `$${m.cost.toFixed(4)}` : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Rate cards table */}
          <Card title="Rate Cards" span={2}>
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Model</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Input per 1K tokens (USD)</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Output per 1K tokens (USD)</th>
                </tr>
              </thead>
              <tbody>
                {(definitions?.rate_cards || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{r.model}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {r.input_per_1k != null ? `$${r.input_per_1k.toFixed(4)}` : '--'}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {r.output_per_1k != null ? `$${r.output_per_1k.toFixed(4)}` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ======================== OPERATIONS TAB ======================== */}
      {tab === 'operations' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* by_component BarChart */}
          <Card title="Operations by Component" span={1}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={byComponentArr.slice(0, 12)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Operations" radius={[0, 4, 4, 0]}>
                  {byComponentArr.slice(0, 12).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* by_action BarChart */}
          <Card title="Operations by Action" span={1}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={byActionArr.slice(0, 12)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={130} tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="count" name="Operations" radius={[0, 4, 4, 0]}>
                  {byActionArr.slice(0, 12).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily operations trend LineChart */}
          <Card title="Daily Operations Trend" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={dailyTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="operations" name="Operations" stroke={COLORS[0]} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: 16 }}>
          {/* Metrics table */}
          <Card title="Metrics Reference">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 200 }}>Metric</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', width: 120 }}>Unit</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.metrics || []).map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{m.name}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{m.description}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{m.unit || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Operation rates table */}
          <Card title="Operation Rates">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Operation</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Rate (USD)</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.operation_rates || []).map((o, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{o.operation}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace', color: '#ef4444' }}>
                      {o.rate_usd != null ? `$${o.rate_usd.toFixed(6)}` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Budget tiers table */}
          <Card title="Budget Tiers">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Category</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b' }}>Monthly Limit (USD)</th>
                </tr>
              </thead>
              <tbody>
                {(definitions.budget_tiers || []).map((bt, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{bt.category}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace', color: '#3b82f6' }}>
                      {bt.monthly_limit_usd != null ? `$${bt.monthly_limit_usd.toFixed(2)}` : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Clinical relevance */}
          {definitions.clinical_relevance && (
            <Card title="Clinical Relevance">
              <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                {definitions.clinical_relevance}
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
