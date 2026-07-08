import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const PRIORITY_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#10b981' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'cost_breakdown', label: 'Cost Breakdown' },
  { id: 'patient_roi', label: 'Patient ROI' },
  { id: 'optimization', label: 'Optimization' },
  { id: 'definitions', label: 'Definitions' },
]

const fmt = (v) => typeof v === 'number' ? `$${v.toFixed(2)}` : v
const fmtPct = (v) => typeof v === 'number' ? `${v.toFixed(1)}%` : v

export default function AIROIDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/ai-roi/overview`),
      axios.get(`${API_URL}/api/ai-roi/breakdown`),
      axios.get(`${API_URL}/api/ai-roi/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI ROI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No AI ROI data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>AI ROI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Investment tracking, cost analysis, return on investment metrics, and optimization recommendations for AI-powered clinical analytics
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'cost_breakdown' && <CostBreakdownTab breakdown={breakdown} />}
      {tab === 'patient_roi' && <PatientROITab breakdown={breakdown} />}
      {tab === 'optimization' && <OptimizationTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const totalInvestment = overview.total_investment
  const roiPercent = overview.roi_percent
  const analysesCompleted = overview.analyses_completed
  const timeSaved = overview.time_saved
  const avgCostPerAnalysis = overview.avg_cost_per_analysis
  const estimatedValue = overview.estimated_value
  const costByCategory = overview.cost_by_category || []
  const costByModel = overview.cost_by_model || []
  const investmentTrend = overview.investment_trend || []
  const valueDrivers = overview.value_drivers || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
      <Card><KPI label="Total Investment" value={fmt(totalInvestment)} color={COLORS[0]} /></Card>
      <Card><KPI label="ROI %" value={fmtPct(roiPercent)} color={COLORS[3]} /></Card>
      <Card><KPI label="Analyses Completed" value={analysesCompleted} color={COLORS[4]} /></Card>
      <Card><KPI label="Time Saved" value={timeSaved} color={COLORS[6]} /></Card>
      <Card><KPI label="Avg Cost/Analysis" value={fmt(avgCostPerAnalysis)} color={COLORS[2]} /></Card>
      <Card><KPI label="Estimated Value" value={fmt(estimatedValue)} color={COLORS[3]} /></Card>

      {/* Cost by Category Pie */}
      <Card title="Cost by Category" span={3}>
        {costByCategory.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={costByCategory} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100}
                label={({ name, value }) => `${name} ($${value.toFixed(2)})`}>
                {costByCategory.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`} />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No category data</div>}
      </Card>

      {/* Cost by Model Bar */}
      <Card title="Cost by Model" span={3}>
        {costByModel.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={costByModel}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {costByModel.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model cost data</div>}
      </Card>

      {/* Investment Trend Line */}
      <Card title="Investment Trend" span={6}>
        {investmentTrend.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={investmentTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`} />
              <Line type="monotone" dataKey="cost" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Cost" />
              <Line type="monotone" dataKey="cumulative" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Cumulative" />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No trend data</div>}
      </Card>

      {/* Value Drivers Table */}
      <Card title={`Value Drivers (${valueDrivers.length})`} span={6}>
        {valueDrivers.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Driver', 'Impact', 'Value', 'Description'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {valueDrivers.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.name}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={d.impact} color={d.impact === 'high' ? '#10b981' : d.impact === 'medium' ? '#f59e0b' : '#94a3b8'} /></td>
                    <td style={{ padding: '6px 10px' }}>{fmt(d.value)}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{d.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No value driver data</div>}
      </Card>
    </div>
  )
}

function CostBreakdownTab({ breakdown }) {
  const monthlyCosts = breakdown?.monthly_costs || []
  const topCostComponents = breakdown?.top_cost_components || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Monthly Costs Table */}
      <Card title={`Monthly Costs (${monthlyCosts.length})`}>
        {monthlyCosts.length > 0 ? (
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
                <tr style={{ background: '#f8fafc' }}>
                  {['Month', 'Total Cost', 'Analyses', 'Avg Cost', 'Cumulative'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Month' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {monthlyCosts.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{m.month}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(m.total_cost)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{m.analyses}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(m.avg_cost)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{fmt(m.cumulative)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No monthly cost data</div>}
      </Card>

      {/* Top Cost Components Bar Chart */}
      <Card title="Top Cost Components">
        {topCostComponents.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topCostComponents}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => `$${Number(v).toFixed(2)}`} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {topCostComponents.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No cost component data</div>}
      </Card>
    </div>
  )
}

function PatientROITab({ breakdown }) {
  const patientROI = breakdown?.patient_level_roi || []
  const [search, setSearch] = useState('')

  const filtered = search
    ? patientROI.filter(p =>
        (p.patient_id || '').toLowerCase().includes(search.toLowerCase()) ||
        (p.patient_name || '').toLowerCase().includes(search.toLowerCase())
      )
    : patientROI

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Patient-Level ROI (${filtered.length})`}>
        <div style={{ marginBottom: 12 }}>
          <input
            type="text"
            placeholder="Search by patient ID or name..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0',
              fontSize: 13, outline: 'none', boxSizing: 'border-box'
            }}
          />
        </div>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Name', 'Analyses', 'Total Cost', 'Estimated Value', 'ROI'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Patient ID' || h === 'Name' ? 'left' : 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, i) => {
                const roiValue = typeof p.roi === 'number' ? p.roi : parseFloat(p.roi) || 0
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 500 }}>{p.patient_name}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.analyses}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(p.total_cost)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{fmt(p.estimated_value)}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 700, color: roiValue >= 0 ? '#10b981' : '#ef4444' }}>
                      {fmtPct(roiValue)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function OptimizationTab({ breakdown }) {
  const recommendations = breakdown?.cost_optimization || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {recommendations.length > 0 ? recommendations.map((rec, i) => (
        <Card key={i} title={rec.title}>
          <div style={{ marginBottom: 8 }}>
            <Badge text={rec.priority} color={PRIORITY_COLORS[rec.priority] || '#6b7280'} />
          </div>
          <p style={{ fontSize: 13, color: '#475569', margin: '0 0 8px', lineHeight: 1.5 }}>{rec.description}</p>
          {rec.estimated_savings && (
            <div style={{ fontSize: 12, color: '#10b981', fontWeight: 600 }}>
              Estimated Savings: {fmt(rec.estimated_savings)}
            </div>
          )}
          {rec.implementation && (
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
              Implementation: {rec.implementation}
            </div>
          )}
        </Card>
      )) : (
        <Card span={2}>
          <div style={{ color: '#94a3b8', fontSize: 13 }}>No optimization recommendations available</div>
        </Card>
      )}
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const metrics = definitions.metrics || []
  const methodology = definitions.methodology || ''
  const assumptions = definitions.assumptions || []
  const glossary = definitions.glossary || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Metrics Table */}
      <Card title={`Metrics (${metrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Methodology */}
      <Card title="Methodology">
        <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{methodology}</p>
      </Card>

      {/* Assumptions */}
      <Card title={`Assumptions (${assumptions.length})`}>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {assumptions.map((a, i) => (
            <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{a}</li>
          ))}
        </ul>
      </Card>

      {/* Glossary Table */}
      <Card title={`Glossary (${glossary.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {glossary.map((g, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
