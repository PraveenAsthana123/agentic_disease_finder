import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ReferenceLine
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
  const colorMap = { converged: '#22c55e', converging: '#eab308', diverging: '#ef4444', complete: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function FederatedLearningDashboard() {
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
          axios.get(`${API_URL}/federated-learning/overview`),
          axios.get(`${API_URL}/federated-learning/breakdown`),
          axios.get(`${API_URL}/federated-learning/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Federated Learning data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'sites', label: 'Site Analysis' },
    { id: 'convergence', label: 'Convergence' },
    { id: 'privacy', label: 'Privacy' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const siteSummary = overview?.site_summary || []
  const roundHistory = overview?.round_history || []

  /* --- Site Analysis data --- */
  const siteDetails = breakdown?.site_details || []
  const seizureTypeDist = breakdown?.seizure_type_distribution || []
  const bandwidthUsage = breakdown?.bandwidth_usage || []

  /* --- Convergence data --- */
  const convergenceCurve = breakdown?.convergence_curve || []
  const aggregationComparison = breakdown?.aggregation_comparison || []
  const gradientNorms = breakdown?.gradient_norms || []

  /* --- Privacy data --- */
  const epsilonBudgetHistory = breakdown?.epsilon_budget_history || []
  const privacyAudit = breakdown?.privacy_audit || []
  const heterogeneityMetrics = breakdown?.heterogeneity_metrics || {}

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Federated Learning Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Privacy-preserving distributed model training across clinical sites with differential privacy guarantees
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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Global Model Accuracy" value={fmtPct(overview.global_model_accuracy)} color={
                overview.global_model_accuracy >= 0.9 ? '#22c55e' : overview.global_model_accuracy >= 0.7 ? '#eab308' : '#ef4444'
              } />
              <KPI label="Total Sites" value={fmt(overview.total_sites)} />
              <KPI label="Communication Rounds" value={fmt(overview.communication_rounds)} />
              <KPI label="Privacy Budget (\u03B5)" value={fmt(overview.privacy_budget_epsilon)} />
              <KPI label="Convergence Status" value={
                <StatusBadge status={overview.convergence_status || 'pending'} />
              } />
            </div>
          </Card>

          {/* Site Summary Table */}
          <Card title="Site Summary" span={2}>
            <div style={{ maxHeight: 320, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Site</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Patients</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Local Acc.</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Weight</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Last Sync</th>
                  </tr>
                </thead>
                <tbody>
                  {siteSummary.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{s.site_name}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(s.n_patients)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(s.local_accuracy)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(s.contribution_weight)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{s.last_sync}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Round History Line Chart */}
          <Card title="Global Accuracy Over Rounds">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={roundHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tick={{ fontSize: 11 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => fmtPct(v)} />
                <Line type="monotone" dataKey="global_accuracy" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Global Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== SITE ANALYSIS TAB ======================== */}
      {tab === 'sites' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Site Detail Table */}
          <Card title="Per-Site Model Performance" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Site</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Patients</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>EEG Records</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Accuracy</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Sensitivity</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Specificity</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>F1</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Weight Div.</th>
                  </tr>
                </thead>
                <tbody>
                  {siteDetails.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{s.name}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(s.n_patients)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(s.n_eeg_records)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(s.accuracy)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(s.sensitivity)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(s.specificity)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(s.f1)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(s.weight_divergence_from_global)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Seizure Type Distribution Bar Chart */}
          <Card title="Seizure Type Distribution by Site">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={seizureTypeDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="site" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="focal" fill={COLORS[0]} name="Focal" radius={[4, 4, 0, 0]} />
                <Bar dataKey="generalized" fill={COLORS[1]} name="Generalized" radius={[4, 4, 0, 0]} />
                <Bar dataKey="unknown" fill={COLORS[2]} name="Unknown" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Bandwidth Usage Bar Chart */}
          <Card title="Bandwidth Usage per Site">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={bandwidthUsage} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="site" width={70} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="bandwidth_mb" name="Bandwidth (MB)" radius={[0, 4, 4, 0]}>
                  {bandwidthUsage.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== CONVERGENCE TAB ======================== */}
      {tab === 'convergence' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Convergence Curve (dual Y-axis) */}
          <Card title="Convergence Curve: Loss & Accuracy Over Rounds" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={convergenceCurve}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tick={{ fontSize: 11 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis yAxisId="left" tick={{ fontSize: 11 }} label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fontSize: 11 }} label={{ value: 'Accuracy', angle: 90, position: 'insideRight', fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="global_loss" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Global Loss" />
                <Line yAxisId="right" type="monotone" dataKey="global_accuracy" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} name="Global Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Aggregation Strategy Comparison Table */}
          <Card title="Aggregation Strategy Comparison">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Strategy</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Accuracy</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Rounds</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Comm. Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {aggregationComparison.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{a.strategy}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmtPct(a.accuracy)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(a.rounds_to_converge)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{a.communication_cost}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Gradient Norms Bar Chart */}
          <Card title="Per-Site Gradient Norms & Clipping Rates">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={gradientNorms}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="site" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="gradient_norm" fill={COLORS[0]} name="Gradient Norm" radius={[4, 4, 0, 0]} />
                <Bar dataKey="clipping_rate" fill={COLORS[4]} name="Clipping Rate" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== PRIVACY TAB ======================== */}
      {tab === 'privacy' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Privacy KPIs */}
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Epsilon Spent (\u03B5)" value={fmt(overview?.epsilon_spent)} color={
                overview?.epsilon_spent <= 1 ? '#22c55e' : overview?.epsilon_spent <= 5 ? '#eab308' : '#ef4444'
              } />
              <KPI label="Delta (\u03B4)" value={overview?.delta != null ? overview.delta.toExponential(1) : '--'} />
              <KPI label="Noise Multiplier" value={fmt(overview?.noise_multiplier)} />
              <KPI label="Gradient Clipping Norm" value={fmt(overview?.gradient_clipping_norm)} />
            </div>
          </Card>

          {/* Per-Round Epsilon Budget Line Chart (cumulative) */}
          <Card title="Cumulative Epsilon Budget Over Rounds" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={epsilonBudgetHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="round" tick={{ fontSize: 11 }} label={{ value: 'Round', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'Cumulative \u03B5', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="cumulative_epsilon" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="Cumulative \u03B5" />
                {epsilonBudgetHistory.length > 0 && epsilonBudgetHistory[0].budget_limit != null && (
                  <ReferenceLine y={epsilonBudgetHistory[0].budget_limit} stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'Budget Limit', fill: '#ef4444', fontSize: 11 }} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Privacy Audit Table */}
          <Card title="Privacy Audit Log">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Round</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>\u03B5 Spent</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Cumulative</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Guarantee</th>
                  </tr>
                </thead>
                <tbody>
                  {privacyAudit.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.round)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.epsilon_spent)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(r.cumulative)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={r.guarantee_status || 'pending'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Data Heterogeneity Metrics */}
          <Card title="Data Heterogeneity Metrics">
            <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
              <tbody>
                <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>Non-IID Score</td>
                  <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{fmt(heterogeneityMetrics.non_iid_score)}</td>
                </tr>
                <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>Label Distribution Divergence</td>
                  <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{fmt(heterogeneityMetrics.label_distribution_divergence)}</td>
                </tr>
                {heterogeneityMetrics.feature_skew != null && (
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>Feature Skew</td>
                    <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{fmt(heterogeneityMetrics.feature_skew)}</td>
                  </tr>
                )}
                {heterogeneityMetrics.quantity_skew != null && (
                  <tr style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>Quantity Skew</td>
                    <td style={{ padding: '8px 12px', color: '#475569', fontFamily: 'monospace', fontSize: 12 }}>{fmt(heterogeneityMetrics.quantity_skew)}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <Card title={definitions.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(definitions.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
