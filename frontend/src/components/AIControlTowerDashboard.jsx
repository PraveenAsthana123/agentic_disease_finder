import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = '/api'

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

const STATUS_COLORS = {
  built: '#10b981', scaffold: '#f59e0b', planned: '#3b82f6',
  PASS: '#10b981', '200': '#10b981', ok: '#10b981',
  FAIL: '#ef4444', SEVERE: '#ef4444'
}

const CHART_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#6366f1']

export default function AIControlTowerDashboard() {
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
      axios.get(`${API}/ai-control-tower/overview`),
      axios.get(`${API}/ai-control-tower/breakdown`),
      axios.get(`${API}/ai-control-tower/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Components' },
    { id: 'activity', label: 'Activity Log' },
    { id: 'oversight', label: 'Oversight & Decisions' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI Control Tower dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No control tower data available.</div>

  const ai = overview.ai_components || {}
  const readiness = ai.total > 0 ? ((ai.built / ai.total) * 100).toFixed(1) : '0'
  const healthStatus = overview.system_health?.backend_http === '200' ? 'Healthy' : 'Degraded'
  const healthColor = healthStatus === 'Healthy' ? '#10b981' : '#ef4444'

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Control Tower</h2>
        <Badge text={`System: ${healthStatus}`} color={healthColor} />
        <Badge text={`Drift: ${overview.drift_status?.verdict || 'N/A'}`} color={overview.drift_status?.verdict?.includes('SEVERE') ? '#ef4444' : '#10b981'} />
        <Badge text={`DQ: ${overview.data_quality?.ai_readiness_grade || 'N/A'}`} color='#3b82f6' />
      </div>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b', fontWeight: tab === t.id ? 600 : 400,
            cursor: 'pointer', fontSize: 13, marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── OVERVIEW TAB ─────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI row */}
          <Card><KPI label="Total Transactions" value={overview.total_transactions} /></Card>
          <Card><KPI label="AI Components Built" value={`${ai.built}/${ai.total - ai.not_pulled}`} sub={`${readiness}% of applicable`} /></Card>
          <Card><KPI label="Total Analyses" value={overview.analyses?.total} sub={`Avg conf: ${overview.analyses?.avg_confidence}`} /></Card>
          <Card><KPI label="Total Cost" value={`$${overview.cost_summary?.total_cost_usd}`} sub={`${overview.cost_summary?.total_records} records`} /></Card>
          <Card><KPI label="HITL Reviews" value={overview.oversight?.hitl_reviews} color="#8b5cf6" /></Card>
          <Card><KPI label="Clinical Decisions" value={overview.oversight?.clinical_decisions} color="#3b82f6" /></Card>
          <Card><KPI label="Error Rate" value={`${(overview.error_action_rate * 100).toFixed(2)}%`} color={overview.error_action_rate > 0.05 ? '#ef4444' : '#10b981'} /></Card>
          <Card><KPI label="Actors" value={overview.actor_activity?.length} sub="distinct system actors" /></Card>

          {/* Component activity bar chart */}
          <Card title="Component Activity (Top 10)" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={overview.component_activity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="cnt" fill="#3b82f6" name="Transactions" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Action distribution pie */}
          <Card title="Action Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={(overview.action_distribution || []).map(r => ({ name: r.action, value: r.cnt }))}
                  cx="50%" cy="50%" outerRadius={90} dataKey="value" label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={{ strokeWidth: 1 }}>
                  {(overview.action_distribution || []).map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily volume line chart */}
          <Card title="Daily Transaction Volume" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={overview.daily_volume || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="cnt" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Transactions" />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* Cost by category */}
          <Card title="Cost by Category" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(overview.cost_summary?.by_category || []).map(r => ({ ...r, total_cost: Math.round(r.total_cost * 100) / 100 }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="total_cost" fill="#8b5cf6" name="Cost (USD)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* System health panel */}
          <Card title="System Health" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(overview.system_health || {}).filter(([k]) => k !== 'timestamp').map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <span style={{ fontSize: 12, color: '#64748b' }}>{k.replace(/_/g, ' ')}</span>
                  <Badge text={String(v)} color={STATUS_COLORS[String(v)] || '#64748b'} />
                </div>
              ))}
            </div>
          </Card>

          {/* Actor activity */}
          <Card title="Actor Activity" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.actor_activity || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="actor" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="cnt" fill="#10b981" name="Transactions" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── COMPONENTS TAB ───────────────────────────────────────── */}
      {tab === 'components' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Component status matrix */}
          <Card title="Component Status Matrix" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Component</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Status</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.component_status_matrix || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 12px', fontWeight: 500 }}>{r.type}</td>
                      <td style={{ padding: '6px 12px' }}><Badge text={r.status} color={STATUS_COLORS[r.status] || '#64748b'} /></td>
                      <td style={{ padding: '6px 12px', color: '#64748b' }}>{r.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Component-action map */}
          <Card title="Per-Component Action Breakdown" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              {Object.entries(breakdown.component_action_map || {}).map(([comp, acts]) => (
                <div key={comp} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{comp}</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {acts.map((a, i) => (
                      <span key={i} style={{ fontSize: 11, padding: '2px 8px', background: '#f1f5f9', borderRadius: 4, color: '#475569' }}>
                        {a.action}: {a.count}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Hourly heatmap */}
          <Card title="Hourly Activity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.hourly_heatmap || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" tick={{ fontSize: 10 }} label={{ value: 'Hour of day', position: 'insideBottom', offset: -5, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="cnt" fill="#06b6d4" name="Transactions" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Cost by service table */}
          <Card title="Cost by Service" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Service</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Category</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Cost (USD)</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Requests</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Tokens In</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Tokens Out</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.cost_by_service || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 10px' }}>{r.model_or_service}</td>
                      <td style={{ padding: '5px 10px' }}><Badge text={r.category} color="#6366f1" /></td>
                      <td style={{ padding: '5px 10px', textAlign: 'right', fontWeight: 600 }}>${(r.total_cost || 0).toFixed(2)}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right' }}>{r.total_requests}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right' }}>{r.total_tokens_in}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right' }}>{r.total_tokens_out}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── ACTIVITY LOG TAB ─────────────────────────────────────── */}
      {tab === 'activity' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Recent transactions */}
          <Card title="Recent Transactions (Last 50)">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>ID</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Component</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Action</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Actor</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Detail</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_transactions || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{r.id}</td>
                      <td style={{ padding: '5px 8px' }}>{r.patient_id || '—'}</td>
                      <td style={{ padding: '5px 8px' }}><Badge text={r.component} color="#3b82f6" /></td>
                      <td style={{ padding: '5px 8px' }}><Badge text={r.action} color={r.action === 'blocked' || r.action === 'error' ? '#ef4444' : '#10b981'} /></td>
                      <td style={{ padding: '5px 8px' }}>{r.actor}</td>
                      <td style={{ padding: '5px 8px', color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.detail || '—'}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11, color: '#94a3b8' }}>{r.ts_local}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Patient profiles */}
          <Card title="Patient Transaction Profiles">
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient ID</th>
                    <th style={{ padding: '6px 10px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Transactions</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Components</th>
                    <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_profiles || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 10px', fontWeight: 500 }}>{r.patient_id}</td>
                      <td style={{ padding: '5px 10px', textAlign: 'right', fontWeight: 600 }}>{r.tx_count}</td>
                      <td style={{ padding: '5px 10px', fontSize: 11, color: '#64748b' }}>{r.components}</td>
                      <td style={{ padding: '5px 10px', fontSize: 11, color: '#64748b' }}>{r.actions}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Cost timeline */}
          <Card title="Cost Timeline">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={breakdown.cost_timeline || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cost_date" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="daily_cost" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} name="Daily Cost (USD)" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── OVERSIGHT & DECISIONS TAB ────────────────────────────── */}
      {tab === 'oversight' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* HITL Reviews */}
          <Card title="Human-in-the-Loop Reviews">
            {(breakdown.hitl_reviews || []).length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No HITL reviews recorded.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                {breakdown.hitl_reviews.map((r, i) => (
                  <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                      <Badge text={`Patient: ${r.patient_id || 'N/A'}`} color="#3b82f6" />
                      <Badge text={`Analysis #${r.analysis_id || 'N/A'}`} color="#8b5cf6" />
                      <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>{r.created_at}</span>
                    </div>
                    {r.fields && typeof r.fields === 'object' && Object.entries(r.fields).map(([k, v]) => (
                      <div key={k} style={{ fontSize: 12, color: '#475569', marginLeft: 8 }}>
                        <strong>{k}:</strong> {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </Card>

          {/* Clinical Decisions */}
          <Card title="Clinical Decisions">
            {(breakdown.clinical_decisions || []).length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No clinical decisions recorded.</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>AI Prediction</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right', borderBottom: '2px solid #e2e8f0' }}>Confidence</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Agreement</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Final Decision</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Reviewer</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.clinical_decisions.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 8px' }}>{r.patient_id}</td>
                      <td style={{ padding: '5px 8px' }}>{r.ai_prediction}</td>
                      <td style={{ padding: '5px 8px', textAlign: 'right' }}>{r.ai_confidence != null ? r.ai_confidence.toFixed(2) : '—'}</td>
                      <td style={{ padding: '5px 8px' }}><Badge text={r.neurologist_agreement || 'N/A'} color={r.neurologist_agreement === 'agree' ? '#10b981' : '#f59e0b'} /></td>
                      <td style={{ padding: '5px 8px', fontWeight: 600 }}>{r.final_decision}</td>
                      <td style={{ padding: '5px 8px' }}>{r.reviewer}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {/* Feedback */}
          <Card title="Clinician Feedback">
            {(breakdown.feedback || []).length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No feedback entries.</div>
            ) : (
              breakdown.feedback.map((r, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <Badge text={`Patient: ${r.patient_id || 'N/A'}`} color="#3b82f6" />
                    <Badge text={`Role: ${r.role || 'N/A'}`} color="#8b5cf6" />
                    <Badge text={`Rating: ${r.rating != null ? r.rating : 'N/A'}`} color={r.rating >= 4 ? '#10b981' : r.rating >= 2 ? '#f59e0b' : '#ef4444'} />
                    <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>{r.created_at}</span>
                  </div>
                  {r.correction && <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}><strong>Correction:</strong> {r.correction}</div>}
                  {r.reason && <div style={{ fontSize: 12, color: '#475569' }}><strong>Reason:</strong> {r.reason}</div>}
                </div>
              ))
            )}
          </Card>

          {/* Component Findings */}
          <Card title="Component Findings">
            {(breakdown.component_findings || []).length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 13 }}>No component findings.</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Component</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Doctor</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Agrees with AI</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0' }}>Finding</th>
                  </tr>
                </thead>
                <tbody>
                  {breakdown.component_findings.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '5px 8px' }}>{r.patient_id}</td>
                      <td style={{ padding: '5px 8px' }}>{r.component}</td>
                      <td style={{ padding: '5px 8px' }}>{r.doctor}</td>
                      <td style={{ padding: '5px 8px' }}><Badge text={r.agree_with_ai || 'N/A'} color={r.agree_with_ai === 'yes' ? '#10b981' : '#f59e0b'} /></td>
                      <td style={{ padding: '5px 8px', color: '#64748b' }}>{r.doctor_finding}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          {/* Actor-component map */}
          <Card title="Actor-Component Mapping">
            {Object.entries(breakdown.actor_component_map || {}).map(([actor, comps]) => (
              <div key={actor} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{actor}</div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {comps.slice(0, 8).map((c, i) => (
                    <span key={i} style={{ fontSize: 11, padding: '2px 8px', background: '#f1f5f9', borderRadius: 4, color: '#475569' }}>
                      {c.component}: {c.count}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </Card>
        </div>
      )}

      {/* ── DEFINITIONS TAB ──────────────────────────────────────── */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {[
            { key: 'control_tower_concept', title: 'AI Control Tower Concept' },
            { key: 'system_components', title: 'System Components' },
            { key: 'metrics_and_kpis', title: 'Metrics & KPIs' },
            { key: 'clinical_relevance', title: 'Clinical Relevance' },
            { key: 'remediation_strategies', title: 'Remediation Strategies' },
          ].map(section => (
            <Card key={section.key} title={section.title}>
              {(definitions[section.key] || []).map((item, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>
                    {item.name || item.standard || item.trigger || item.level}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>
                    {item.description || item.strategy || item.action}
                  </div>
                </div>
              ))}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
