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

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#f97316', '#64748b']

export default function AIObservabilityDashboard() {
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
      axios.get(`${API}/ai-observability/overview`),
      axios.get(`${API}/ai-observability/breakdown`),
      axios.get(`${API}/ai-observability/definitions`),
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
    { id: 'activity', label: 'System Activity' },
    { id: 'cost', label: 'Cost Analytics' },
    { id: 'audit', label: 'Audit Trail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading AI Observability dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No AI observability data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Observability</h2>
        <Badge
          text={`${overview.kpis?.total_transactions || 0} transactions`}
          color="#3b82f6"
        />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Generated: {overview.generated_at}</span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'activity' && renderActivity()}
      {tab === 'cost' && renderCost()}
      {tab === 'audit' && renderAudit()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const kpis = [
      { label: 'Total Transactions', value: k.total_transactions, color: '#3b82f6' },
      { label: 'Components', value: k.total_components, color: '#8b5cf6' },
      { label: 'Actors', value: k.total_actors, color: '#06b6d4' },
      { label: 'Analyses', value: k.total_analyses, color: '#10b981' },
      { label: 'Conversations', value: k.total_conversations, color: '#f59e0b' },
      { label: 'Cost Records', value: k.total_cost_records, color: '#ec4899' },
      { label: 'Avg Confidence', value: k.avg_confidence != null ? (k.avg_confidence * 100).toFixed(1) + '%' : 'N/A', color: '#f97316' },
      { label: 'Total Cost', value: k.total_cost_usd != null ? '$' + k.total_cost_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '$0', color: '#1e293b' },
    ]

    const compDist = (overview.component_distribution || []).slice(0, 10)
    const actionDist = overview.action_distribution || []
    const actorData = overview.actor_activity || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Component Distribution (Top 10)" span={4}>
          {compDist.length > 0 && (
            <ResponsiveContainer width="100%" height={Math.max(200, compDist.length * 32)}>
              <BarChart data={compDist} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="component" tick={{ fontSize: 11 }} width={130} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {compDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Action Distribution" span={2}>
          {actionDist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={actionDist} dataKey="count" nameKey="action" cx="50%" cy="50%" outerRadius={100} label={({ action, count }) => `${action}: ${count}`}>
                  {actionDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Actor Activity" span={2}>
          {actorData.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={actorData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="actor" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Transactions" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {actorData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderActivity() {
    const dailyVol = overview.daily_transaction_volume || []
    const hourlyData = overview.hourly_heatmap || []
    const perComponent = breakdown?.per_component_actions || []
    const perActor = breakdown?.per_actor_components || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Daily Transaction Volume">
          {dailyVol.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={dailyVol} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" name="Transactions" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Hourly Activity Heatmap">
          {hourlyData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={hourlyData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" tick={{ fontSize: 11 }} label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5, fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [v, 'Transactions']} labelFormatter={(h) => `Hour ${h}:00`} />
                <Bar dataKey="count" name="Transactions" fill="#10b981" radius={[4, 4, 0, 0]}>
                  {hourlyData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title={`Per-Component Actions (${perComponent.length} components)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Component</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Top Actions</th>
                </tr>
              </thead>
              <tbody>
                {perComponent.map((pc, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{pc.component}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {(pc.actions || []).slice(0, 3).map((a, j) => (
                          <Badge key={j} text={`${a.action} (${a.count})`} color={COLORS[j % COLORS.length]} />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={`Per-Actor Components (${perActor.length} actors)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Actor</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Components Used</th>
                </tr>
              </thead>
              <tbody>
                {perActor.map((pa, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{pa.actor}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {(pa.components || []).map((c, j) => (
                          <Badge key={j} text={`${c.component} (${c.count})`} color={COLORS[j % COLORS.length]} />
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderCost() {
    const costTimeline = breakdown?.cost_timeline || []
    const costByCategory = overview.cost_by_category || []
    const costByService = breakdown?.cost_by_service || []
    const hasTokens = costTimeline.some(c => c.tokens_in != null || c.tokens_out != null)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Cost Over Time">
          {costTimeline.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={costTimeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? '$' + v.toFixed(2) : v} />
                <Line type="monotone" dataKey="total_cost" name="Total Cost ($)" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6' }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Cost by Category">
          {costByCategory.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={costByCategory} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v, name) => name === 'Total Cost' ? '$' + (typeof v === 'number' ? v.toFixed(2) : v) : v} />
                <Bar dataKey="total_cost" name="Total Cost" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {costByCategory.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title={`Cost by Service (${costByService.length} services)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Model / Service</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Total Cost</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Requests</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Avg Cost</th>
                </tr>
              </thead>
              <tbody>
                {costByService.map((cs, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{cs.model_or_service}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      ${typeof cs.total_cost === 'number' ? cs.total_cost.toFixed(2) : cs.total_cost}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>{cs.requests}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      ${typeof cs.avg_cost === 'number' ? cs.avg_cost.toFixed(4) : cs.avg_cost}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {hasTokens && (
          <Card title="Token Usage Over Time">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={costTimeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="tokens_in" name="Tokens In" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
                <Line type="monotone" dataKey="tokens_out" name="Tokens Out" stroke="#f59e0b" strokeWidth={2} dot={{ fill: '#f59e0b' }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>
    )
  }

  function renderAudit() {
    const analysisSummary = breakdown?.analysis_summary || []
    const patientProfiles = breakdown?.patient_transaction_profiles || []
    const errorActions = breakdown?.error_actions || []
    const roleDistribution = overview.conversation_role_distribution || []
    const convTimeline = breakdown?.conversation_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Analysis Summary (${analysisSummary.length} analyses)`}>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Disease</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Signal Quality</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {analysisSummary.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{a.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={a.disease} color="#8b5cf6" />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {a.confidence != null ? (a.confidence * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={a.signal_quality || 'N/A'} color={a.signal_quality === 'good' ? '#10b981' : a.signal_quality === 'moderate' ? '#f59e0b' : '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{a.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={`Patient Transaction Profiles (${patientProfiles.length} patients)`}>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Transactions</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Components</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Latest Activity</th>
                </tr>
              </thead>
              <tbody>
                {patientProfiles.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={p.transaction_count} color={p.transaction_count > 10 ? '#3b82f6' : '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                      {(p.components || []).slice(0, 4).join(', ')}
                      {(p.components || []).length > 4 && ` +${p.components.length - 4} more`}
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#64748b' }}>{p.latest_ts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {errorActions.length > 0 && (
          <Card title="Error Actions Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Action</th>
                    <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Count</th>
                    <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {errorActions.map((ea, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{ea.action}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>{ea.count}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {typeof ea.pct === 'number' ? ea.pct.toFixed(1) + '%' : ea.pct}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        <Card title="Conversation Role Distribution" span={1}>
          {roleDistribution.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={roleDistribution} dataKey="count" nameKey="role" cx="50%" cy="50%" outerRadius={100} label={({ role, count }) => `${role}: ${count}`}>
                  {roleDistribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Conversation Timeline">
          {convTimeline.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={convTimeline} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" name="Conversations" stroke="#ec4899" strokeWidth={2} dot={{ fill: '#ec4899' }} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, si) => (
          <Card key={si} title={sec.title}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
