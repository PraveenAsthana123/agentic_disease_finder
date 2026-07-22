import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function RoutingDashboard() {
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
          axios.get(`${API_URL}/api/routing/overview`),
          axios.get(`${API_URL}/api/routing/breakdown`),
          axios.get(`${API_URL}/api/routing/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load routing data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading routing data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Routing data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const compData = (overview.component_distribution || []).slice(0, 12)
  const actionData = (overview.action_distribution || []).slice(0, 10)
  const actorData = overview.actor_distribution || []
  const decisionData = overview.decision_outcomes || []
  const matrixData = (overview.routing_matrix || []).slice(0, 15)
  const dailyData = overview.daily_volume || []
  const hourlyData = overview.hourly_pattern || []
  const convRouting = overview.conversation_routing || []
  const crossTab = (breakdown?.cross_tab || []).slice(0, 30)
  const recentEvents = breakdown?.recent_events || []
  const patientRouting = breakdown?.patient_routing || []
  const decisionDetail = breakdown?.decision_detail || []
  const componentStats = breakdown?.component_stats || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#64748b'
  })
  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  const renderOverview = () => (
    <>
      {/* Automation banner */}
      <div style={{ ...cardStyle, background: '#3b82f611', border: '2px solid #3b82f644' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Automation Rate</div>
            <div style={{ fontSize: 36, fontWeight: 800, color: '#3b82f6' }}>{s.automation_rate_pct}%</div>
          </div>
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Auto-Routed</span><br/><strong>{fmt(s.auto_routed)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Human-Routed</span><br/><strong>{fmt(s.human_routed)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Agreement Rate</span><br/><strong>{s.agreement_rate_pct}%</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Components</span><br/><strong>{fmt(s.distinct_components)}</strong></div>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {[
          { label: 'Routed Events', value: s.total_routed_events, color: '#3b82f6' },
          { label: 'Clinical Decisions', value: s.total_clinical_decisions, color: '#10b981' },
          { label: 'Findings', value: s.total_component_findings, color: '#8b5cf6' },
          { label: 'Conversations', value: s.total_conversations, color: '#f59e0b' },
          { label: 'Actors', value: s.distinct_actors, color: '#06b6d4' },
          { label: 'Action Types', value: s.distinct_actions, color: '#ec4899' },
          { label: 'Patients', value: s.unique_patients, color: '#84cc16' },
        ].map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 11, color: '#64748b' }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
          </div>
        ))}
      </div>

      {/* Component distribution bar chart */}
      {compData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Component Routing Distribution</h4>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compData} layout="vertical" margin={{ left: 110 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="component" tick={{ fontSize: 11 }} width={100} />
              <Tooltip formatter={(v) => [v.toLocaleString(), 'Routed']} />
              <Bar dataKey="routed" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Action distribution + actor pie */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        {actionData.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Action Type Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={actionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="action" tick={{ fontSize: 10, angle: -30 }} height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {actorData.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Actor Workload</h4>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={actorData} dataKey="routed" nameKey="actor" cx="50%" cy="50%"
                     outerRadius={80} label={({ actor, pct }) => `${actor} ${pct}%`}>
                  {actorData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Daily volume trend */}
      {dailyData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Daily Routing Volume</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="routed" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Hourly pattern */}
      {hourlyData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Hourly Routing Pattern (UTC)</h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="routed" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Conversation role routing */}
      {convRouting.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Conversation Role Routing</h4>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {convRouting.map((r, i) => (
              <div key={i} style={kpiStyle(COLORS[i % COLORS.length])}>
                <div style={{ fontSize: 11, color: '#64748b' }}>{r.role}</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: COLORS[i % COLORS.length] }}>{fmt(r.turns)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )

  const renderRoutes = () => (
    <>
      {/* Routing matrix */}
      {matrixData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Top Routing Paths (Component x Action)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Component</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Action</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Count</th>
                  <th style={{ textAlign: 'left', padding: 6, width: '40%' }}>Volume</th>
                </tr>
              </thead>
              <tbody>
                {matrixData.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6, fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: 6 }}>{r.action}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.count)}</td>
                    <td style={{ padding: 6 }}>
                      <div style={{ background: '#e2e8f0', borderRadius: 4, overflow: 'hidden', height: 14 }}>
                        <div style={{ background: COLORS[i % COLORS.length], height: '100%', width: `${Math.min(r.count / (matrixData[0]?.count || 1) * 100, 100)}%`, borderRadius: 4 }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Cross-tab */}
      {crossTab.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Component x Actor x Action Cross-Tab</h4>
          <div style={{ overflowX: 'auto', maxHeight: 350, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Component</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Actor</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Action</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {crossTab.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: 6, fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: 6 }}>{r.actor}</td>
                    <td style={{ padding: 6 }}>{r.action}</td>
                    <td style={{ padding: 6, textAlign: 'right', fontWeight: 600 }}>{fmt(r.count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Component stats */}
      {componentStats.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Component Routing Stats</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Component</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Total</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Patients</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Actors</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Actions</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Last Event</th>
                </tr>
              </thead>
              <tbody>
                {componentStats.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6, fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.total)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.patients)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.actors)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.actions)}</td>
                    <td style={{ padding: 6, fontSize: 10, color: '#64748b' }}>{r.last_event || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )

  const renderDecisions = () => (
    <>
      {/* Decision outcomes */}
      {decisionData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Decision Routing Outcomes</h4>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
            {decisionData.map((d, i) => (
              <div key={i} style={{ ...kpiStyle(COLORS[i % COLORS.length]), minWidth: 160 }}>
                <div style={{ fontSize: 10, color: '#64748b' }}>Agreement: {d.agreement}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: COLORS[i % COLORS.length] }}>{d.final_decision}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{d.count} decision{d.count !== 1 ? 's' : ''}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Decision detail table */}
      {decisionDetail.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Clinical Decision Detail</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>AI Prediction</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Confidence</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Agreement</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Final</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Reviewer</th>
                  <th style={{ textAlign: 'left', padding: 6 }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {decisionDetail.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6 }}>{r.patient_id}</td>
                    <td style={{ padding: 6, fontWeight: 600 }}>{r.ai_prediction}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{r.ai_confidence != null ? `${(r.ai_confidence * 100).toFixed(0)}%` : '--'}</td>
                    <td style={{ padding: 6 }}>
                      <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600,
                        background: r.agreement === 'Yes' ? '#dcfce7' : r.agreement === 'No' ? '#fef2f2' : '#f1f5f9',
                        color: r.agreement === 'Yes' ? '#166534' : r.agreement === 'No' ? '#991b1b' : '#64748b'
                      }}>{r.agreement || 'pending'}</span>
                    </td>
                    <td style={{ padding: 6 }}>{r.final_decision || '--'}</td>
                    <td style={{ padding: 6, fontSize: 11 }}>{r.reviewer || '--'}</td>
                    <td style={{ padding: 6, fontSize: 10, color: '#64748b' }}>{r.created_at || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Patient routing */}
      {patientRouting.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Patient Routing Summary</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>Patient</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Events</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Components</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Actions</th>
                  <th style={{ textAlign: 'right', padding: 6 }}>Actors</th>
                </tr>
              </thead>
              <tbody>
                {patientRouting.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 6, fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.events)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.components)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.actions)}</td>
                    <td style={{ padding: 6, textAlign: 'right' }}>{fmt(r.actors)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent events */}
      {recentEvents.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', fontSize: 14, color: '#1e293b' }}>Recent Routing Events</h4>
          <div style={{ overflowX: 'auto', maxHeight: 350, overflowY: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                  <th style={{ textAlign: 'left', padding: 5 }}>ID</th>
                  <th style={{ textAlign: 'left', padding: 5 }}>Patient</th>
                  <th style={{ textAlign: 'left', padding: 5 }}>Component</th>
                  <th style={{ textAlign: 'left', padding: 5 }}>Action</th>
                  <th style={{ textAlign: 'left', padding: 5 }}>Actor</th>
                  <th style={{ textAlign: 'left', padding: 5 }}>Time</th>
                </tr>
              </thead>
              <tbody>
                {recentEvents.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 5, color: '#64748b' }}>{r.id}</td>
                    <td style={{ padding: 5 }}>{r.patient_id || '--'}</td>
                    <td style={{ padding: 5, fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: 5 }}>{r.action}</td>
                    <td style={{ padding: 5 }}>{r.actor}</td>
                    <td style={{ padding: 5, fontSize: 10, color: '#64748b' }}>{r.ts_utc || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )

  const renderDefinitions = () => (
    <div style={cardStyle}>
      <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#1e293b' }}>Metric Definitions</h4>
      {(defs?.metrics || []).map((m, i) => (
        <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < (defs?.metrics?.length || 0) - 1 ? '1px solid #f1f5f9' : 'none' }}>
          <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{m.name}</div>
          <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{m.definition}</div>
          <div style={{ fontSize: 10, color: '#94a3b8' }}>Source: {m.source}</div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: '0 0 6px', fontSize: 20, color: '#1e293b' }}>Routing Dashboard</h2>
        <p style={{ margin: 0, fontSize: 12, color: '#64748b' }}>
          Decision routing across {s.distinct_components} components, {s.distinct_actors} actors, {s.distinct_actions} action types
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {[
          ['overview', 'Overview'],
          ['routes', 'Routes & Stats'],
          ['decisions', 'Decisions & Events'],
          ['definitions', 'Definitions'],
        ].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={tabStyle(tab === id)}>{label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'routes' && renderRoutes()}
      {tab === 'decisions' && renderDecisions()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
