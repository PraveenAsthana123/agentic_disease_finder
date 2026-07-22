import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const ROLE_COLORS = { author: '#3b82f6', reviewer: '#10b981', chair: '#f59e0b' }
const STATUS_COLORS = { built: '#10b981', planned: '#3b82f6', partial: '#f59e0b', scaffold: '#f59e0b', unknown: '#94a3b8' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AgentCouncilDashboard() {
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
          axios.get(`${API_URL}/api/agent-council/overview`),
          axios.get(`${API_URL}/api/agent-council/breakdown`),
          axios.get(`${API_URL}/api/agent-council/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Council of Agents data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Council of Agents data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Council of Agents data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const roleDist = overview.role_distribution || []
  const qualityTrend = overview.decision_quality_trend || []
  const reviewStatus = overview.review_status_breakdown || []
  const councilEvents = overview.council_events || []
  const roster = breakdown?.agent_roster || []
  const sessions = breakdown?.recent_sessions || []
  const votes = breakdown?.voting_history || []
  const assignments = breakdown?.review_assignments || []
  const definitions = defs?.definitions || []
  const complianceRefs = defs?.compliance_refs || []
  const remediation = defs?.remediation || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: active ? '1px solid #2563eb' : '1px solid #cbd5e1'
  })
  const roleBadge = (role) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${ROLE_COLORS[role] || '#94a3b8'}22`, color: ROLE_COLORS[role] || '#64748b'
  })
  const statusBadge = (st) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${STATUS_COLORS[st] || '#94a3b8'}22`, color: STATUS_COLORS[st] || '#64748b'
  })

  const kpiItems = [
    { label: 'Total Agents', value: s.total_agents, color: '#3b82f6' },
    { label: 'Authors', value: s.authors, color: '#3b82f6' },
    { label: 'Reviewers', value: s.reviewers, color: '#10b981' },
    { label: 'Chairs', value: s.chairs, color: '#f59e0b' },
    { label: 'Decisions Reviewed', value: s.decisions_reviewed, color: '#8b5cf6' },
    { label: 'Consensus Rate', value: s.consensus_rate != null ? `${s.consensus_rate}%` : '--', color: '#10b981' },
    { label: 'Override Rate', value: s.override_rate != null ? `${s.override_rate}%` : '--', color: '#ef4444' },
    { label: 'Coverage', value: s.coverage_pct != null ? `${s.coverage_pct}%` : '--', color: '#06b6d4' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 22, color: '#1e293b' }}>Council of Agents Dashboard</h2>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18, flexWrap: 'wrap' }}>
        {[
          ['overview', 'Overview'],
          ['roster', 'Agent Roster'],
          ['sessions', 'Council Sessions'],
          ['consensus', 'Consensus Metrics'],
          ['definitions', 'Definitions'],
        ].map(([id, label]) => (
          <div key={id} style={tabStyle(tab === id)} onClick={() => setTab(id)}>{label}</div>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI row */}
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

          {/* Role distribution + Review status charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Role Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={roleDist} dataKey="count" nameKey="role" cx="50%" cy="50%"
                    innerRadius={50} outerRadius={85}
                    label={({ role, count }) => `${role}: ${count}`}>
                    {roleDist.map((d, i) => (
                      <Cell key={i} fill={ROLE_COLORS[d.role] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Review Status Breakdown</h4>
              {reviewStatus.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={reviewStatus}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="status" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Reviews" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No review status data available</div>
              )}
            </div>
          </div>

          {/* Decision quality trend */}
          {qualityTrend.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Decision Quality Trend</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={qualityTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="decisions" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Decisions" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Council events */}
          {councilEvents.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Council Event Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={councilEvents}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {tab === 'roster' && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Agent Roster ({roster.length} agents)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>ID</th>
                  <th style={{ padding: '8px 10px' }}>Task</th>
                  <th style={{ padding: '8px 10px' }}>Module</th>
                  <th style={{ padding: '8px 10px' }}>Role</th>
                  <th style={{ padding: '8px 10px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {roster.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{a.id}</td>
                    <td style={{ padding: '8px 10px', color: '#1e293b', maxWidth: 300 }}>{a.task}</td>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.module}</td>
                    <td style={{ padding: '8px 10px' }}><span style={roleBadge(a.role)}>{a.role}</span></td>
                    <td style={{ padding: '8px 10px' }}><span style={statusBadge(a.status)}>{a.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'sessions' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Recent Council Sessions</h4>
            {sessions.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Date</th>
                    <th style={{ padding: '8px 10px' }}>Total Events</th>
                    <th style={{ padding: '8px 10px' }}>Components</th>
                    <th style={{ padding: '8px 10px' }}>Action Types</th>
                    <th style={{ padding: '8px 10px' }}>Participants</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{s.date}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.total_events)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.components)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.action_types)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(s.participants)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No session data available</div>
            )}
          </div>

          {/* Review assignments */}
          {assignments.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Review Assignments Matrix</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 10px' }}>Component</th>
                    <th style={{ padding: '8px 10px' }}>Review Count</th>
                    <th style={{ padding: '8px 10px' }}>Reviewers</th>
                  </tr>
                </thead>
                <tbody>
                  {assignments.map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{a.component}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.review_count)}</td>
                      <td style={{ padding: '8px 10px' }}>{fmt(a.reviewers)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {tab === 'consensus' && (
        <>
          {/* Consensus summary */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Consensus Summary</h4>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16 }}>
              <div>
                <span style={{ fontSize: 12, color: '#64748b' }}>Consensus Rate:</span>{' '}
                <strong style={{ color: '#10b981', fontSize: 18 }}>{s.consensus_rate != null ? `${s.consensus_rate}%` : '--'}</strong>
              </div>
              <div>
                <span style={{ fontSize: 12, color: '#64748b' }}>Override Rate:</span>{' '}
                <strong style={{ color: '#ef4444', fontSize: 18 }}>{s.override_rate != null ? `${s.override_rate}%` : '--'}</strong>
              </div>
              <div>
                <span style={{ fontSize: 12, color: '#64748b' }}>Total Decisions:</span>{' '}
                <strong style={{ color: '#3b82f6', fontSize: 18 }}>{fmt(s.decisions_reviewed)}</strong>
              </div>
            </div>
          </div>

          {/* Voting history */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Voting History</h4>
            {votes.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Source</th>
                      <th style={{ padding: '8px 10px' }}>Prediction / Finding</th>
                      <th style={{ padding: '8px 10px' }}>Agreement</th>
                      <th style={{ padding: '8px 10px' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {votes.map((v, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{v.patient_id}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={{
                            display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                            background: v.source === 'clinical_decision' ? '#3b82f622' : '#10b98122',
                            color: v.source === 'clinical_decision' ? '#3b82f6' : '#10b981'
                          }}>
                            {v.source === 'clinical_decision' ? 'Clinical' : 'Expert'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px' }}>{v.ai_prediction || v.finding || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={{
                            display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 600,
                            background: v.agreement === 'agree' ? '#10b98122' : '#ef444422',
                            color: v.agreement === 'agree' ? '#10b981' : '#ef4444'
                          }}>
                            {v.agreement || '--'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>{v.date || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', fontSize: 13, padding: 20 }}>No voting history available</div>
            )}
          </div>
        </>
      )}

      {tab === 'definitions' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Council Concepts</h4>
            {definitions.map((d, i) => (
              <div key={i} style={{ marginBottom: 10 }}>
                <strong style={{ color: '#1e293b' }}>{d.term}:</strong>{' '}
                <span style={{ color: '#475569', fontSize: 13 }}>{d.definition}</span>
              </div>
            ))}
          </div>

          {complianceRefs.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Compliance References</h4>
              {complianceRefs.map((ref, i) => (
                <div key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>- {ref}</div>
              ))}
            </div>
          )}

          {remediation.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Remediation Strategies</h4>
              {remediation.map((r, i) => (
                <div key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>- {r}</div>
              ))}
            </div>
          )}
        </>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: agent_tasks.json ({roster.length} agents), clinical.db (clinical_decisions, expert_reviews, hitl_reviews, transaction_log)
      </div>
    </div>
  )
}
