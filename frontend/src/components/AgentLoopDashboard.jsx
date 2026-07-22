import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const DRIFT_COLORS = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function pct(v) {
  return v != null ? `${v}%` : '--'
}

export default function AgentLoopDashboard() {
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
          axios.get(`${API_URL}/api/agent-loop/overview`),
          axios.get(`${API_URL}/api/agent-loop/breakdown`),
          axios.get(`${API_URL}/api/agent-loop/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load agent loop data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading agent loop data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Agent loop data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const daily = overview.daily_trend || []
  const compData = (overview.actions_by_component || []).slice(0, 10)
  const actionData = (overview.actions_by_type || []).slice(0, 10)
  const ratingData = overview.rating_distribution || []
  const components = breakdown?.components || []
  const corrections = breakdown?.recent_corrections || []
  const disagreements = breakdown?.decision_disagreements || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#64748b'
  })

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 120
  })

  const driftColor = DRIFT_COLORS[s.drift_severity] || '#64748b'

  const renderOverview = () => (
    <>
      {/* Drift score banner */}
      <div style={{ ...cardStyle, background: `${driftColor}11`, border: `2px solid ${driftColor}44` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Goal Drift Score</div>
            <div style={{ fontSize: 36, fontWeight: 800, color: driftColor }}>{s.goal_drift_score}</div>
            <div style={{ fontSize: 12, fontWeight: 600, color: driftColor, textTransform: 'uppercase' }}>{s.drift_severity} drift</div>
          </div>
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 }}>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Correction Rate</span><br/><strong>{pct(s.correction_rate_pct)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Agreement Rate</span><br/><strong>{pct(s.agreement_rate_pct)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Component Alignment</span><br/><strong>{pct(s.component_alignment_pct)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>HITL Override Rate</span><br/><strong>{pct(s.hitl_override_rate_pct)}</strong></div>
            <div><span style={{ fontSize: 11, color: '#64748b' }}>Block Rate</span><br/><strong>{pct(s.block_rate_pct)}</strong></div>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {[
          { label: 'Agent Actions', value: s.total_agent_actions, color: '#3b82f6' },
          { label: 'Active Components', value: s.active_components, color: '#10b981' },
          { label: 'Blocked Actions', value: s.blocked_actions, color: '#ef4444' },
          { label: 'Conversation Turns', value: s.conversation_turns, color: '#8b5cf6' },
          { label: 'Loop Ratio', value: s.loop_ratio, color: '#06b6d4' },
          { label: 'Avg Rating', value: s.avg_feedback_rating, color: '#f59e0b' },
          { label: 'Avg Confidence', value: s.avg_ai_confidence, color: '#ec4899' },
        ].map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
          </div>
        ))}
      </div>

      {/* Activity trend */}
      {daily.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Activity (Actions + Feedback)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={daily}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="actions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Agent Actions" />
              <Line type="monotone" dataKey="feedback" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} name="Feedback" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Component activity + Action types */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
        {compData.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Top Components by Actions</h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={compData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={110} />
                <Tooltip />
                <Bar dataKey="actions" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Actions" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {actionData.length > 0 && (
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Action Type Distribution</h4>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={actionData} dataKey="count" nameKey="action" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                  label={({ action, count }) => `${action}: ${count}`}>
                  {actionData.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Feedback rating distribution */}
      {ratingData.length > 0 && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Feedback Rating Distribution</h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={ratingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="rating" tick={{ fontSize: 11 }} label={{ value: 'Rating', position: 'bottom', fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Feedback Count" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )

  const renderComponents = () => (
    <>
      {components.length > 0 ? (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Component Loop Detail</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Component</th>
                  <th style={{ padding: '8px 10px' }}>Total Actions</th>
                  <th style={{ padding: '8px 10px' }}>Blocked</th>
                  <th style={{ padding: '8px 10px' }}>Block Rate</th>
                  <th style={{ padding: '8px 10px' }}>Action Breakdown</th>
                </tr>
              </thead>
              <tbody>
                {components.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.component}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(c.total)}</td>
                    <td style={{ padding: '8px 10px', color: c.blocked > 0 ? '#ef4444' : '#64748b' }}>{fmt(c.blocked)}</td>
                    <td style={{ padding: '8px 10px' }}>{pct(c.total > 0 ? Math.round(c.blocked / c.total * 100) : 0)}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>
                      {(c.actions || []).map(a => `${a.action}:${a.count}`).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div style={{ ...cardStyle, color: '#64748b', textAlign: 'center' }}>No component data available.</div>
      )}
    </>
  )

  const renderDrift = () => (
    <>
      {/* Corrections table */}
      <div style={cardStyle}>
        <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Recent Corrections (Goal-Drift Signals)</h4>
        {corrections.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Patient</th>
                  <th style={{ padding: '8px 10px' }}>Role</th>
                  <th style={{ padding: '8px 10px' }}>AI Output</th>
                  <th style={{ padding: '8px 10px' }}>Correction</th>
                  <th style={{ padding: '8px 10px' }}>Rating</th>
                  <th style={{ padding: '8px 10px' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {corrections.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{c.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{c.role}</td>
                    <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.ai_output}</td>
                    <td style={{ padding: '8px 10px', fontSize: 12, color: '#dc2626', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.correction}</td>
                    <td style={{ padding: '8px 10px' }}>{c.rating != null ? c.rating : '--'}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8' }}>{c.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: 20 }}>No corrections recorded.</div>
        )}
      </div>

      {/* Decision disagreements */}
      <div style={cardStyle}>
        <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Decision Disagreements (AI vs Neurologist)</h4>
        {disagreements.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 10px' }}>Patient</th>
                  <th style={{ padding: '8px 10px' }}>AI Prediction</th>
                  <th style={{ padding: '8px 10px' }}>Confidence</th>
                  <th style={{ padding: '8px 10px' }}>Final Decision</th>
                  <th style={{ padding: '8px 10px' }}>Reviewer</th>
                  <th style={{ padding: '8px 10px' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {disagreements.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 12 }}>{d.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ai_prediction}</td>
                    <td style={{ padding: '8px 10px', color: d.ai_confidence < 0.5 ? '#ef4444' : '#10b981' }}>
                      {d.ai_confidence != null ? `${(d.ai_confidence * 100).toFixed(1)}%` : '--'}
                    </td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{d.final_decision}</td>
                    <td style={{ padding: '8px 10px', fontSize: 12 }}>{d.reviewer}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8' }}>{d.date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: '#64748b', fontSize: 13, textAlign: 'center', padding: 20 }}>No disagreements recorded.</div>
        )}
      </div>
    </>
  )

  const renderDefinitions = () => (
    <div style={{ ...cardStyle, background: '#eff6ff', border: '1px solid #bfdbfe' }}>
      <h4 style={{ margin: '0 0 10px', color: '#1e40af' }}>Agent Loop / Goal-Drift Metric Definitions</h4>
      {(defs?.definitions || []).map((d, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <strong style={{ color: '#1e3a5f' }}>{d.metric}:</strong>{' '}
          <span style={{ color: '#334155', fontSize: 13 }}>{d.description}</span>
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Source: {d.source}</div>
        </div>
      ))}
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Components' },
    { id: 'drift', label: 'Goal Drift Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Agent Loop / Goal-Drift Dashboard</h2>
      </div>

      {/* Tab nav */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'components' && renderComponents()}
      {tab === 'drift' && renderDrift()}
      {tab === 'definitions' && renderDefinitions()}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: clinical.db (transaction_log, conversation_log, feedback, clinical_decisions, component_findings, hitl_reviews)
      </div>
    </div>
  )
}
