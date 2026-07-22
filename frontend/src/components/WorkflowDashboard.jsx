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

export default function WorkflowDashboard() {
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
          axios.get(`${API_URL}/api/workflow/overview`),
          axios.get(`${API_URL}/api/workflow/breakdown`),
          axios.get(`${API_URL}/api/workflow/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load workflow data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading workflow data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Workflow data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const actionData = (overview.action_distribution || []).slice(0, 10)
  const actorData = overview.actor_distribution || []
  const dailyData = overview.daily_trend || []
  const conversationRoles = overview.conversation_roles || []
  const roles = overview.roles || []
  const breakdownRoles = breakdown?.roles || []
  const componentActions = (breakdown?.component_actions || []).slice(0, 30)
  const recentEvents = breakdown?.recent_events || []
  const hourlyData = breakdown?.hourly_pattern || []
  const patientWorkflows = breakdown?.patient_workflows || []
  const workflowComponents = overview.workflow_components || []

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
      {/* KPIs */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
        {[
          { label: 'Total Roles', value: s.total_roles, color: '#3b82f6' },
          { label: 'Total Phases', value: s.total_phases, color: '#10b981' },
          { label: 'Total Steps', value: s.total_steps, color: '#8b5cf6' },
          { label: 'Sign-offs', value: s.total_signoffs, color: '#f59e0b' },
          { label: 'Workflow Events', value: s.workflow_events, color: '#06b6d4' },
          { label: 'Total Events', value: s.total_events, color: '#ec4899' },
          { label: 'HITL Reviews', value: s.hitl_reviews, color: '#64748b' },
          { label: 'Human Rate %', value: s.human_rate_pct, color: '#84cc16' },
        ].map((k, i) => (
          <div key={i} style={kpiStyle(k.color)}>
            <div style={{ fontSize: 11, color: '#64748b' }}>{k.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>{fmt(k.value)}</div>
          </div>
        ))}
      </div>

      {/* Daily trend bar chart */}
      {dailyData.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Daily Workflow Events</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Action distribution + Actor distribution side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
        {actionData.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Action Distribution</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={actionData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="action" type="category" tick={{ fontSize: 10 }} width={100} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {actorData.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Actor Distribution</div>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={actorData} dataKey="events" nameKey="actor" cx="50%" cy="50%" outerRadius={80} label={({ actor, percent }) => `${actor} ${(percent * 100).toFixed(0)}%`}>
                  {actorData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Conversation roles pie chart */}
      {conversationRoles.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Conversation Roles</div>
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={conversationRoles} dataKey="turns" nameKey="role" cx="50%" cy="50%" outerRadius={80} label={({ role, percent }) => `${role} ${(percent * 100).toFixed(0)}%`}>
                {conversationRoles.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )

  const renderRoles = () => {
    const roleStepData = breakdownRoles.map(r => ({ name: r.name, steps: r.total_steps }))
    return (
      <>
        {/* Role cards */}
        {breakdownRoles.map((r, i) => (
          <div key={i} style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 12 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>{r.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{r.summary}</div>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <div style={kpiStyle('#3b82f6')}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>Phases</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#3b82f6' }}>{fmt(r.total_phases)}</div>
                </div>
                <div style={kpiStyle('#10b981')}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>Steps</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#10b981' }}>{fmt(r.total_steps)}</div>
                </div>
                <div style={kpiStyle('#f59e0b')}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>Sign-offs</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: '#f59e0b' }}>{fmt(r.signoff_count)}</div>
                </div>
              </div>
            </div>
            {/* Signoff badges */}
            {(r.signoffs || []).length > 0 && (
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10 }}>
                {r.signoffs.map((so, j) => (
                  <span key={j} style={{ background: '#eff6ff', color: '#3b82f6', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 600 }}>{so}</span>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Bar chart comparing roles by steps */}
        {roleStepData.length > 0 && (
          <div style={cardStyle}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Steps per Role</div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={roleStepData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={120} />
                <Tooltip />
                <Bar dataKey="steps" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </>
    )
  }

  const renderActivity = () => (
    <>
      {/* Workflow components bar chart */}
      {workflowComponents.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Events per Component</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={workflowComponents} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip />
              <Bar dataKey="events" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Component actions table */}
      {componentActions.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Component x Action Breakdown</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  {['Component', 'Action', 'Count'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {componentActions.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.component}</td>
                    <td style={{ padding: '6px 10px' }}>{r.action}</td>
                    <td style={{ padding: '6px 10px' }}>{fmt(r.count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Hourly pattern bar chart */}
      {hourlyData.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Hourly Event Pattern (UTC)</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Recent events table */}
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Recent Workflow Events</div>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                {['Timestamp', 'Component', 'Action', 'Actor', 'Patient', 'Detail'].map(h => (
                  <th key={h} style={{ padding: '6px 8px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recentEvents.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px 8px', fontSize: 10, color: '#64748b' }}>{e.ts?.slice(0, 19)}</td>
                  <td style={{ padding: '4px 8px', fontWeight: 600 }}>{e.component}</td>
                  <td style={{ padding: '4px 8px' }}>
                    <span style={{ background: '#eff6ff', color: '#3b82f6', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 600 }}>{e.action}</span>
                  </td>
                  <td style={{ padding: '4px 8px' }}>{e.actor}</td>
                  <td style={{ padding: '4px 8px' }}>{e.patient_id || '--'}</td>
                  <td style={{ padding: '4px 8px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )

  const renderPatients = () => (
    <>
      {/* Patient workflows table */}
      <div style={cardStyle}>
        <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Patient Workflow Summary</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Components', 'Actions', 'Events'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patientWorkflows.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                  <td style={{ padding: '6px 10px' }}>{fmt(p.components)}</td>
                  <td style={{ padding: '6px 10px' }}>{fmt(p.actions)}</td>
                  <td style={{ padding: '6px 10px' }}>{fmt(p.events)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Patient workflows bar chart */}
      {patientWorkflows.length > 0 && (
        <div style={cardStyle}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Events per Patient</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={patientWorkflows}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="events" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  )

  const renderDefinitions = () => (
    <div style={cardStyle}>
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>Metric Definitions</div>
      {(defs?.definitions || []).map((d, i) => (
        <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < (defs?.definitions || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
          <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.metric}</div>
          <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{d.definition}</div>
        </div>
      ))}
    </div>
  )

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 18 }}>
        <h2 style={{ fontSize: 20, fontWeight: 800, margin: 0 }}>Workflow Dashboard</h2>
        <p style={{ fontSize: 12, color: '#64748b', margin: '4px 0 0' }}>
          Clinical workflow roles, phases &amp; sign-offs from clinical.db &mdash; {fmt(s.total_roles)} roles, {fmt(s.total_steps)} steps, {fmt(s.total_events)} events
        </p>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'roles', label: 'Roles' },
          { id: 'activity', label: 'Activity' },
          { id: 'patients', label: 'Patients' },
          { id: 'definitions', label: 'Definitions' },
        ].map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'roles' && renderRoles()}
      {tab === 'activity' && renderActivity()}
      {tab === 'patients' && renderPatients()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
