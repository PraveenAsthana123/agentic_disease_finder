import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#795548']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

export default function ExecutiveAIDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/executive-ai/overview`),
          axios.get(`${API_URL}/api/executive-ai/breakdown`),
          axios.get(`${API_URL}/api/executive-ai/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Executive AI data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Executive AI Dashboard...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Executive AI data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const actorDist = overview.actor_distribution || []
  const topComponents = overview.top_components || []
  const actionDist = overview.action_distribution || []
  const dailyThru = overview.daily_throughput || []
  const deptAI = overview.department_ai_utilization || []
  const convRoles = overview.conversation_roles || []
  const componentDetail = breakdown?.component_detail || []
  const deptComponent = breakdown?.department_component_cross || []
  const weeklyVol = breakdown?.weekly_volume || []
  const expertReviews = breakdown?.recent_expert_reviews || []
  const hitlReviews = breakdown?.recent_hitl_reviews || []
  const defsList = defs?.definitions || []

  const kpiCards = [
    { label: 'AI Operations', value: s.total_ai_operations, color: '#1e88e5', icon: '\u2699' },
    { label: 'Automation Rate', value: `${s.automation_rate_pct}%`, color: '#4caf50', icon: '\u26A1' },
    { label: 'AI Penetration', value: `${s.ai_penetration_pct}%`, color: '#7c4dff', icon: '\u2B50' },
    { label: 'Conversations', value: s.total_conversations, color: '#ff9800', icon: '\uD83D\uDCAC' },
    { label: 'Assessments', value: s.total_assessments, color: '#00bcd4', icon: '\uD83D\uDCCB' },
    { label: 'Expert Reviews', value: s.expert_reviews, color: '#e91e63', icon: '\uD83D\uDD0D' },
    { label: 'HITL Reviews', value: s.hitl_reviews, color: '#f44336', icon: '\uD83D\uDEE1' },
    { label: 'Oversight Rate', value: `${s.oversight_rate_pct}%`, color: s.oversight_rate_pct > 5 ? '#4caf50' : '#ff9800', icon: '\u2696' },
  ]

  // Radar data for AI health
  const radarData = [
    { metric: 'Automation', value: Math.min(s.automation_rate_pct, 100) },
    { metric: 'Penetration', value: Math.min(s.ai_penetration_pct, 100) },
    { metric: 'Oversight', value: Math.min(s.oversight_rate_pct * 10, 100) },
    { metric: 'Volume', value: Math.min(s.total_ai_operations / 10, 100) },
    { metric: 'Engagement', value: Math.min(s.total_conversations / 5, 100) },
    { metric: 'Assessments', value: Math.min(s.total_assessments / 5, 100) },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'components', label: 'Components' },
    { id: 'oversight', label: 'Oversight' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const cardStyle = (color) => ({
    background: '#fff',
    border: `2px solid ${color}22`,
    borderRadius: 12,
    padding: '16px 20px',
    textAlign: 'center',
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  })

  const sectionStyle = {
    background: '#fff',
    border: '1px solid #e2e8f0',
    borderRadius: 12,
    padding: 24,
    marginBottom: 20,
  }

  const tabBarStyle = {
    display: 'flex', gap: 4, marginBottom: 20, background: '#f1f5f9',
    borderRadius: 10, padding: 4,
  }

  const tabStyle = (active) => ({
    padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
    fontSize: 13, fontWeight: active ? 600 : 400,
    background: active ? '#fff' : 'transparent',
    color: active ? '#0f172a' : '#64748b',
    boxShadow: active ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
  })

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h2 style={{ margin: 0, color: '#0f172a' }}>Executive AI Dashboard</h2>
          <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 14 }}>
            AI adoption, utilization, and governance KPIs from real clinical.db
          </p>
        </div>
      </div>

      {/* Tab Bar */}
      <div style={tabBarStyle}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(activeTab === t.id)} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {activeTab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12, marginBottom: 24 }}>
            {kpiCards.map((k, i) => (
              <div key={i} style={cardStyle(k.color)}>
                <div style={{ fontSize: 22, marginBottom: 4 }}>{k.icon}</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: k.color }}>
                  {typeof k.value === 'number' ? fmt(k.value) : k.value}
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Row: AI Health Radar + Actor Distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>AI Health Radar</h3>
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar name="AI Health" dataKey="value" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>Actor Distribution</h3>
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={actorDist} cx="50%" cy="50%" outerRadius={90} dataKey="operations"
                    label={({ actor, operations }) => `${actor}: ${operations}`}>
                    {actorDist.map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* AI Throughput Trend (14 days) */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>AI Throughput - 14 Day Trend</h3>
            {dailyThru.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={dailyThru}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="ops" />
                  <YAxis yAxisId="comp" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="ops" type="monotone" dataKey="operations" name="Operations" stroke="#1e88e5" strokeWidth={2} dot={{ r: 3 }} />
                  <Line yAxisId="comp" type="monotone" dataKey="components_active" name="Active Components" stroke="#ff9800" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p style={{ color: '#94a3b8', textAlign: 'center' }}>No recent throughput data</p>
            )}
          </div>

          {/* Department AI Utilization */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Department AI Utilization</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={deptAI} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="department" tick={{ fontSize: 12 }} width={100} />
                <Tooltip />
                <Legend />
                <Bar dataKey="patients" name="Patients" fill="#1e88e5" radius={[0, 4, 4, 0]} />
                <Bar dataKey="ai_operations" name="AI Ops" fill="#4caf50" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Action Distribution + Conversation Roles */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>Action Type Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={actionDist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="action" tick={{ fontSize: 10, angle: -30 }} height={60} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="operations" fill="#7c4dff" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>Conversation Roles</h3>
              {convRoles.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie data={convRoles} cx="50%" cy="50%" outerRadius={80} dataKey="count"
                      label={({ role, count }) => `${role}: ${count}`}>
                      {convRoles.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p style={{ color: '#94a3b8', textAlign: 'center' }}>No conversation data</p>
              )}
            </div>
          </div>
        </>
      )}

      {/* ═══ COMPONENTS TAB ═══ */}
      {activeTab === 'components' && (
        <>
          {/* Top Components Bar */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Top AI Components by Volume</h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={topComponents}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 10, angle: -30 }} height={70} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="operations" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Component Detail Table */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Component Detail</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Component</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Operations</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Patients</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>First Seen</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {componentDetail.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{c.component}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(c.ops)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(c.patients_touched)}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{c.first_seen || '--'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{c.last_seen || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Weekly Volume Trend */}
          {weeklyVol.length > 0 && (
            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>Weekly AI Volume (8 weeks)</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={weeklyVol}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="ops" name="Operations" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="components" name="Components" fill="#ff9800" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Dept x Component Cross-tab */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Department x Component Cross-Tab</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Department</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Component</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Operations</th>
                  </tr>
                </thead>
                <tbody>
                  {deptComponent.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{d.dept}</td>
                      <td style={{ padding: '8px 12px' }}>{d.component}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(d.ops)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ═══ OVERSIGHT TAB ═══ */}
      {activeTab === 'oversight' && (
        <>
          {/* Oversight KPIs */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 24 }}>
            <div style={cardStyle('#1e88e5')}>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#1e88e5' }}>{fmt(s.total_ai_operations)}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>Total AI Ops</div>
            </div>
            <div style={cardStyle('#4caf50')}>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#4caf50' }}>{fmt(s.expert_reviews)}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>Expert Reviews</div>
            </div>
            <div style={cardStyle('#e91e63')}>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#e91e63' }}>{fmt(s.hitl_reviews)}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>HITL Reviews</div>
            </div>
            <div style={cardStyle(s.oversight_rate_pct > 5 ? '#4caf50' : '#ff9800')}>
              <div style={{ fontSize: 28, fontWeight: 700, color: s.oversight_rate_pct > 5 ? '#4caf50' : '#ff9800' }}>
                {s.oversight_rate_pct}%
              </div>
              <div style={{ fontSize: 12, color: '#64748b' }}>Oversight Rate</div>
            </div>
          </div>

          {/* Automation vs Human */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>System vs Human Operations</h3>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'System (Automated)', value: s.system_ops },
                      { name: 'Human-Initiated', value: s.human_ops },
                    ]}
                    cx="50%" cy="50%" outerRadius={80} dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    <Cell fill="#4caf50" />
                    <Cell fill="#1e88e5" />
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={sectionStyle}>
              <h3 style={{ marginTop: 0, color: '#0f172a' }}>Feedback & Engagement</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '20px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569' }}>Feedback Submissions</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: '#1e88e5' }}>{fmt(s.feedback_count)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569' }}>AI-Touched Patients</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: '#7c4dff' }}>{fmt(s.ai_touched_patients)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#475569' }}>Total Patients</span>
                  <span style={{ fontSize: 20, fontWeight: 700, color: '#4caf50' }}>{fmt(s.total_patients)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Recent Expert Reviews */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Recent Expert Reviews</h3>
            {expertReviews.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {Object.keys(expertReviews[0]).map(k => (
                      <th key={k} style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {expertReviews.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      {Object.values(r).map((v, j) => (
                        <td key={j} style={{ padding: '8px 12px', fontSize: 12 }}>{String(v ?? '--')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ color: '#94a3b8', textAlign: 'center' }}>No expert reviews recorded yet</p>
            )}
          </div>

          {/* Recent HITL Reviews */}
          <div style={sectionStyle}>
            <h3 style={{ marginTop: 0, color: '#0f172a' }}>Recent HITL Reviews</h3>
            {hitlReviews.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {Object.keys(hitlReviews[0]).map(k => (
                      <th key={k} style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {hitlReviews.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      {Object.values(r).map((v, j) => (
                        <td key={j} style={{ padding: '8px 12px', fontSize: 12 }}>{String(v ?? '--')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ color: '#94a3b8', textAlign: 'center' }}>No HITL reviews recorded yet</p>
            )}
          </div>
        </>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {activeTab === 'definitions' && (
        <div style={{ ...sectionStyle, background: '#f0f9ff', borderColor: '#bae6fd' }}>
          <h3 style={{ marginTop: 0, color: '#0c4a6e' }}>Metric Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
            {defsList.map((d, i) => (
              <div key={i} style={{ background: '#fff', borderRadius: 8, padding: 12, border: '1px solid #e0f2fe' }}>
                <strong style={{ color: '#0f172a' }}>{d.metric}</strong>
                <p style={{ margin: '4px 0 2px', fontSize: 13, color: '#475569' }}>{d.description}</p>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>Source: {d.source}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ textAlign: 'center', color: '#94a3b8', fontSize: 11, marginTop: 16 }}>
        System ops: {fmt(s.system_ops)} / {fmt(s.total_ai_operations)} total ({s.automation_rate_pct}% automated) | Data from clinical.db
      </div>
    </div>
  )
}
